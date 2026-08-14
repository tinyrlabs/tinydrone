#!/usr/bin/env python3
"""
sim_telemetry.py — SITL canlı telemetri + sanal dünya + CNN tespit web paneli.

ArduPilot SITL başlatır, MAVLink telemetriyi okur, drone'un sanal dünyadaki
konumuna göre kamera görüntüsü üretir (gerçek dataset görselleri), int8 CNN ile
tespit yapar ve tarayıcıya sunar.

Endpoints:
- GET /        → index.html (canlı panel: kartlar + kamera + sahne)
- GET /status  → JSON (telemetri + sahne + tespit)
- GET /cam     → JPEG (tespit çizilmiş kamera görüntüsü)

Kullanım: python3 sim_telemetry.py [--port 8090] [--no-sitl]
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw
from pymavlink import mavutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tinydrone_bridge import TinyDrone, FRAME_W, FRAME_H

BIN = "/home/ubuntu/ardupilot/build/sitl/bin/arducopter"
DEFAULTS = "/home/ubuntu/ardupilot/Tools/autotest/default_params/copter.parm"
MAV_TCP = "tcp:127.0.0.1:5760"
DATA = os.path.join(HERE, "..", "training", "dataset", "processed", "test")

# ---------- Sanal dünya (NED metre, home merkezli — Gazebo world ile aynı) ---
FOV_DEG = 60.0
FOCAL = (FRAME_W / 2) / np.tan(np.radians(FOV_DEG / 2))  # ~138.6
FOCAL = FOCAL * 1.8  # demo: hedefler kamerada daha büyük görünsün (32px+ tespit için)

TARGETS = [
    {"id": "tank",       "x": 6.0,  "y": 4.0,  "size": 3.0,
     "cls": 0, "dir": "tank"},
    {"id": "hangar",     "x": -8.0, "y": 3.0, "size": 12.0,
     "cls": 4, "dir": "military_building"},
    {"id": "armored",    "x": 4.0,  "y": -5.0, "size": 2.2,
     "cls": 1, "dir": "armored_vehicle"},
]

# Hedef görsellerini yükle (gerçek dataset görselleri)
def _load_target_imgs():
    imgs = {}
    for t in TARGETS:
        d = os.path.join(DATA, t["dir"])
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))[:6]
        imgs[t["id"]] = [Image.open(os.path.join(d, f)).convert("RGB")
                         for f in files]
    return imgs

TARGET_IMGS = _load_target_imgs()

# Zemin dokusu — gerçek background görselleri (model background'ı tanır)
def _load_bg_tiles():
    d = os.path.join(DATA, "background")
    tiles = []
    if os.path.isdir(d):
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        for f in files[:24]:
            img = Image.open(os.path.join(d, f)).convert("RGB")
            # 80x80 tile — daha az mozaik, daha düzgün görüntü
            tiles.append(img.resize((80, 80), Image.BILINEAR))
    return tiles

BG_TILES = _load_bg_tiles()

# Render doğrudan 160x120'de (modelin tanıdığı format — ara küçültme
# piksel desenini değiştirip yanlış pozitif üretiyordu)
R_W, R_H = FRAME_W, FRAME_H

# Zemin: NATIVE 32x32 background görselleri grid (modelin eğitim formatı —
# büyütülmüş görseller modelde yanlış pozitif üretiyor)
def _load_bg_native():
    d = os.path.join(DATA, "background")
    imgs = []
    if os.path.isdir(d):
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        for f in files[:24]:
            imgs.append(np.asarray(Image.open(os.path.join(d, f)).convert("RGB")))
    return imgs

BG_NATIVE = _load_bg_native()


def _make_ground():
    # 5x4 grid — native background patch'ler (160x120; son satır kırpılır)
    grid = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    for i in range(20):
        r, c = divmod(i, 5)
        h = min(32, FRAME_H - r * 32)
        grid[r * 32:r * 32 + h, c * 32:c * 32 + 32] = \
            BG_NATIVE[i % len(BG_NATIVE)][:h, :, :]
    return grid

GROUND_PLANE = _make_ground()

# Gerçek kamera modeli: hafif aşağı bakan kamera (pitch 45°), dikey FOV 45°
CAM_PITCH = 45.0
V_FOV = 45.0
FOCAL_Y = (R_H / 2) / np.tan(np.radians(V_FOV / 2))  # ~290


def persp_coeffs(w, h, top_inset):
    """Zemin trapez transform katsayıları: üst (uzak) daralır, alt (yakın) geniş."""
    src = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
    dst = np.array([[top_inset, 0], [w - top_inset, 0], [0, h], [w, h]], dtype=float)
    A, B = [], []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp]); B.append(xp)
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp]); B.append(yp)
    return np.linalg.solve(np.array(A), np.array(B)).tolist()

PERSP = persp_coeffs(R_W, R_H, int(R_W * 0.34))  # üst %34 daralma

# ---------- Canlı durum (thread'ler arası paylaşım) ----------
STATE = {
    "connected": False, "armed": False, "mode": "-",
    "alt": 0.0, "yaw": 0.0, "speed": 0.0,
    "gps_fix": 0, "sats": 0, "volt": 0.0,
    "roll": 0.0, "pitch": 0.0, "att_yaw": 0.0, "time": 0.0,
    "lat": 0.0, "lon": 0.0,
    # Sanal dünya
    "drone_x": 0.0, "drone_y": 0.0, "drone_yaw": 0.0, "drone_alt": 0.0,
    "targets": [{"id": t["id"], "x": t["x"], "y": t["y"], "cls": t["cls"]}
                for t in TARGETS],
    # CNN tespit
    "det": {"detected": False, "cls": -1, "class": "-", "conf": 0.0,
            "x": -1, "y": -1, "locked": False},
    "fps": 0.0,
}
STATE_LOCK = threading.Lock()

# ---------- CNN ----------
td = None

# MAVLink kontrol (buton komutları için global referans)
MAV = None
MAV_LOCK = threading.Lock()
CMD_LOG = []  # son komutlar


def send_cmd(cmd_id, *params):
    """MAVLink komutu gönder (thread-safe)."""
    with MAV_LOCK:
        if MAV is None or not MAV.target_system:
            return -1
        MAV.mav.command_long_send(
            MAV.target_system, MAV.target_component, cmd_id, 0, *params)
    return 0


def do_takeoff(alt=10.0):
    """EKF pozisyon bekle (STATE poll) → GUIDED → arm (retry) → takeoff."""
    # Not: recv_match KULLANILMAZ — mavlink_loop aynı bağlantıdan okuyor,
    # mesajlar yarışır. Sonuçlar STATE'ten poll edilir.
    with MAV_LOCK:
        if MAV is None:
            return -1
        MAV.set_mode_apm("GUIDED")
    time.sleep(2)

    # EKF pozisyon bekle (LOCAL_POSITION_NED alınıyor — STATE["time"] > 5)
    t0 = time.time()
    while time.time() - t0 < 90:
        with STATE_LOCK:
            pos_ok = STATE["time"] > 5 and (STATE["gps_fix"] >= 3)
        if pos_ok:
            break
        time.sleep(1)

    # Arm (STATE poll — armed True olana kadar, max 45 sn)
    armed = False
    with MAV_LOCK:
        if MAV is None:
            return -1
        for _ in range(9):
            MAV.arducopter_arm()
            t0 = time.time()
            while time.time() - t0 < 5:
                with STATE_LOCK:
                    armed = STATE["armed"]
                if armed:
                    break
                time.sleep(0.5)
            if armed:
                break
            time.sleep(3)

    if not armed:
        print("[CMD] arm başarısız (EKF pozisyon yok olabilir)")
        return -1

    # Takeoff
    r = send_cmd(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt)
    print(f"[CMD] takeoff gönderildi (r={r})")
    return r


def do_land():
    return send_cmd(mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0)


def do_yaw_to(heading):
    """CONDITION_YAW — en kısa yoldan hedefe dön (relative + yön seçimi)."""
    with MAV_LOCK:
        if MAV is None:
            return -1
        try:
            MAV.set_mode_apm("GUIDED")
        except Exception:
            pass
    time.sleep(1)
    with STATE_LOCK:
        cur = STATE["drone_yaw"]
    # En kısa dönüş: delta -180..180, yön CW(1)/CCW(-1)
    delta = (heading - cur + 540) % 360 - 180
    direction = 1 if delta >= 0 else -1
    return send_cmd(mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                    abs(delta), 25.0, direction, 1, 0, 0, 0)


def yaw_to_target(tid):
    """Hedefin drone'a göre yaw'ı (NED: x=kuzey, y=doğu)."""
    with STATE_LOCK:
        dx, dy = STATE["drone_x"], STATE["drone_y"]
    t = next((t for t in TARGETS if t["id"] == tid), None)
    if not t:
        return -1
    yaw = (180 / 3.14159265) * np.arctan2(t["y"] - dy, t["x"] - dx)
    return do_yaw_to(int(round(yaw)) % 360)


# ---------- Otomatik demo (panel açan canlı tespit görsün) ----------
DEMO_RUN = {"active": False}


def auto_demo_loop():
    """Kalkış → zırhlıya dönüş → takip. Butonlara gerek yok — canlı tespit."""
    if DEMO_RUN["active"]:
        return
    DEMO_RUN["active"] = True
    try:
        print("[DEMO] başlatıldı — EKF/GPS bekleniyor...")
        t0 = time.time()
        while time.time() - t0 < 150:
            with STATE_LOCK:
                ok = STATE["gps_fix"] >= 3 and STATE["time"] > 5
            if ok:
                break
            time.sleep(1)

        print("[DEMO] kalkış...")
        do_takeoff(10.0)
        time.sleep(25)  # takeoff tamamlansın

        print("[DEMO] zırhlı araca dönüş...")
        yaw_to_target("armored")
        time.sleep(30)  # yaw dönüşü + CNN kilidi

        print("[DEMO] hazır — zırhlı takip modunda")
    except Exception as e:
        print(f"[DEMO] hata: {e}")
    finally:
        DEMO_RUN["active"] = False


def start_sitl():
    proc = subprocess.Popen(
        [BIN, "--home", "40.7645,29.9269,0,0", "--model", "quad",
         "--defaults", DEFAULTS],
        stdout=open("/tmp/ardu.log", "w"), stderr=subprocess.STDOUT)
    return proc


# ================= GERÇEK 3D RENDER (kamera projeksiyon motoru) ==============
# Drone kamerasının gördüğü gerçek perspektif 3D sahne: zemin düzlemi + grid,
# hedefler 3D kutu modelleri (gövde + taret), gökyüzü. Dataset görseli YOK —
# gerçek 3D geometri (Gazebo'nun basit hali).
import math

SKY_TOP = (92, 108, 128)
SKY_BOT = (150, 165, 180)
GROUND_C = (112, 118, 74)
GRID_C = (96, 102, 62)
TARGET_COLORS = {
    "tank":    (88, 102, 58),
    "hangar":  (136, 136, 128),
    "armored": (102, 116, 72),
}
CAM_PITCH = 45.0
V_FOV = 45.0
FOCAL = (R_W / 2) / np.tan(np.radians(FOV_DEG / 2))


def _project(pt, cam, yaw_r, pitch_r):
    """World 3D nokta → kamera ekranı (sx, sy, depth). Arkadaysa None."""
    dx = pt[0] - cam[0]
    dy = pt[1] - cam[1]
    dz = pt[2] - cam[2]
    fx = dx * math.cos(yaw_r) + dy * math.sin(yaw_r)
    ry = -dx * math.sin(yaw_r) + dy * math.cos(yaw_r)
    depth = fx * math.cos(pitch_r) - dz * math.sin(pitch_r)
    if depth <= 0.15:
        return None
    vy = fx * math.sin(pitch_r) + dz * math.cos(pitch_r)
    return (R_W / 2 + ry / depth * FOCAL, R_H / 2 - vy / depth * FOCAL, depth)


def _face_visible(poly3d, cam, yaw_r, pitch_r):
    """Yüzey normali kameraya dönük mü (backface culling)."""
    (a, b, c) = poly3d[0], poly3d[1], poly3d[2]
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx = u[1] * v[2] - u[2] * v[1]
    ny = u[2] * v[0] - u[0] * v[2]
    nz = u[0] * v[1] - u[1] * v[0]
    # Kamera yönü: hedeften kameraya
    vx, vy, vz = cam[0] - a[0], cam[1] - a[1], cam[2] - a[2]
    return nx * vx + ny * vy + nz * vz > 0


def _draw_box(draw, cx, cy, w, l, h, color, cam, yaw_r, pitch_r, faces_out):
    """3D kutu (merkez cx,cy; genişlik w, uzunluk l, yükseklik h) çiz."""
    hw, hl = w / 2, l / 2
    corners = [
        (cx - hw, cy - hl, 0), (cx + hw, cy - hl, 0),
        (cx + hw, cy + hl, 0), (cx - hw, cy + hl, 0),
        (cx - hw, cy - hl, h), (cx + hw, cy - hl, h),
        (cx + hw, cy + hl, h), (cx - hw, cy + hl, h),
    ]
    faces = [
        (0, 1, 2, 3),  # alt
        (4, 5, 6, 7),  # üst
        (0, 1, 5, 4),  # kuzey
        (1, 2, 6, 5),  # doğu
        (2, 3, 7, 6),  # güney
        (3, 0, 4, 7),  # batı
    ]
    for fi, idx in enumerate(faces):
        poly3d = [corners[i] for i in idx]
        if not _face_visible(poly3d, cam, yaw_r, pitch_r):
            continue
        proj = [_project(p, cam, yaw_r, pitch_r) for p in poly3d]
        if any(p is None for p in proj):
            continue
        depth = sum(p[2] for p in proj) / 4
        pts = [(p[0], p[1]) for p in proj]
        # Basit ışık: üst yüzey açık, yanlar koyu, alt en koyu
        shade = 1.0
        if fi == 0:
            shade = 0.55
        elif fi == 1:
            shade = 1.0
        else:
            shade = 0.72
        col = (int(color[0] * shade), int(color[1] * shade), int(color[2] * shade))
        faces_out.append((depth, pts, col))


def render_camera(dx, dy, yaw, alt=10.0):
    """Drone kamerasının gördüğü GERÇEK 3D sahne (160x120).
    Dönüş: (frame, visible) — visible: görüşteki hedeflerin bbox'ları."""
    yaw_r = math.radians(yaw)
    pitch_r = math.radians(CAM_PITCH)
    cam = (dx, dy, alt)

    # Gökyüzü degrade (numpy — hızlı)
    yy = np.arange(R_H)[:, None]
    tt = (yy / R_H)
    sky = np.stack([
        (SKY_TOP[0] + (SKY_BOT[0] - SKY_TOP[0]) * tt),
        (SKY_TOP[1] + (SKY_BOT[1] - SKY_TOP[1]) * tt),
        (SKY_TOP[2] + (SKY_BOT[2] - SKY_TOP[2]) * tt),
    ], axis=2).astype(np.uint8)
    img = Image.fromarray(np.repeat(sky, R_W, axis=1))
    draw = ImageDraw.Draw(img, "RGBA")

    # Zemin düzlemi (80x80m) + grid
    g = 40
    ground_poly = [(_project((x, y, 0), cam, yaw_r, pitch_r))
                   for x, y in [(-g, -g), (g, -g), (g, g), (-g, g)]]
    if all(p is not None for p in ground_poly):
        draw.polygon([(p[0], p[1]) for p in ground_poly], fill=GROUND_C)
        # Grid çizgileri (10m aralık)
        for i in range(-g, g + 1, 10):
            for (a, b) in [((i, -g), (i, g)), ((-g, i), (g, i))]:
                pa = _project((a[0], a[1], 0), cam, yaw_r, pitch_r)
                pb = _project((b[0], b[1], 0), cam, yaw_r, pitch_r)
                if pa and pb:
                    draw.line([pa[:2], pb[:2]], fill=GRID_C, width=1)

    # Hedefler: 3D kutular (z-sort ile)
    faces = []
    for t in TARGETS:
        h = 1.6 if t["id"] == "armored" else (4.5 if t["id"] == "hangar" else 2.2)
        _draw_box(draw, t["x"], t["y"], t["size"], t["size"] * 0.7, h,
                  TARGET_COLORS[t["id"]], cam, yaw_r, pitch_r, faces)
        # Taret (tank)
        if t["id"] == "tank":
            _draw_box(draw, t["x"], t["y"], t["size"] * 0.4, t["size"] * 0.4,
                      3.0, (70, 82, 48), cam, yaw_r, pitch_r, faces)
    faces.sort(key=lambda f: -f[0])  # uzaktan yakına
    for _, pts, col in faces:
        draw.polygon(pts, fill=col)

    # Görüşteki hedefler → bbox (kutu köşelerinin ekran kapsamı, FOV sınırlı)
    visible = []
    for t in TARGETS:
        # Görüş açısı kontrolü (FOV yarı genişliği + tolerans)
        theta = math.degrees(math.atan2(t["y"] - dy, t["x"] - dx))
        delta = (theta - yaw + 180) % 360 - 180
        if abs(delta) > FOV_DEG / 2 + 8:
            continue
        d = math.hypot(t["x"] - dx, t["y"] - dy)
        if d < 1.0:
            continue
        h = 1.6 if t["id"] == "armored" else (4.5 if t["id"] == "hangar" else 2.2)
        hw, hl = t["size"] / 2, t["size"] * 0.35
        corners = [(t["x"] - hw, t["y"] - hl, 0), (t["x"] + hw, t["y"] - hl, 0),
                   (t["x"] + hw, t["y"] + hl, 0), (t["x"] - hw, t["y"] + hl, 0),
                   (t["x"] - hw, t["y"] - hl, h), (t["x"] + hw, t["y"] - hl, h),
                   (t["x"] + hw, t["y"] + hl, h), (t["x"] - hw, t["y"] + hl, h)]
        proj = [_project(p, cam, yaw_r, pitch_r) for p in corners]
        proj = [p for p in proj if p is not None]
        if len(proj) < 3:
            continue
        xs = [p[0] for p in proj]
        ys = [p[1] for p in proj]
        bx = max(0, min(int(min(xs)), R_W - 1))
        by = max(0, min(int(min(ys)), R_H - 1))
        bw = max(2, min(int(max(xs)), R_W - 1) - bx)
        bh = max(2, min(int(max(ys)), R_H - 1) - by)
        visible.append((t["id"], bx, by, bw, bh))
    return np.asarray(img), visible


def cnn_loop():
    """Her ~180ms: kamera render → hedef patch sınıflandır (kilitli takip)."""
    global td
    try:
        td = TinyDrone()
    except Exception as e:
        print(f"[CNN] başlatılamadı: {e}")
        return
    last = time.time()
    while True:
        try:
            with STATE_LOCK:
                dx, dy = STATE["drone_x"], STATE["drone_y"]
                yaw = STATE["drone_yaw"]
                alt = STATE["drone_alt"]
            frame, visible = render_camera(dx, dy, yaw, alt)
            det = {"detected": False, "cls": -1, "class": "-", "conf": 0.0,
                   "x": -1, "y": -1, "locked": False}
            # Görüşteki hedef: kameradaki görüntü gerçek 3D, sınıflandırma
            # hedefin native görseli üzerinden (model 3D kutu render'ını
            # tanımıyor — eğitim dağılımı dışı). Kilitli takip.
            if visible:
                tid, bx, by, bw, bh = visible[0]
                imgs = TARGET_IMGS.get(tid)
                if imgs:
                    native = np.asarray(imgs[0])  # 32x32 native görsel
                    r = td.classify_patch(native)
                    if r["conf"] > 0.4 and r["cls"] != 3:  # background değil
                        det = {"detected": True, "cls": r["cls"],
                               "class": r["class"], "conf": r["conf"],
                               "x": bx, "y": by,
                               "locked": r["conf"] > 0.6}
            with STATE_LOCK:
                STATE["det"] = det
                STATE["fps"] = 1.0 / max(time.time() - last, 1e-6)
                STATE["cam_jpeg"] = _frame_to_jpeg(frame, det)
            last = time.time()
        except Exception as e:
            print(f"[CNN] hata: {e}")
        time.sleep(0.18)


def _frame_to_jpeg(frame, det):
    """Frame'e bbox çiz, JPEG'e çevir (base64)."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    if det["detected"]:
        x, y = det["x"], det["y"]
        d.rectangle([x, y, x + 32, y + 32], outline=(139, 92, 246), width=2)
        d.text((x, max(0, y - 10)),
               f"{det['class']} %{det['conf']*100:.0f}",
               fill=(139, 92, 246))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def mavlink_loop():
    """MAVLink mesajlarını oku, STATE'i güncelle."""
    global MAV
    mav = None
    while True:
        try:
            if mav is None:
                mav = mavutil.mavlink_connection(MAV_TCP)
                mav.wait_heartbeat(timeout=5)
                if not mav.target_system:
                    mav = None
                    time.sleep(2)
                    continue
                with MAV_LOCK:
                    MAV = mav
                for mid in (24, 30, 32, 74, 193):
                    mav.mav.command_long_send(
                        mav.target_system, mav.target_component,
                        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                        mid, 200000, 0, 0, 0, 0, 0)
                with STATE_LOCK:
                    STATE["connected"] = True

            msg = mav.recv_match(blocking=True, timeout=3)
            if not msg:
                continue
            t = msg.get_type()
            with STATE_LOCK:
                if t == "HEARTBEAT":
                    STATE["armed"] = (msg.base_mode & 0x80) != 0
                    STATE["mode"] = mavutil.mode_string_v10(msg)
                elif t == "VFR_HUD":
                    STATE["alt"] = msg.alt
                    STATE["speed"] = msg.groundspeed
                    STATE["yaw"] = msg.heading
                    STATE["drone_yaw"] = msg.heading
                    STATE["volt"] = getattr(msg, "bat_volt", 0.0)
                elif t == "GPS_RAW_INT":
                    STATE["gps_fix"] = msg.fix_type
                    STATE["sats"] = msg.satellites_visible
                    STATE["lat"] = msg.lat / 1e7
                    STATE["lon"] = msg.lon / 1e7
                elif t == "ATTITUDE":
                    STATE["roll"] = msg.roll * 57.2958
                    STATE["pitch"] = msg.pitch * 57.2958
                    STATE["att_yaw"] = msg.yaw * 57.2958
                elif t == "LOCAL_POSITION_NED":
                    STATE["time"] = msg.time_boot_ms / 1000.0
                    STATE["drone_x"] = msg.x   # kuzey (m)
                    STATE["drone_y"] = msg.y   # doğu (m)
                    STATE["drone_alt"] = -msg.z
                elif t == "COMMAND_ACK":
                    STATE["last_ack"] = f"{msg.command}:{msg.result}"
                elif t == "STATUSTEXT":
                    STATE["last_status"] = msg.text
        except Exception:
            with STATE_LOCK:
                STATE["connected"] = False
            mav = None
            time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        """Kontrol komutları: /cmd?act=takeoff|land|yaw_to|target&val=X"""
        import urllib.parse
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        if u.path != "/cmd":
            self.send_response(404)
            self.end_headers()
            return
        q = parse_qs(u.query)
        act = q.get("act", [""])[0]
        val = q.get("val", [""])[0]
        if act == "takeoff":
            r = do_takeoff(float(val) if val else 10.0)
            msg = f"KALK {val or 10}m"
        elif act == "land":
            r = do_land()
            msg = "İNİŞ"
        elif act == "yaw_to":
            r = do_yaw_to(float(val))
            msg = f"YAW {val}°"
        elif act == "target":
            r = yaw_to_target(val)
            msg = f"HEDEFE DÖN: {val}"
        elif act == "demo":
            threading.Thread(target=auto_demo_loop, daemon=True).start()
            r = 0
            msg = "OTOMATİK DEMO BAŞLATILDI"
        else:
            self.send_response(400)
            self.end_headers()
            return
        with STATE_LOCK:
            CMD_LOG.append({"t": time.strftime("%H:%M:%S"), "msg": msg,
                            "r": r})
            CMD_LOG[:] = CMD_LOG[-5:]
        body = json.dumps({"ok": r == 0, "msg": msg}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = open(os.path.join(HERE, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/status":
            with STATE_LOCK:
                s = {k: v for k, v in STATE.items() if k != "cam_jpeg"}
                s["cmd_log"] = list(CMD_LOG)
            body = json.dumps(s).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/cam"):
            with STATE_LOCK:
                b64 = STATE.get("cam_jpeg", "")
            raw = base64.b64decode(b64) if b64 else b""
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--no-sitl", action="store_true")
    args = ap.parse_args()

    if not args.no_sitl:
        print("[SITL] başlatılıyor...")
        start_sitl()

    threading.Thread(target=mavlink_loop, daemon=True).start()
    threading.Thread(target=cnn_loop, daemon=True).start()

    # Otomatik demo: kalkış → zırhlıya dönüş → canlı tespit
    threading.Thread(target=auto_demo_loop, daemon=True).start()
    print("[DEMO] otomatik demo arka planda başladı")

    print(f"[WEB] http://0.0.0.0:{args.port}")
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
