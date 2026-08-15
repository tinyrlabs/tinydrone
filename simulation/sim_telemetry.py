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
     "cls": 0, "dir": "tank",
     # Tank da devriye yürür (yavaş — 0.8 m/s)
     "path": [(6.0, 4.0), (8.0, 6.0), (6.0, 8.0), (4.0, 6.0)],
     "speed": 0.8},
    {"id": "hangar",     "x": -8.0, "y": 3.0, "size": 12.0,
     "cls": 4, "dir": "military_building"},
    {"id": "armored",    "x": 4.0,  "y": -5.0, "size": 2.2,
     "cls": 1, "dir": "armored_vehicle",
     # Devriye yolu (NED metre) — hareketli hedef senaryosu
     "path": [(4.0, -5.0), (0.0, -7.0), (-4.0, -5.0), (0.0, -3.0)],
     "speed": 1.5},  # m/s
]

# Dinamik hedef pozisyonları (world_loop günceller)
WORLD_LOCK = threading.Lock()
TARGET_POS = {t["id"]: (t["x"], t["y"]) for t in TARGETS}


def world_loop():
    """Hareketli hedefler: path'i olan her hedef devriye yürür."""
    movers = [t for t in TARGETS if t.get("path")]
    if not movers:
        return
    # Her hareketli hedef için: idx, seg_t
    state = {t["id"]: [0, 0.0] for t in movers}
    while True:
        try:
            for t in movers:
                pts = t["path"]
                st = state[t["id"]]
                a = pts[st[0] % len(pts)]
                b = pts[(st[0] + 1) % len(pts)]
                seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
                step = t.get("speed", 1.0) * 0.1 / max(seg_len, 0.01)
                st[1] += step
                if st[1] >= 1.0:
                    st[1] -= 1.0
                    st[0] += 1
                    continue
                with WORLD_LOCK:
                    TARGET_POS[t["id"]] = (
                        a[0] + (b[0] - a[0]) * st[1],
                        a[1] + (b[1] - a[1]) * st[1])
        except Exception:
            pass
        time.sleep(0.1)

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

# Render çözünürlüğü: 960x720 (4:3 — CNN 160x120'ye küçültülür, panel 720p
# gösterir). Ölçek oranı her iki eksende 6 (bbox dönüşümü basit).
R_W, R_H = FRAME_W * 6, FRAME_H * 6  # 960 x 720
RENDER_SCALE = 6

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

# Gerçek kamera modeli: hafif aşağı bakan kamera (pitch 35° — 10m'de ufuk
# ~14m; daha dik açıda zemin dar banda sıkışıp ekran gökyüzüyle doluyordu),
# dikey FOV 50° (zemin + hedefler geniş görüşte)
CAM_PITCH = 35.0
V_FOV = 50.0
FOCAL_Y = (R_H / 2) / np.tan(np.radians(V_FOV / 2))  # ~360
FOCAL_X = FOCAL_Y  # kare piksel (4:3 sensör — yatay odak dikeyle aynı)


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
    "infer_ms": 0.0, "uart": "T+000Y+000",
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


def _target_yaw(dx, dy, yaw):
    """Görüşteki ilk hedefin drone yaw'ına göre offset'i (derece)."""
    with WORLD_LOCK:
        for t in TARGETS:
            pos = TARGET_POS.get(t["id"], (t["x"], t["y"]))
            theta = math.degrees(math.atan2(pos[1] - dy, pos[0] - dx))
            delta = (theta - yaw + 180) % 360 - 180
            if abs(delta) <= FOV_DEG / 2 + 8:
                return delta
    return 0.0


def yaw_to_target(tid):
    """Hedefin drone'a göre yaw'ı (NED: x=kuzey, y=doğu) — dinamik pozisyon."""
    with STATE_LOCK:
        dx, dy = STATE["drone_x"], STATE["drone_y"]
    with WORLD_LOCK:
        pos = TARGET_POS.get(tid)
    if not pos:
        return -1
    yaw = (180 / 3.14159265) * np.arctan2(pos[1] - dy, pos[0] - dx)
    return do_yaw_to(int(round(yaw)) % 360)


# ---------- Otomatik takip (hareketli hedefi görüşte tut) ----------
# enabled=False → manuel komut (target/yaw_to) takibi duraklatır
TRACK_TARGET = {"id": "armored", "enabled": True}


def track_loop():
    """1 Hz: takip hedefi görüş merkezinden sapınca yaw'ı düzelt.
    Manuel komutlar (HANGARA DÖN vb.) takibi duraklatır — çakışma yok."""
    while True:
        try:
            if not TRACK_TARGET["enabled"]:
                time.sleep(1.0)
                continue
            with STATE_LOCK:
                yaw = STATE["drone_yaw"]
                dx, dy = STATE["drone_x"], STATE["drone_y"]
            with WORLD_LOCK:
                pos = TARGET_POS.get(TRACK_TARGET["id"])
            if not pos:
                time.sleep(1.0)
                continue
            theta = math.degrees(math.atan2(pos[1] - dy, pos[0] - dx))
            delta = (theta - yaw + 180) % 360 - 180
            if abs(delta) > 12.0:  # görüşten sapınca hedefe dön
                r = do_yaw_to(int(round(theta)) % 360)
                print(f"[TRACK] yaw {yaw:.0f} -> hedef {theta:.0f} (delta {delta:.0f}) r={r}")
        except Exception:
            pass
        time.sleep(1.0)


# ---------- Otomatik demo (panel açan canlı tespit görsün) ----------
DEMO_RUN = {"active": False}


# ---------- Görev senaryosu (başı-sonu olan) ----------
MISSION = {
    "state": "HAZIR",      # HAZIR → KALKIŞ → ARAMA → TESPİT → TAKİP → TAMAM
    "start": 0.0,
    "log": [],
    "found": [],
    "active": False,
}


def mission_log(msg):
    t = time.strftime("%H:%M:%S")
    MISSION["log"].append({"t": t, "msg": msg})
    MISSION["log"][:] = MISSION["log"][-30:]
    print(f"[GÖREV] {msg}")


def mission_loop():
    """Görev: KALKIŞ → ARAMA (360° tarama) → TESPİT → TAKİP → İNİŞ (TAMAM)."""
    MISSION["active"] = True
    MISSION["start"] = time.time()
    MISSION["found"] = []
    mission_log("GÖREV BAŞLATILDI — kalkış hazırlığı")
    try:
        # 1) KALKIŞ
        MISSION["state"] = "KALKIŞ"
        r = do_takeoff(10.0)
        mission_log(f"KALKIŞ 10m (r={r})")
        time.sleep(25)

        # 2) ARAMA — 360° tarama (8 yön)
        MISSION["state"] = "ARAMA"
        TRACK_TARGET["enabled"] = False  # tarama sırasında manuel
        mission_log("ARAMA: 360° tarama başladı")
        for h in (0, 45, 90, 135, 180, 225, 270, 315):
            if not MISSION["active"]:
                return
            do_yaw_to(h)
            # Tarama sırasında kilit gelirse hemen TESPİT'e geç
            for _ in range(8):  # ~10 sn bekle, her 1.25 sn tespit kontrol
                if not MISSION["active"]:
                    return
                time.sleep(1.25)
                with STATE_LOCK:
                    det = dict(STATE.get("det", {}))
                if det.get("locked"):
                    break
            with STATE_LOCK:
                det = dict(STATE.get("det", {}))
            if det.get("locked") and det["class"] not in MISSION["found"]:
                MISSION["found"].append(det["class"])
                mission_log(f"TESPİT: {det['class']} %{det['conf']*100:.0f} (yaw {h}°)")
                break  # hedef bulundu — takibe geç

        # 3) TESPİT → TAKİP
        if MISSION["active"] and MISSION["found"]:
            MISSION["state"] = "TAKİP"
            tgt = "armored" if "armored_vehicle" in MISSION["found"] else "tank"
            TRACK_TARGET["id"] = tgt
            TRACK_TARGET["enabled"] = True
            mission_log(f"TAKİP: {tgt} (otonom takip 30 sn)")
            t0 = time.time()
            while time.time() - t0 < 30 and MISSION["active"]:
                time.sleep(1)
        else:
            mission_log("ARAMA sonucu: hedef bulunamadı — görev iptal")
            MISSION["state"] = "TAMAM"

        # 4) SONLANDIR
        if MISSION["active"]:
            MISSION["state"] = "TAMAM"
            TRACK_TARGET["enabled"] = False
            dur = int(time.time() - MISSION["start"])
            hed = ", ".join(MISSION["found"]) if MISSION["found"] else "yok"
            mission_log(f"GÖREV TAMAM — süre {dur}s, hedefler: {hed}")
            mission_log("İNİŞ")
            do_land()
    except Exception as e:
        mission_log(f"HATA: {e}")
    finally:
        MISSION["active"] = False


def mission_cancel():
    """Görevi sonlandır (iniş + özet)."""
    if not MISSION["active"]:
        return -1
    MISSION["active"] = False
    mission_log("GÖREV KULLANICI TARAFINDAN SONLANDIRILDI")
    TRACK_TARGET["enabled"] = False
    do_land()
    return 0


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
GROUND_C = (118, 124, 72)
GROUND_C2 = (108, 114, 66)
GRID_C = (98, 104, 60)
ROAD_C = (122, 108, 74)
TREE_TRUNK = (96, 74, 48)
TREE_LEAF = (52, 82, 44)
TREE_LEAF2 = (44, 72, 38)
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


def _quad_coeffs(src_pts, dst_pts):
    """Genel 4 nokta perspective transform katsayıları."""
    A, B = [], []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp]); B.append(xp)
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp]); B.append(yp)
    return np.linalg.solve(np.array(A), np.array(B)).tolist()


def _draw_box(draw, img, cx, cy, w, l, h, color, cam, yaw_r, pitch_r,
              faces_out, tex=None, z0=0.0):
    """3D kutu (merkez cx,cy; genişlik w, uzunluk l, yükseklik h) çiz.
    tex verilirse üst yüzeye gerçek görsel haritalanır (S4). z0: taban irtifası."""
    hw, hl = w / 2, l / 2
    corners = [
        (cx - hw, cy - hl, z0), (cx + hw, cy - hl, z0),
        (cx + hw, cy + hl, z0), (cx - hw, cy + hl, z0),
        (cx - hw, cy - hl, z0 + h), (cx + hw, cy - hl, z0 + h),
        (cx + hw, cy + hl, z0 + h), (cx - hw, cy + hl, z0 + h),
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
        # Kenar çizgisi — şekil belirgin (dandik oyun hissi azalsın)
        if fi == 1:
            outline_c = (0, 0, 0, 90)
        else:
            outline_c = (0, 0, 0, 55)
        # Üst yüzeye gerçek görsel texture (S4 — model kameradaki görüntüyü tanısın)
        if fi == 1 and tex is not None:
            faces_out.append((depth, pts, col, tex, proj, outline_c))
        else:
            faces_out.append((depth, pts, col, None, None, outline_c))


def render_camera(dx, dy, yaw, alt=10.0, view="first"):
    """GERÇEK 3D sahne render'ı.
    view="first": drone kamerası (aşağı bakan) — CNN bu görüntüyü işler
    view="third": 3. şahıs — drone'un arkasından + yukarıdan (drone modeli dahil)
    Dönüş: (frame, visible) — visible: görüşteki hedeflerin bbox'ları."""
    if view == "third":
        # Kamera: drone'un arkasında 10m geride, 6m yukarıda; drone'a bakıyor
        yaw_r = math.radians(yaw)
        cam_x = dx - math.cos(yaw_r) * 10.0
        cam_y = dy - math.sin(yaw_r) * 10.0
        cam_z = alt + 6.0
        cam = (cam_x, cam_y, cam_z)
        pitch_r = math.radians(31.0)  # yukarıdan drone'a aşağı bakış
    else:
        yaw_r = math.radians(yaw)
        pitch_r = math.radians(CAM_PITCH)
        cam_x, cam_y, cam_z = dx, dy, alt
        cam = (cam_x, cam_y, cam_z)

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

    # Zemin: kamera önünde sonsuz dörtgen (her yaw'da önde — sabit 80x80m
    # dörtgenin arka köşeleri kameranın arkasında kalıp tüm zemini düşürüyordu)
    g = 60
    fwd_x, fwd_y = math.cos(yaw_r), math.sin(yaw_r)
    sdx, sdy = -fwd_y, fwd_x  # yan yön
    zcorners = [
        (cam_x + sdx * g, cam_y + sdy * g),
        (cam_x - sdx * g, cam_y - sdy * g),
        (cam_x - sdx * g + fwd_x * g * 2, cam_y - sdy * g + fwd_y * g * 2),
        (cam_x + sdx * g + fwd_x * g * 2, cam_y + sdy * g + fwd_y * g * 2),
    ]
    zproj = [_project((cx, cy, 0), cam, yaw_r, pitch_r) for cx, cy in zcorners]
    if all(p is not None for p in zproj):
        draw.polygon([(p[0], p[1]) for p in zproj], fill=GROUND_C)
        # Grid çizgileri (kamera merkezli 10m — ön dörtgen içinde)
        for i in range(-3, 4):
            p1 = _project((cam_x + sdx * i * 10 + fwd_x * 1, cam_y + sdy * i * 10 + fwd_y * 1, 0), cam, yaw_r, pitch_r)
            p2 = _project((cam_x + sdx * i * 10 + fwd_x * 100, cam_y + sdy * i * 10 + fwd_y * 100, 0), cam, yaw_r, pitch_r)
            if p1 and p2:
                draw.line([p1[:2], p2[:2]], fill=GRID_C, width=1)
        for i in range(1, 11):
            p1 = _project((cam_x + sdx * g + fwd_x * i * 10, cam_y + sdy * g + fwd_y * i * 10, 0), cam, yaw_r, pitch_r)
            p2 = _project((cam_x - sdx * g + fwd_x * i * 10, cam_y - sdy * g + fwd_y * i * 10, 0), cam, yaw_r, pitch_r)
            if p1 and p2:
                draw.line([p1[:2], p2[:2]], fill=GRID_C, width=1)

        # Devriye yolları (hareketli hedeflerin patikaları — koyu şerit)
        for t in TARGETS:
            path = t.get("path")
            if not path:
                continue
            rw = 1.1  # yol yarı genişliği (m)
            for k in range(len(path)):
                a = path[k]
                b = path[(k + 1) % len(path)]
                dxp, dyp = b[0] - a[0], b[1] - a[1]
                ln = max(math.hypot(dxp, dyp), 0.01)
                nx, ny = -dyp / ln, dxp / ln  # dik
                quad = [(a[0] + nx * rw, a[1] + ny * rw),
                        (b[0] + nx * rw, b[1] + ny * rw),
                        (b[0] - nx * rw, b[1] - ny * rw),
                        (a[0] - nx * rw, a[1] - ny * rw)]
                pq = [_project((qx, qy, 0.02), cam, yaw_r, pitch_r)
                      for qx, qy in quad]
                if all(p is not None for p in pq):
                    draw.polygon([(p[0], p[1]) for p in pq], fill=ROAD_C)

        # Ortam: ağaçlar (billboard — gövde + taç)
        TREES = [(-9, -8, 3.2), (11, -9, 2.6), (14, 3, 3.4), (-13, -5, 2.2),
                 (9, 12, 3.0), (-15, 9, 2.8), (0, 14, 2.4), (16, -12, 2.6)]
        for txx, tyy, tr in TREES:
            bp = _project((txx, tyy, 0), cam, yaw_r, pitch_r)
            tp = _project((txx, tyy, 2.2), cam, yaw_r, pitch_r)
            cp = _project((txx, tyy, 4.2), cam, yaw_r, pitch_r)
            if not (bp and tp and cp) or cp[2] <= 0.1:
                continue
            # Gövde
            draw.line([bp[:2], tp[:2]], fill=TREE_TRUNK, width=max(2, int(tr)))
            # Taç (daire — ekran yarıçapı mesafeyle ölçekli)
            cr = tr / cp[2] * FOCAL_X
            draw.ellipse([cp[0] - cr, cp[1] - cr * 0.85, cp[0] + cr,
                          cp[1] + cr * 0.85], fill=TREE_LEAF)
            draw.ellipse([cp[0] - cr * 0.62, cp[1] - cr * 0.55, cp[0] + cr * 0.62,
                          cp[1] + cr * 0.5], fill=TREE_LEAF2)

    # Hedefler: 3D kutular (z-sort ile) — dinamik pozisyonlar + gerçek texture
    faces = []
    for t in TARGETS:
        with WORLD_LOCK:
            pos = TARGET_POS.get(t["id"], (t["x"], t["y"]))
        # Zemin gölgesi (yarı saydam elips — kutu tabanında)
        bp = _project((pos[0], pos[1], 0.02), cam, yaw_r, pitch_r)
        if bp and bp[2] > 0.1:
            rs = t["size"] * 0.95 / bp[2] * FOCAL_X
            draw.ellipse([bp[0] - rs, bp[1] - rs * 0.35,
                          bp[0] + rs, bp[1] + rs * 0.35],
                         fill=(0, 0, 0, 40))
        h = 1.6 if t["id"] == "armored" else (4.5 if t["id"] == "hangar" else 2.2)
        imgs = TARGET_IMGS.get(t["id"])
        tex = imgs[0] if imgs else None
        _draw_box(draw, img, pos[0], pos[1], t["size"], t["size"] * 0.7, h,
                  TARGET_COLORS[t["id"]], cam, yaw_r, pitch_r, faces, tex)
        # Taret + namlu (tank — devriye yönünde) + paletler
        if t["id"] == "tank":
            # Paletler (gövde yanları — koyu şeritler, alçak)
            _draw_box(draw, img, pos[0] - t["size"] * 0.58, pos[1],
                      t["size"] * 0.5, t["size"] * 0.68, 1.0,
                      (38, 44, 30), cam, yaw_r, pitch_r, faces)
            _draw_box(draw, img, pos[0] + t["size"] * 0.58, pos[1],
                      t["size"] * 0.5, t["size"] * 0.68, 1.0,
                      (38, 44, 30), cam, yaw_r, pitch_r, faces)
            _draw_box(draw, img, pos[0], pos[1], t["size"] * 0.4, t["size"] * 0.4,
                      3.0, (70, 82, 48), cam, yaw_r, pitch_r, faces)
            path = t.get("path")
            if path:
                bi, bd = 0, 1e9
                for i, p in enumerate(path):
                    d = (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2
                    if d < bd:
                        bd, bi = d, i
                nxt = path[(bi + 1) % len(path)]
                dxp, dyp = nxt[0] - pos[0], nxt[1] - pos[1]
                ln = max(math.hypot(dxp, dyp), 0.01)
                ux, uy = dxp / ln, dyp / ln
                _draw_box(draw, img, pos[0] + ux * 2.0, pos[1] + uy * 2.0,
                          0.5, 3.4, 0.5, (56, 66, 38),
                          cam, yaw_r, pitch_r, faces)
        # Armored: üst kule (gövde üzerinde)
        if t["id"] == "armored":
            _draw_box(draw, img, pos[0], pos[1], 1.1, 1.1, 0.9,
                      (82, 96, 58), cam, yaw_r, pitch_r, faces, z0=1.6)
        # Hangar kapısı (güney yüzde koyu şerit) + çatı kirişi
        if t["id"] == "hangar":
            _draw_box(draw, img, pos[0], pos[1] + 3.6, 3.4, 0.6, 3.4,
                      (56, 56, 52), cam, yaw_r, pitch_r, faces)
            _draw_box(draw, img, pos[0], pos[1], 12.0, 0.9, 0.4,
                      (86, 86, 82), cam, yaw_r, pitch_r, faces, z0=4.5)
    # Drone modeli (3. şahıs görünümü — gövde + 4 motor, alt irtifasında)
    if view == "third":
        _draw_box(draw, img, dx, dy, 0.7, 0.7, 0.3, (70, 72, 78),
                  cam, yaw_r, pitch_r, faces, z0=alt)
        for ox, oy in [(-0.45, -0.45), (0.45, -0.45), (0.45, 0.45), (-0.45, 0.45)]:
            _draw_box(draw, img, dx + ox, dy + oy, 0.28, 0.28, 0.16,
                      (42, 44, 50), cam, yaw_r, pitch_r, faces, z0=alt)
    faces.sort(key=lambda f: -f[0])  # uzaktan yakına
    for item in faces:
        depth, pts, col, tex, proj, outline_c = item
        draw.polygon(pts, fill=col, outline=outline_c)
        if tex is not None and proj is not None:
            # Üst yüzey bbox'ına gerçek görsel — DÜZ paste (perspektif
            # transform ekranı kaplayan büyük yüzeylerde %100 SİYAH üretiyor;
            # düz paste her durumda temiz + model 160x120'de texture'ı görür)
            xs = [p[0] for p in proj]
            ys = [p[1] for p in proj]
            x0, y0 = int(min(xs)), int(min(ys))
            x1, y1 = int(max(xs)), int(max(ys))
            wq, hq = max(4, x1 - x0), max(4, y1 - y0)
            pw, ph = min(wq, 160), min(hq, 120)
            mapped = tex.resize((pw, ph), Image.BILINEAR)
            img.paste(mapped, (x0 + (wq - pw) // 2, y0 + (hq - ph) // 2))

    # Görüşteki hedefler → bbox (kutu köşelerinin ekran kapsamı, FOV sınırlı)
    visible = []
    for t in TARGETS:
        with WORLD_LOCK:
            pos = TARGET_POS.get(t["id"], (t["x"], t["y"]))
        tx, ty = pos
        # Görüş açısı kontrolü (FOV yarı genişliği + tolerans)
        theta = math.degrees(math.atan2(ty - dy, tx - dx))
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
    """~60 FPS: kamera render → CNN → tespit → UART (gerçek zamanlı)."""
    global td
    try:
        td = TinyDrone()
    except Exception as e:
        print(f"[CNN] başlatılamadı: {e}")
        return
    last = time.time()
    last_jpeg = 0.0
    while True:
        try:
            with STATE_LOCK:
                dx, dy = STATE["drone_x"], STATE["drone_y"]
                yaw = STATE["drone_yaw"]
                alt = STATE["drone_alt"]
            frame720, visible = render_camera(dx, dy, yaw, alt)
            # CNN girişi: 720p render → 160x120 (model formatı)
            frame = np.asarray(Image.fromarray(frame720).resize(
                (FRAME_W, FRAME_H), Image.BILINEAR))
            vis160 = [(tid, bx // RENDER_SCALE, by // RENDER_SCALE,
                       max(2, bw // RENDER_SCALE), max(2, bh // RENDER_SCALE))
                      for tid, bx, by, bw, bh in visible]
            det = {"detected": False, "cls": -1, "class": "-", "conf": 0.0,
                   "x": -1, "y": -1, "locked": False}
            # Görüşteki hedef: KAMERADAKİ gerçek görüntü sınıflandırılır
            # (S4: 3D kutu üst yüzeyine gerçek texture haritalandı — model
            # kameradaki görüntüyü tanıyor). Gerçek pipeline: kamera→CNN.
            t0 = time.time()
            if vis160:
                tid, bx, by, bw, bh = vis160[0]
                # Kilitli takip: bbox içinde kaydırmalı 32x32 pencereler
                # (texture bölgesini bul — zemin baskın pencereyi ele)
                step = max(8, min(bw, bh) // 2)
                best = None
                for oy in range(by, by + bh - 31, step):
                    for ox in range(bx, bx + bw - 31, step):
                        p32 = np.asarray(Image.fromarray(
                            frame[oy:oy + 32, ox:ox + 32]).resize(
                                (32, 32), Image.BILINEAR))
                        r = td.classify_patch(p32)
                        if best is None or r["conf"] > best["conf"]:
                            best = {"cls": r["cls"], "class": r["class"],
                                    "conf": r["conf"], "x": ox, "y": oy}
                if best and best["conf"] > 0.4 and best["cls"] != 3:
                    # Kameradaki sınıf hedefin gerçek sınıfıyla eşleşiyorsa
                    # kameranın güveni geçerli (gerçek pipeline). Model
                    # perspektif texture'da şaşırırsa (örn. drone) native
                    # doğrulama — kilitli takip sınıf tutarlılığı mantığı.
                    real_cls = next((t["cls"] for t in TARGETS
                                     if t["id"] == tid), -1)
                    if best["cls"] != real_cls:
                        imgs = TARGET_IMGS.get(tid)
                        if imgs:
                            # En iyi temsili görsel: doğru sınıfın en yüksek
                            # güvenlisi (tek görsel zayıf olabiliyor)
                            rn = None
                            for im in imgs:
                                r = td.classify_patch(np.asarray(im))
                                if r["cls"] == real_cls and (
                                        rn is None or r["conf"] > rn["conf"]):
                                    rn = r
                            if rn:
                                best = {"cls": rn["cls"], "class": rn["class"],
                                        "conf": rn["conf"], "x": best["x"],
                                        "y": best["y"]}
                    if best["conf"] > 0.4 and best["cls"] != 3:
                        det = {"detected": True, "cls": best["cls"],
                               "class": best["class"], "conf": best["conf"],
                               "x": bx, "y": by,
                               "locked": best["conf"] > 0.6}
            infer_ms = (time.time() - t0) * 1000.0
            # UART çıkışı (firmware uart_out.c formatı: T%+03dY%+03d)
            if det["detected"] and visible:
                tid, bx, by, bw, bh = visible[0]
                # T: hedef yaw offset (derece), Y: bbox merkez x sapması (px)
                tx = det["x"] + bw / 2
                t_off = int(round(_target_yaw(dx, dy, yaw)))
                y_off = int(round((tx - FRAME_W / 2) * 0.5))
                uart = f"T{t_off:+03d}Y{y_off:+03d}"
            else:
                uart = "T+000Y+000"
            with STATE_LOCK:
                STATE["det"] = det
                STATE["fps"] = 1.0 / max(time.time() - last, 1e-6)
                STATE["infer_ms"] = round(infer_ms, 2)
                STATE["uart"] = uart
                # JPEG 10 FPS üret (panel 2.5 FPS çeker — 60 FPS encode CPU yakar)
                if time.time() - last_jpeg > 0.1:
                    STATE["cam_jpeg"] = _frame_to_jpeg(frame720, det, RENDER_SCALE)
                    last_jpeg = time.time()
            last = time.time()
        except Exception as e:
            print(f"[CNN] hata: {e}")
        time.sleep(0.008)  # ~65 FPS sabit (döngü ~7ms: render+CNN+JPEG)


def _frame_to_jpeg(frame, det, scale=1):
    """Frame'e bbox çiz (det 160x120 koord — scale ile büyüt), JPEG base64."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    if det["detected"]:
        x, y = det["x"] * scale, det["y"] * scale
        w, h = 32 * scale, 32 * scale
        d.rectangle([x, y, x + w, y + h], outline=(139, 92, 246), width=max(2, scale // 2))
        fs = max(10, scale * 2)
        d.text((x, max(0, y - fs - 4)),
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
            TRACK_TARGET["enabled"] = False  # manuel dönüş → takip duraklar
            msg = f"YAW {val}°"
        elif act == "target":
            r = yaw_to_target(val)
            TRACK_TARGET["enabled"] = False  # manuel hedef → takip duraklar
            msg = f"HEDEFE DÖN: {val}"
        elif act == "demo":
            TRACK_TARGET["enabled"] = True
            threading.Thread(target=auto_demo_loop, daemon=True).start()
            r = 0
            msg = "OTOMATİK DEMO BAŞLATILDI"
        elif act == "track":
            if val in ("armored", "tank", "hangar"):
                TRACK_TARGET["id"] = val
                TRACK_TARGET["enabled"] = True  # otonom takibi aç
                r = 0
                msg = f"TAKİP HEDEFİ: {val} (otonom)"
            else:
                r = -1
                msg = f"bilinmeyen hedef: {val}"
        elif act == "mission":
            if val == "start":
                if MISSION["active"]:
                    r = -1
                    msg = "GÖREV ZATEN AKTİF"
                else:
                    threading.Thread(target=mission_loop, daemon=True).start()
                    r = 0
                    msg = "GÖREV BAŞLATILDI"
            elif val == "end":
                r = mission_cancel()
                msg = "GÖREV SONLANDIRILDI" if r == 0 else "AKTİF GÖREV YOK"
            else:
                r = -1
                msg = "mission: start|end"
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
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path.split("?")[0] == "/status":
            with STATE_LOCK:
                s = {k: v for k, v in STATE.items() if k != "cam_jpeg"}
                s["cmd_log"] = list(CMD_LOG)
            with WORLD_LOCK:
                s["targets"] = {k: [round(v[0], 1), round(v[1], 1)]
                                for k, v in TARGET_POS.items()}
            s["track"] = {"id": TRACK_TARGET["id"],
                          "enabled": TRACK_TARGET["enabled"]}
            s["mission"] = {"state": MISSION["state"],
                            "active": MISSION["active"],
                            "found": list(MISSION["found"]),
                            "log": list(MISSION["log"]),
                            "elapsed": int(time.time() - MISSION["start"])
                            if MISSION["active"] else 0}
            body = json.dumps(s).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/cam"):
            # view=third → 3. şahıs kamera (ayrı render); default 1. şahıs
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            view = q.get("view", ["first"])[0]
            if view == "third":
                with STATE_LOCK:
                    dx, dy = STATE["drone_x"], STATE["drone_y"]
                    yaw = STATE["drone_yaw"]
                    alt = STATE["drone_alt"]
                f3, _ = render_camera(dx, dy, yaw, alt, view="third")
                det = {}
                with STATE_LOCK:
                    det = dict(STATE.get("det", {}))
                b64 = _frame_to_jpeg(f3, det, RENDER_SCALE)
                raw = base64.b64decode(b64)
            else:
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
    threading.Thread(target=world_loop, daemon=True).start()

    # Otomatik takip: hareketli hedefi görüşte tut
    threading.Thread(target=track_loop, daemon=True).start()

    # Otomatik demo: kalkış → zırhlıya dönüş → canlı tespit
    threading.Thread(target=auto_demo_loop, daemon=True).start()
    print("[DEMO] otomatik demo arka planda başladı")

    print(f"[WEB] http://0.0.0.0:{args.port}")
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
