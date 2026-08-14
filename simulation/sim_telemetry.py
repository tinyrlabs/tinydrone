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
    {"id": "armored",    "x": 3.0,  "y": -4.0, "size": 2.2,
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

# Render yüksek çözünürlükte yapılıp küçültülür (yumuşak görüntü — mozaik yok)
R_W, R_H = 320, 240

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


def render_camera(dx, dy, yaw, alt=10.0):
    """Drone (dx,dy) NED + yaw (derece) + alt (m) → 160x120 kamera görüntüsü.
    320x240'ta render edilip küçültülür (yumuşak, düzgün görüntü).
    Zemin drone hareketiyle kayar, hedef boyutu 3D mesafeye göre ölçeklenir."""
    scale = R_W / FRAME_W  # 2.0
    if BG_TILES:
        img = Image.new("RGB", (R_W, R_H))
        rad = np.radians(yaw)
        off_x = int((-dx * np.cos(rad) - dy * np.sin(rad)) * 4) % 80
        off_y = int((dx * np.sin(rad) - dy * np.cos(rad)) * 4) % 80
        seed = int(time.time() * 3)
        for ty in range(-80, R_H, 80):
            for tx in range(-80, R_W, 80):
                tile = BG_TILES[(seed + (ty + off_y) // 80 + (tx + off_x) // 80) % len(BG_TILES)]
                img.paste(tile, (tx + off_x, ty + off_y))
    else:
        frame = np.full((R_H, R_W, 3), 96, dtype=np.uint8)
        img = Image.fromarray(frame)

    for t in TARGETS:
        vx, vy = t["x"] - dx, t["y"] - dy
        r3d = np.hypot(np.hypot(vx, vy), alt)
        if r3d < 1.0:
            continue
        theta = np.degrees(np.arctan2(vy, vx))
        delta = (theta - yaw + 180) % 360 - 180
        if abs(delta) > FOV_DEG / 2:
            continue
        # Yüksek çözünürlükte çiz (scale ×)
        sx = R_W / 2 + (delta / (FOV_DEG / 2)) * (R_W / 2)
        size_px = int(t["size"] / r3d * FOCAL * scale)
        size_px = max(8, min(size_px, R_W))
        imgs = TARGET_IMGS.get(t["id"])
        if not imgs:
            continue
        # SABİT görsel — dönen görseller captcha "araç seçin" efekti veriyordu
        tgt = imgs[0]
        tgt = tgt.resize((size_px, size_px), Image.BILINEAR)
        sy = int(R_H * 0.55)
        # NOT: mask'sız paste — RGB görsel mask olamaz ("bad transparency mask")
        img.paste(tgt, (int(sx - size_px / 2), sy))
    # Yumuşak küçültme → 160x120 (mozaik/captcha görünümü yok)
    return np.asarray(img.resize((FRAME_W, FRAME_H), Image.BILINEAR))


def cnn_loop():
    """Her ~180ms: kamera render → CNN tespit → STATE güncelle (canlı video)."""
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
            frame = render_camera(dx, dy, yaw, alt)
            det = td.detect_track(frame)
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
