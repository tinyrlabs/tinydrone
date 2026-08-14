#!/usr/bin/env python3
"""
sim_telemetry.py — SITL canlı telemetri web paneli.

ArduPilot SITL başlatır, MAVLink telemetriyi okur, tarayıcıya SSE ile iter.
- GET /        → index.html (canlı panel)
- GET /events  → SSE (telemetri akışı, 5 Hz)
- GET /status  → JSON (son durum)

Kullanım: python3 sim_telemetry.py [--port 8090] [--no-sitl]
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pymavlink import mavutil

BIN = "/home/ubuntu/ardupilot/build/sitl/bin/arducopter"
DEFAULTS = "/home/ubuntu/ardupilot/Tools/autotest/default_params/copter.parm"
MAV_TCP = "tcp:127.0.0.1:5760"
HERE = os.path.dirname(os.path.abspath(__file__))

# Canlı durum (thread'ler arası paylaşım)
STATE = {
    "connected": False, "armed": False, "mode": "-",
    "alt": 0.0, "yaw": 0.0, "speed": 0.0,
    "gps_fix": 0, "sats": 0, "volt": 0.0,
    "roll": 0.0, "pitch": 0.0, "time": 0.0,
    "lat": 0.0, "lon": 0.0,
}
STATE_LOCK = threading.Lock()


def start_sitl():
    proc = subprocess.Popen(
        [BIN, "--home", "40.7645,29.9269,0,0", "--model", "quad",
         "--defaults", DEFAULTS],
        stdout=open("/tmp/ardu.log", "w"), stderr=subprocess.STDOUT)
    return proc


def mavlink_loop():
    """MAVLink mesajlarını oku, STATE'i güncelle."""
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
                # Stream iste
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
                    STATE["volt"] = getattr(msg, "bat_volt", 0.0)
                elif t == "GPS_RAW_INT":
                    STATE["gps_fix"] = msg.fix_type
                    STATE["sats"] = msg.satellites_visible
                    STATE["lat"] = msg.lat / 1e7
                    STATE["lon"] = msg.lon / 1e7
                elif t == "ATTITUDE":
                    STATE["roll"] = msg.roll * 57.2958
                    STATE["pitch"] = msg.pitch * 57.2958
                elif t == "LOCAL_POSITION_NED":
                    STATE["time"] = msg.time_boot_ms / 1000.0
        except Exception as e:
            with STATE_LOCK:
                STATE["connected"] = False
            mav = None
            time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = open(os.path.join(HERE, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/status":
            with STATE_LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    with STATE_LOCK:
                        body = json.dumps(STATE)
                    self.wfile.write(f"data: {body}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--no-sitl", action="store_true", help="mevcut SITL'e bağlan")
    args = ap.parse_args()

    if not args.no_sitl:
        print("[SITL] başlatılıyor...")
        start_sitl()

    t = threading.Thread(target=mavlink_loop, daemon=True)
    t.start()

    print(f"[WEB] http://0.0.0.0:{args.port} — panel hazır")
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
