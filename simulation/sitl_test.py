#!/usr/bin/env python3
"""
sitl_test.py — ArduPilot SITL headless doğrulama.
arducopter SITL başlat → MAVLink bağlan → arm → takeoff → yükseklik doğrula.
"""
import sys
import time
import subprocess
from pymavlink import mavutil

BIN = "/home/ubuntu/ardupilot/build/sitl/bin/arducopter"
MAV_TCP = "tcp:127.0.0.1:5760"  # SITL SERIAL0 TCP 5760'da dinler


DEFAULTS = "/home/ubuntu/ardupilot/Tools/autotest/default_params/copter.parm"


def start_sitl():
    print("[SITL] başlatılıyor...")
    proc = subprocess.Popen(
        [BIN, "--home", "40.7645,29.9269,0,0", "--model", "quad",
         "--defaults", DEFAULTS],
        stdout=open("/tmp/ardu.log", "w"), stderr=subprocess.STDOUT)
    return proc


def main():
    proc = start_sitl()
    try:
        # MAVLink bağlantısı (UDP dinle)
        mav = None
        for i in range(20):
            try:
                mav = mavutil.mavlink_connection(MAV_TCP)
                mav.wait_heartbeat(timeout=3)
                if mav.target_system:
                    break
            except Exception:
                mav = None
            time.sleep(1)
        if not mav:
            print("FAIL: MAVLink bağlantısı kurulamadı")
            return 1
        print(f"[MAV] bağlandı — sysid={mav.target_system}")

        # Telemetri stream iste (GPS_RAW_INT=24, LOCAL_POSITION_NED=32,
        # VFR_HUD=74, EKF_STATUS_REPORT=193)
        for msg_id in (24, 32, 74, 193):
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, 200000, 0, 0, 0, 0, 0)
        print("[STREAM] telemetri istendi")

        # GPS fix + EKF pozisyon bekle (60 sn timeout — ARM'de sim yavaş)
        print("[GPS] fix bekleniyor...")
        fixed = False
        t0 = time.time()
        while time.time() - t0 < 60:
            m = mav.recv_match(type="GPS_RAW_INT", blocking=True, timeout=2)
            if m and m.fix_type >= 3:
                fixed = True
                print(f"[GPS] fix OK (uydular={m.satellites_visible})")
                break
        if not fixed:
            print("[GPS] uyarı: fix alınamadı, yine de deneniyor")

        # EKF position estimate bekle (EKF_STATUS_REPORT flags bit1 = velocity,
        # bit2 = pos_horiz, bit3 = pos_vert)
        print("[EKF] pozisyon bekleniyor...")
        ekf_ok = False
        t0 = time.time()
        while time.time() - t0 < 60:
            m = mav.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=2)
            if m:
                flags = m.flags
                if (flags & 0b00000110) == 0b00000110:  # pos_horiz + pos_vert
                    ekf_ok = True
                    print(f"[EKF] pozisyon OK (flags=0x{flags:x})")
                    break
        if not ekf_ok:
            print("[EKF] uyarı: pozisyon alınamadı, yine de deneniyor")
        time.sleep(3)

        # Home zorla set et (mevcut konum) + ACK bekle
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME, 0,
            1, 0, 0, 0, 0, 0, 0)  # param1=1 → mevcut konum home
        time.sleep(2)
        while True:
            m = mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=2)
            if m and m.command == mavutil.mavlink.MAV_CMD_DO_SET_HOME:
                print(f"[HOME] ACK={m.result}")
                break
            if not m:
                break
        time.sleep(6)  # AHRS home kabul + EKF settle
        print("[HOME] beklendi")

        # Mod GUIDED + ACK bekle
        mav.set_mode_apm("GUIDED")
        print("[MOD] GUIDED gönderildi")
        time.sleep(2)

        # Arm + ACK bekle (retry: EKF core'ları settle olunca arm başarılı olur)
        ack = None
        for attempt in range(6):
            mav.arducopter_arm()
            t0 = time.time()
            while time.time() - t0 < 5:
                m = mav.recv_match(type=["COMMAND_ACK", "STATUSTEXT"],
                                   blocking=True, timeout=2)
                if not m:
                    continue
                if m.get_type() == "STATUSTEXT":
                    if "PreArm" in m.text or "Arm" in m.text:
                        print(f"[SYS] {m.text}")
                elif m.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    ack = m.result
                    break
            if ack == 0:
                print(f"[ARM] OK (deneme {attempt + 1})")
                break
            print(f"[ARM] deneme {attempt + 1} başarısız (ACK={ack}), 5sn bekleniyor")
            time.sleep(5)
        print(f"[ARM] ACK={ack} ({'OK' if ack == 0 else 'HATA'})")
        time.sleep(2)

        # Takeoff 10m + ACK
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, 10)
        ack2 = None
        t0 = time.time()
        while time.time() - t0 < 5:
            m = mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=2)
            if m and m.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                ack2 = m.result
                break
        print(f"[TAKEOFF] ACK={ack2} ({'OK' if ack2 == 0 else 'HATA'})")

        # Yüksekliği izle (20 sn) — VFR_HUD (alt) + LOCAL_POSITION_NED (z)
        max_alt = 0.0
        t0 = time.time()
        while time.time() - t0 < 20:
            msg = mav.recv_match(type=["VFR_HUD", "LOCAL_POSITION_NED"],
                                 blocking=True, timeout=3)
            if msg:
                if msg.get_type() == "VFR_HUD":
                    alt = msg.alt
                else:
                    alt = -msg.z
                if alt > max_alt:
                    max_alt = alt
                print(f"  alt={alt:.1f}m (max={max_alt:.1f})")
        print(f"\nSONUÇ: max yükseklik={max_alt:.1f}m")
        if max_alt > 2.0:
            print("PASS ✓ — SITL uçuyor")
            return 0
        print("FAIL — drone yükselmedi")
        return 1
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
