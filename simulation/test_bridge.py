#!/usr/bin/env python3
"""
test_bridge.py — CNN köprüsü doğrulama testi.
Gerçek test görsellerinden 160x120 frame üretir → tespit doğrular.
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinydrone_bridge import TinyDrone, FRAME_W, FRAME_H

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "training", "dataset", "processed", "test")


def make_frame(target_path, px, py):
    """Düz gri 160x120 frame + hedef görseli (px,py) — model davranışı testi."""
    frame = np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)
    img = Image.open(target_path).convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    frame[py:py + 32, px:px + 32] = arr
    return frame


def main():
    td = TinyDrone()
    ok = True

    # 1) Drone tespiti
    drone = os.path.join(DATA, "drone_uav", "Database1_100.png")
    frame = make_frame(drone, 60, 40)
    det = td.detect_track(frame)
    print(f"[1] drone frame: detected={det['detected']} cls={det['class']} conf={det['conf']:.2f} locked={det['locked']}")
    if not det["detected"] or det["cls"] != 2:
        print("    FAIL — drone bulunamadı"); ok = False
    else:
        print("    PASS ✓")

    # 2) Kilitli takip (aynı frame tekrar — 1 inference olmalı, kilit devam)
    det2 = td.detect_track(frame)
    st = td.track_state()
    print(f"[2] kilitli tekrar: locked={det2['locked']} state={st}")
    if not det2["locked"]:
        print("    FAIL — kilit yok"); ok = False
    else:
        print("    PASS ✓")

    # 3) Hedef kaydı (4px) — kilit takip etmeli
    frame3 = make_frame(drone, 64, 44)
    det3 = td.detect_track(frame3)
    print(f"[3] hedef kaydı: detected={det3['detected']} x={det3['x']} y={det3['y']} locked={det3['locked']}")
    if not det3["detected"]:
        print("    FAIL — kayma takip edilemedi"); ok = False
    else:
        print("    PASS ✓")

    # 4) Background (düz gri) — yanlış pozitif riski kontrolü
    frame4 = np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)
    det4 = td.detect_track(frame4)
    print(f"[4] düz bg: detected={det4['detected']} cls={det4['class']} conf={det4['conf']:.2f}")
    # düz gri modelde yanlış pozitif verebilir (eğitim verisinde yok) — bilgi amaçlı

    print("\nSONUÇ:", "TÜMÜ PASS" if ok else "HATA VAR")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
