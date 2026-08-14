#!/usr/bin/env python3
"""
tinydrone_bridge.py — libtinydrone.so için ctypes sarmalayıcı.

Kullanım:
    from tinydrone_bridge import TinyDrone
    td = TinyDrone()
    det = td.detect_track(frame)   # frame: 160x120x3 numpy uint8 (RGB)
    # det: {'detected': bool, 'x': int, 'y': int, 'cls': int,
    #       'conf': float, 'locked': bool}
"""
import ctypes
import os
import numpy as np

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtinydrone.so")
FRAME_W = 160
FRAME_H = 120
CLASS_NAMES = ["tank", "armored_vehicle", "drone_uav", "background", "military_building"]


class TinyDrone:
    def __init__(self, lib_path=LIB_PATH):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"{lib_path} yok — önce 'make' çalıştır")
        self.lib = ctypes.CDLL(lib_path)

        # td_detect_track(frame, w, h, &x, &y, &cls, &conf, &locked) -> int
        self.lib.td_detect_track.restype = ctypes.c_int
        self.lib.td_detect_track.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.td_init.restype = None
        self.lib.td_init.argtypes = []
        self.lib.td_track_state.restype = None
        self.lib.td_track_state.argtypes = [
            ctypes.POINTER(ctypes.c_int)] * 5

        self.lib.td_init()

    def detect_track(self, frame):
        """frame: 160x120x3 uint8 RGB numpy → tespit sözlüğü."""
        assert frame.shape == (FRAME_H, FRAME_W, 3), f"beklenen {(FRAME_H, FRAME_W, 3)}, gelen {frame.shape}"
        buf = np.ascontiguousarray(frame, dtype=np.uint8)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        x, y, cls = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        conf = ctypes.c_double()
        locked = ctypes.c_int()
        det = self.lib.td_detect_track(ptr, FRAME_W, FRAME_H,
                                       ctypes.byref(x), ctypes.byref(y),
                                       ctypes.byref(cls), ctypes.byref(conf),
                                       ctypes.byref(locked))
        return {
            "detected": bool(det),
            "x": x.value, "y": y.value,
            "cls": cls.value,
            "class": CLASS_NAMES[cls.value] if 0 <= cls.value < len(CLASS_NAMES) else "?",
            "conf": conf.value,
            "locked": bool(locked.value),
        }

    def track_state(self):
        locked, lost, lx, ly, lcls = (ctypes.c_int() for _ in range(5))
        self.lib.td_track_state(ctypes.byref(locked), ctypes.byref(lost),
                                ctypes.byref(lx), ctypes.byref(ly),
                                ctypes.byref(lcls))
        return {"locked": bool(locked.value), "lost": lost.value,
                "last_x": lx.value, "last_y": ly.value, "last_cls": lcls.value}
