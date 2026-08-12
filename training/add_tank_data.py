#!/usr/bin/env python3
"""
Yeni tank görsellerini dataset'e ekle: 32x32 resize + 70/15/15 split.
"""
import os, random
from pathlib import Path
from PIL import Image

SRC = Path("/tmp/tank_ds/extracted/images")
DST = Path("/home/ubuntu/projects/tinydrone/training/dataset/processed")

random.seed(42)

files = []
for root, _, fnames in os.walk(SRC):
    for fn in fnames:
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(root, fn))

print(f"Toplam tank görseli: {len(files)}")
random.shuffle(files)

n = len(files)
n_train = int(n * 0.7)
n_val = int(n * 0.15)
splits = [("train", files[:n_train]), ("val", files[n_train:n_train + n_val]),
          ("test", files[n_train + n_val:])]

total_added = 0
for split, flist in splits:
    out_dir = DST / split / "tank"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.png")))
    for i, fp in enumerate(flist):
        try:
            img = Image.open(fp).convert("RGB").resize((32, 32), Image.LANCZOS)
            dst = out_dir / f"tankds_{existing + i:05d}.png"
            img.save(dst)
            total_added += 1
        except Exception as e:
            print(f"  hata: {fp}: {e}")

print(f"Eklenen: {total_added}")

# Final sayılar
for split in ["train", "val", "test"]:
    print(f"{split}/tank: {len(list((DST/split/'tank').glob('*.png')))}")
