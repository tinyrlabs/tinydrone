#!/usr/bin/env python3
"""
xView'den military_building sınıfı üret: Hangar(49) + Facility(51) + Hut/Tent(46)
Görseller 640x640 uydu görüntüsü, bbox'lar normalized YOLO formatında.
Crop → 32x32 resize → 70/15/15 split → dataset/processed/*/military_building/
"""
import os, random
import pandas as pd
from PIL import Image

SRC = "/tmp/xview_ds"
IMG_DIR = f"{SRC}/images"
LABELS = f"{SRC}/xview_labels.parquet"
DST = "/home/ubuntu/projects/tinydrone/training/dataset/processed"
CLASS_IDS = {46, 49, 51}
MIN_BBOX_PX = 8          # minimum crop boyutu (piksel)
MAX_PER_IMG = 4          # görsel başına max crop (aynı bina tekrarını önle)

random.seed(42)
df = pd.read_parquet(LABELS)
for col in ["x_center", "y_center", "width", "height"]:
    df[col] = df[col].astype(float)
mb = df[df["Class_ID"].isin(CLASS_IDS)].copy()
mb["file"] = mb["File_Name"].str.replace(".txt", ".jpg")

# Görsel başına sınıf bazlı örnekleri grupla
groups = mb.groupby("file")
items = []  # (file, x, y, w, h)
for f, g in groups:
    if not os.path.exists(f"{IMG_DIR}/{f}"):
        continue
    for _, r in g.head(MAX_PER_IMG).iterrows():
        w_px = r["width"] * 640
        h_px = r["height"] * 640
        if w_px < MIN_BBOX_PX or h_px < MIN_BBOX_PX:
            continue
        x = (r["x_center"] - r["width"] / 2) * 640
        y = (r["y_center"] - r["height"] / 2) * 640
        items.append((f, x, y, w_px, h_px))

random.shuffle(items)
n = len(items)
train = items[:int(n * 0.7)]
val = items[int(n * 0.7):int(n * 0.85)]
test = items[int(n * 0.85):]

print(f"Toplam crop: {n} (train {len(train)}, val {len(val)}, test {len(test)})")

for split, subset in [("train", train), ("val", val), ("test", test)]:
    outdir = f"{DST}/{split}/military_building"
    os.makedirs(outdir, exist_ok=True)
    for i, (f, x, y, w, h) in enumerate(subset):
        img = Image.open(f"{IMG_DIR}/{f}").convert("RGB")
        box = (int(x), int(y), int(x + w), int(y + h))
        crop = img.crop(box).resize((32, 32), Image.BILINEAR)
        crop.save(f"{outdir}/xview_{i:05d}.png")
    print(f"{split}/military_building: {len(subset)}")

print("TAMAM")
