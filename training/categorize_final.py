#!/usr/bin/env python3
"""Pass 3: Fix DS5/DS6 -> drone_uav, DS2/other -> background."""
from pathlib import Path
from PIL import Image
import random

PROC = Path("/home/ubuntu/projects/tinydrone/training/dataset/processed")
TMP = Path("/tmp/tinydrone_kaggle")
TARGET = (32, 32)
random.seed(42)

# Clear previous miscategorized
import shutil
for split in ("train", "val", "test"):
    for cls in ["tank", "armored_vehicle", "military_building", "drone_uav", "background"]:
        d = PROC / split / cls
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

def classify(img_path):
    p = str(img_path)
    pl = p.lower()
    
    # DS4: UAV Battle Tank Detection -> tank
    if "/ds4/" in pl:
        return "tank"
    # DS5: Drone YOLO Detection -> drone_uav
    if "/ds5/" in pl:
        return "drone_uav"
    # DS6: Amateur UAV -> drone_uav
    if "/ds6/" in pl:
        return "drone_uav"
    # DS2: Normal vs Military
    if "/ds2/" in pl:
        if "/millitary/" in pl:
            return "armored_vehicle"
        if "/other/" in pl:
            return "background"
        return "armored_vehicle"
    
    return "armored_vehicle"

all_images = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    all_images.extend(TMP.rglob(ext))

print(f"Classifying {len(all_images)} images...")

from collections import defaultdict
categorized = defaultdict(list)
for img_path in all_images:
    categorized[classify(img_path)].append(img_path)

print("\nDistribution:")
for c in ["tank", "armored_vehicle", "military_building", "drone_uav", "background"]:
    print(f"  {c}: {len(categorized[c])}")

# Resize and split
for cls in ["tank", "armored_vehicle", "military_building", "drone_uav", "background"]:
    images = categorized[cls]
    if not images:
        continue
    
    random.shuffle(images)
    n = len(images)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    for split_name, split_images in [
        ("train", images[:n_train]),
        ("val", images[n_train:n_train + n_val]),
        ("test", images[n_train + n_val:]),
    ]:
        out_dir = PROC / split_name / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in split_images:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(TARGET, Image.LANCZOS)
                fname = f"{img_path.parent.name}_{img_path.stem[:35]}.png"
                img.save(str(out_dir / fname), "PNG")
            except Exception:
                pass
    
    counts = {sn: len(list((PROC / sn / cls).glob("*.png"))) for sn in ("train", "val", "test")}
    print(f"  {cls}: train={counts['train']}, val={counts['val']}, test={counts['test']}")

print(f"\n=== FINAL ===")
total = 0
for split in ("train", "val", "test"):
    s_total = sum(len(list((PROC / split / c).glob("*.png"))) for c in ["tank", "armored_vehicle", "military_building", "drone_uav", "background"])
    print(f"  {split}: {s_total}")
    total += s_total
print(f"  TOTAL: {total}")
