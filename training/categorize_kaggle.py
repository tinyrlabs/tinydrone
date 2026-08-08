#!/usr/bin/env python3
"""
Kaggle datasetleri -> tinydrone 5-sinif kategorizasyon + 32x32 resize + split
"""
import os, sys, shutil
from pathlib import Path
from PIL import Image

RAW = Path("/home/ubuntu/projects/tinydrone/training/dataset/raw")
PROC = Path("/home/ubuntu/projects/tinydrone/training/dataset/processed")
TMP = Path("/tmp/tinydrone_kaggle")
TARGET = (32, 32)

CLASSES = ["tank", "armored_vehicle", "military_building", "drone_uav", "background"]
for c in CLASSES:
    (RAW / c).mkdir(parents=True, exist_ok=True)

def map_label(name):
    """Map filename/dirname to class."""
    n = name.lower()
    # DS2 specific: "millitary" -> armored_vehicle, "other" -> background
    if n == "millitary":
        return "armored_vehicle"
    if n == "other":
        return "background"
    # Tank keywords
    if any(k in n for k in ["tank", "t-72", "t-80", "t-90", "mbt", "abrams", "leopard", "challenger", "t64", "t55", "battle_tank"]):
        return "tank"
    # Drone/UAV/aircraft
    if any(k in n for k in ["drone", "uav", "quadcopter", "aircraft", "plane", "jet", "fighter", "helicopter",
                             "bayraktar", "tb2", "reaper", "predator", "su-", "mig-", "f-16", "f-15", "f-35"]):
        return "drone_uav"
    # Building/structure
    if any(k in n for k in ["building", "bunker", "hangar", "barrack", "base", "structure", "fortification"]):
        return "military_building"
    # Armored vehicle (APC, IFV, military truck)
    if any(k in n for k in ["apc", "ifv", "humvee", "btr", "bradley", "armored", "armoured",
                             "vehicle", "truck", "military_car", "bmp", "mtlb", "stryker"]):
        return "armored_vehicle"
    # Fallback: if it looks military, classify as armored_vehicle
    if any(k in n for k in ["military", "army", "war", "combat", "ukraine", "russian"]):
        return "armored_vehicle"
    return None

def categorize_and_process():
    """Walk through downloaded Kaggle data, categorize, resize, split."""
    if not TMP.exists():
        print(f"TMP dir not found: {TMP}")
        return
    
    all_images = list(TMP.rglob("*.jpg")) + list(TMP.rglob("*.jpeg")) + list(TMP.rglob("*.png"))
    print(f"Found {len(all_images)} images in Kaggle downloads")
    
    categorized = {c: [] for c in CLASSES}
    
    for img_path in all_images:
        # Try to classify from parent dir name first
        parent = img_path.parent.name.lower()
        cls = map_label(parent) or map_label(img_path.stem)
        if cls:
            categorized[cls].append(img_path)
        else:
            # Default: armored vehicle for military datasets
            categorized["armored_vehicle"].append(img_path)
    
    # Print distribution
    print("\nDistribution:")
    for c in CLASSES:
        print(f"  {c}: {len(categorized[c])}")
    
    # Resize and split: 70/15/15
    import random
    random.seed(42)
    
    for cls in CLASSES:
        images = categorized[cls]
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        
        for split_name, split_images in splits.items():
            out_dir = PROC / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(TARGET, Image.LANCZOS)
                    fname = f"{img_path.stem[:40]}.png"
                    img.save(str(out_dir / fname), "PNG")
                except Exception:
                    pass
        
        print(f"  {cls}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Final stats
    print(f"\n=== FINAL ===")
    total = 0
    for split in ("train", "val", "test"):
        s_total = sum(len(list((PROC / split / c).glob("*.png"))) for c in CLASSES)
        print(f"  {split}: {s_total}")
        total += s_total
    print(f"  TOTAL: {total}")

if __name__ == "__main__":
    categorize_and_process()
