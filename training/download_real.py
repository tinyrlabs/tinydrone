#!/usr/bin/env python3
"""
Faz 0: Gerçek askeri hedef veri seti indirme

Kaynaklar:
  Kaggle:
    - amanullahasraf/military-vehicles-detection-dataset
    - antoreepjana/tanks-dataset  
    - dasmehdixtr/drone-detection-dataset
  HuggingFace:
    - leibnitz-lab/military_vehicles
    - haffnerj/military_vehicles
"""

import os
import sys
import shutil
import zipfile
import tarfile
from pathlib import Path
import subprocess

DATASET_DIR = Path("/home/ubuntu/projects/tinydrone/training/dataset")
RAW_DIR = DATASET_DIR / "raw"

def run(cmd, **kwargs):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    return result.returncode == 0, result.stdout[:500], result.stderr[:500]

def download_kaggle():
    """Download Kaggle datasets using kagglehub."""
    print("=" * 60)
    print("KAGGLE DATASETS")
    print("=" * 60)
    
    datasets = [
        ("amanullahasraf/military-vehicles-detection-dataset", "tank+armored_vehicle"),
        ("antoreepjana/tanks-dataset", "tank"),
        ("dasmehdixtr/drone-detection-dataset", "drone_uav"),
        ("pacificparadise/tank-detection-dataset", "tank"),
    ]
    
    for ds_id, target_classes in datasets:
        print(f"\n--- {ds_id} ---")
        try:
            import kagglehub
            path = kagglehub.dataset_download(ds_id, force_download=False)
            print(f"  Downloaded to: {path}")
            
            # Move images to appropriate class directories
            _categorize_images(Path(path), ds_id, target_classes)
        except Exception as e:
            print(f"  FAILED: {e}")

def download_huggingface():
    """Download HuggingFace datasets."""
    print("\n" + "=" * 60)
    print("HUGGINGFACE DATASETS")
    print("=" * 60)
    
    hf_datasets = [
        "leibnitz-lab/military_vehicles",
        "haffnerj/military_vehicles",
    ]
    
    for ds_id in hf_datasets:
        print(f"\n--- {ds_id} ---")
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_id, split="train", trust_remote_code=True)
            print(f"  Loaded: {len(ds)} samples")
            print(f"  Features: {ds.features}")
            
            # Extract images
            count = 0
            for i, sample in enumerate(ds):
                if "image" in sample:
                    img = sample["image"]
                    label = sample.get("label", "unknown")
                    if isinstance(label, int):
                        label = ds.features["label"].int2str(label) if "label" in ds.features else str(label)
                    elif label is None:
                        label = "unknown"
                    
                    cls_dir = RAW_DIR / _map_label(label)
                    cls_dir.mkdir(parents=True, exist_ok=True)
                    
                    fname = f"hf_{ds_id.replace('/', '_')}_{i:05d}.png"
                    img.save(str(cls_dir / fname))
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"  ... {count}/{len(ds)}")
            
            print(f"  Extracted: {count} images")
        except Exception as e:
            print(f"  FAILED: {e}")

def _map_label(label):
    """Map dataset labels to our 5 classes."""
    label_lower = str(label).lower().replace(" ", "_")
    
    tank_keywords = ["tank", "t-", "mbt", "abrams", "leopard", "challenger"]
    vehicle_keywords = ["apc", "ifv", "humvee", "armored", "armoured", "btr", "bradley", "vehicle", "truck", "military_car"]
    drone_keywords = ["drone", "uav", "quadcopter", "helicopter", "aircraft", "plane", "jet", "fighter"]
    building_keywords = ["building", "bunker", "hangar", "base", "structure"]
    
    for kw in tank_keywords:
        if kw in label_lower:
            return "tank"
    for kw in vehicle_keywords:
        if kw in label_lower:
            return "armored_vehicle"
    for kw in drone_keywords:
        if kw in label_lower:
            return "drone_uav"
    for kw in building_keywords:
        if kw in label_lower:
            return "military_building"
    
    return "armored_vehicle"  # Default for military datasets

def _categorize_images(path, ds_id, target_classes):
    """Categorize downloaded images by class."""
    target_list = target_classes.split("+")
    
    # Find all image files
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(path.rglob(ext))
    
    if not images:
        print(f"  No images found in {path}")
        return
    
    print(f"  Found {len(images)} images")
    
    # If there are subdirectories that look like class names, use them
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    class_dirs = {d.name.lower(): d for d in subdirs}
    
    for cls in target_list:
        cls_dir = RAW_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
    
    if class_dirs:
        # Categorized by folder
        for dir_name, dir_path in class_dirs.items():
            mapped = _map_label(dir_name)
            if mapped in target_list:
                target_dir = RAW_DIR / mapped
                for img in dir_path.rglob("*"):
                    if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        fname = f"kag_{ds_id.replace('/', '_')}_{img.name}"
                        shutil.copy(img, target_dir / fname)
    else:
        # All images to first target class
        target_dir = RAW_DIR / target_list[0]
        for img in images:
            fname = f"kag_{ds_id.replace('/', '_')}_{img.name}"
            shutil.copy(img, target_dir / fname)

def print_stats():
    """Show current dataset stats."""
    print("\n" + "=" * 60)
    print("CURRENT RAW DATASET")
    print("=" * 60)
    for cls in ["tank", "armored_vehicle", "military_building", "drone_uav", "background"]:
        cls_dir = RAW_DIR / cls
        count = len(list(cls_dir.rglob("*"))) if cls_dir.exists() else 0
        print(f"  {cls}: {count}")

if __name__ == "__main__":
    print("tinydrone — Real Dataset Download\n")
    
    download_kaggle()
    download_huggingface()
    print_stats()
    
    print("\nDone. Run 'python3 download.py --process' to resize and split.")
