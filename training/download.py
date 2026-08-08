#!/usr/bin/env python3
"""
Faz 0: tinydrone veri seti toplama pipeline'ı

Kaynaklar:
  1. OpenImages V7 — tank, military vehicle, aircraft
  2. xView — overhead military vehicle detection
  3. VisDrone — drone-taken aerial images (background sınıfı için)
  4. Synthetic augmentation — eksik sınıflar için

Sınıflar: tank, armored_vehicle, military_building, drone_uav, background
"""

import os
import sys
import json
import time
import shutil
import hashlib
import zipfile
import tarfile
import argparse
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

# ============================================================
# Config
# ============================================================

DATASET_DIR = Path(__file__).parent / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"

CLASSES = ["tank", "armored_vehicle", "military_building", "drone_uav", "background"]

TARGET_SIZE = (32, 32)   # tinycml CNN input
TARGET_PER_CLASS = 500   # Minimum per class

USER_AGENT = "Mozilla/5.0 (compatible; tinydrone-dataset-bot/1.0)"

# ============================================================
# OpenImages Downloader
# ============================================================

# OpenImages class labels we want
OI_CLASS_MAP = {
    "Tank":           ("tank", "/m/07cmd"),
    "Armored car":    ("armored_vehicle", "/m/02kw3q"),
    "Military vehicle": ("armored_vehicle", "/m/029fjf"),
    "Helicopter":     ("drone_uav", "/m/0c236"),
    "Aircraft":       ("drone_uav", "/m/0k5j"),
    "Military aircraft": ("drone_uav", "/m/01df0n"),
    "Building":       ("military_building", "/m/0cgh4"),
    "Truck":          ("armored_vehicle", "/m/07r04"),
}

def download_openimages(class_name, limit=200):
    """Download images from OpenImages using their public URL format.
    
    Uses OpenImages V7 image IDs from their class listings.
    Falls back to downloading from public annotation CSVs if available.
    """
    target_dir = RAW_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Map relevant OI classes to our target class
    oi_targets = [(oi_name, oi_mid) for oi_name, (our_cls, oi_mid) 
                  in OI_CLASS_MAP.items() if our_cls == class_name]
    
    downloaded = 0
    for oi_name, oi_mid in oi_targets:
        print(f"  OpenImages: {oi_name} ({oi_mid}) -> {class_name}")
        
        # Try downloading annotation CSV for this class
        csv_url = (f"https://storage.googleapis.com/openimages/v7/oidv7-class-train-annotations/"
                   f"{oi_name.replace(' ', '')}.csv")
        
        try:
            req = urllib.request.Request(csv_url, headers={"User-Agent": USER_AGENT})
            resp = urllib.request.urlopen(req, timeout=30)
            csv_data = resp.read().decode()
            
            # Parse ImageID column
            image_ids = []
            for line in csv_data.strip().split("\n")[1:]:  # Skip header
                parts = line.split(",")
                if len(parts) > 0:
                    image_ids.append(parts[0])
            
            # Download each image
            for img_id in image_ids[:limit - downloaded]:
                img_url = f"https://storage.googleapis.com/openimages/v7/images/{img_id}.jpg"
                save_path = target_dir / f"oi_{img_id}.jpg"
                
                if save_path.exists():
                    continue
                    
                try:
                    req2 = urllib.request.Request(img_url, headers={"User-Agent": USER_AGENT})
                    resp2 = urllib.request.urlopen(req2, timeout=15)
                    save_path.write_bytes(resp2.read())
                    downloaded += 1
                    if downloaded % 50 == 0:
                        print(f"    ... {downloaded}/{limit}")
                except Exception:
                    pass  # Skip broken images
                    
            if downloaded >= limit:
                break
                
        except urllib.error.HTTPError as e:
            print(f"    CSV not available: {e.code}")
        except Exception as e:
            print(f"    Error: {e}")
    
    return downloaded

# ============================================================
# xView Dataset Downloader
# ============================================================

XVIS_URLS = {
    "train_images": "http://xviewdataset.org.s3-website.us-east-2.amazonaws.com/xview_train_images.tgz",
    "train_labels": "http://xviewdataset.org.s3-website.us-east-2.amazonaws.com/xview_train_labels.tgz",
}

XVIEW_MILITARY_CLASSES = {
    11: "tank",            # "Truck Tractor w/ Box Trailer" — not exact, but overhead
    14: "tank",            # "Truck Tractor"  
    15: "tank",            # "Vehicle Towing"
    17: "armored_vehicle", # "Truck w/ Box"
    18: "armored_vehicle", # "Truck w/ Flatbed"
    19: "armored_vehicle", # "Truck w/ Liquid"
    61: "armored_vehicle", # "Bus"
    73: "tank",            # "Heavy Equipment"
    74: "military_building", # "Cargo/Container Stacks"
    # xView doesn't have explicit "military vehicle" — 
    # but truck/equipment classes work for overhead imagery
}

def download_xview(class_name, limit=200):
    """Download xView overhead imagery for military vehicle classes."""
    target_dir = RAW_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    xview_raw = RAW_DIR / "_xview_raw"
    xview_raw.mkdir(exist_ok=True)
    
    # Download if not already cached
    for name, url in XVIS_URLS.items():
        fname = xview_raw / url.split("/")[-1]
        if not fname.exists():
            print(f"  Downloading xView {name}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                resp = urllib.request.urlopen(req, timeout=300)
                fname.write_bytes(resp.read())
                print(f"    Done: {fname.stat().st_size / 1e6:.0f}MB")
            except Exception as e:
                print(f"    Failed: {e}")
                return 0
    
    # Extract and filter
    relevant_ids = [k for k, v in XVIEW_MILITARY_CLASSES.items() if v == class_name]
    if not relevant_ids:
        return 0
    
    # Load labels
    labels_file = xview_raw / "xview_train_labels.tgz"
    images_file = xview_raw / "xview_train_images.tgz"
    
    if not labels_file.exists() or not images_file.exists():
        return 0
    
    downloaded = 0
    try:
        import tarfile
        
        # Read labels to find relevant images
        with tarfile.open(labels_file, "r:gz") as tf:
            label_members = [m for m in tf.getmembers() if m.name.endswith(".geojson")]
            
            for member in label_members[:500]:  # Sample first 500
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = json.loads(f.read())
                
                for feature in data.get("features", []):
                    cat_id = feature["properties"].get("type_id")
                    if cat_id in relevant_ids and downloaded < limit:
                        img_name = feature["properties"]["image_id"]
                        # Extract image
                        try:
                            with tarfile.open(images_file, "r:gz") as img_tf:
                                img_member = None
                                for m in img_tf.getmembers():
                                    if img_name in m.name and m.name.endswith((".jpg", ".png")):
                                        img_member = m
                                        break
                                if img_member:
                                    img_f = img_tf.extractfile(img_member)
                                    if img_f:
                                        save_path = target_dir / f"xview_{img_name}"
                                        save_path.write_bytes(img_f.read())
                                        downloaded += 1
                        except Exception:
                            pass
                        break  # One image per geojson
                
                if downloaded >= limit:
                    break
                    
    except Exception as e:
        print(f"    xView extraction error: {e}")
    
    return downloaded

# ============================================================
# VisDrone Background Downloader
# ============================================================

VISDRONE_URLS = {
    "task1_images": "https://github.com/ultralytics/visdrone-det/raw/main/VisDrone2019-DET-train-images-part1.zip",
}

def download_visdrone_background(limit=300):
    """Download VisDrone images for background/negative class."""
    class_name = "background"
    target_dir = RAW_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = RAW_DIR / "_visdrone_cache"
    cache_dir.mkdir(exist_ok=True)
    
    downloaded = 0
    
    for name, url in VISDRONE_URLS.items():
        fname = cache_dir / url.split("/")[-1]
        if not fname.exists():
            print(f"  Downloading VisDrone {name}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                resp = urllib.request.urlopen(req, timeout=300)
                fname.write_bytes(resp.read())
            except Exception as e:
                print(f"    Failed: {e}")
                continue
        
        # Extract and copy to background class
        try:
            with zipfile.ZipFile(fname, "r") as zf:
                for member in zf.infolist():
                    if member.filename.endswith((".jpg", ".png")) and downloaded < limit:
                        save_path = target_dir / f"visdrone_{Path(member.filename).name}"
                        if save_path.exists():
                            continue
                        zf.extract(member, cache_dir / "extracted")
                        extracted = cache_dir / "extracted" / member.filename
                        if extracted.exists():
                            shutil.copy(extracted, save_path)
                            downloaded += 1
        except Exception as e:
            print(f"    Extraction error: {e}")
    
    # Clean up temp
    if (cache_dir / "extracted").exists():
        shutil.rmtree(cache_dir / "extracted")
    
    return downloaded

# ============================================================
# Synthetic Data Generator (for under-represented classes)
# ============================================================

def generate_synthetic(class_name, count=100):
    """Generate synthetic placeholder images using geometric patterns.
    
    For military targets: simple geometric shapes on varied backgrounds.
    Useful when real images aren't available in sufficient numbers.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        print(f"  [skip] PIL/numpy not available for synthetic generation")
        return 0
    
    target_dir = RAW_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    class_colors = {
        "tank": [(60, 80, 40), (70, 70, 50), (100, 90, 60)],     # olive/camo greens
        "armored_vehicle": [(80, 70, 50), (90, 85, 65), (70, 80, 60)],
        "military_building": [(120, 110, 100), (140, 130, 120), (100, 100, 110)],  # concrete
        "drone_uav": [(40, 40, 50), (60, 60, 70), (30, 30, 40)],  # dark grey
        "background": [(34, 139, 34), (70, 130, 50), (100, 150, 80)],  # green terrain
    }
    
    colors = class_colors.get(class_name, [(80, 80, 80)])
    generated = 0
    
    for i in range(count):
        img = Image.new("RGB", (320, 240))
        draw = ImageDraw.Draw(img)
        
        # Random terrain background
        bg_color = (
            np.random.randint(20, 80),
            np.random.randint(60, 120),
            np.random.randint(10, 60),
        )
        draw.rectangle([0, 0, 320, 240], fill=bg_color)
        
        # Add some ground texture
        for _ in range(20):
            x, y = np.random.randint(0, 320), np.random.randint(0, 240)
            r = np.random.randint(1, 5)
            shade = tuple(max(0, min(255, c + np.random.randint(-20, 20))) for c in bg_color)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=shade)
        
        # Draw target shape
        color = colors[np.random.randint(0, len(colors))]
        cx, cy = np.random.randint(80, 240), np.random.randint(60, 180)
        
        if class_name in ("tank", "armored_vehicle"):
            # Tank-like: rectangular body + turret
            body_w, body_h = np.random.randint(60, 120), np.random.randint(30, 50)
            body_color = (
                max(0, min(255, color[0] + np.random.randint(-15, 15))),
                max(0, min(255, color[1] + np.random.randint(-15, 15))),
                max(0, min(255, color[2] + np.random.randint(-15, 15))),
            )
            draw.rectangle(
                [cx - body_w//2, cy - body_h//2, cx + body_w//2, cy + body_h//2],
                fill=body_color, outline=(0, 0, 0)
            )
            # Turret
            turret_r = np.random.randint(10, 20)
            draw.ellipse(
                [cx - turret_r, cy - body_h//2 - turret_r,
                 cx + turret_r, cy - body_h//2 + turret_r],
                fill=body_color, outline=(0, 0, 0)
            )
            # Barrel
            barrel_len = np.random.randint(20, 40)
            barrel_color = (max(0, color[0]-30), max(0, color[1]-30), max(0, color[2]-30))
            draw.line([cx + turret_r//2, cy - body_h//2, 
                       cx + turret_r//2 + barrel_len, cy - body_h//2],
                      fill=barrel_color, width=3)
            
        elif class_name == "military_building":
            # Rectangular building
            bw, bh = np.random.randint(40, 100), np.random.randint(30, 80)
            draw.rectangle(
                [cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2],
                fill=color, outline=(30, 30, 30)
            )
            # Windows
            for wx in range(cx - bw//2 + 8, cx + bw//2 - 8, 8):
                for wy in range(cy - bh//2 + 8, cy + bh//2 - 8, 10):
                    draw.rectangle([wx, wy, wx+5, wy+5], fill=(200, 220, 255))
                    
        elif class_name == "drone_uav":
            # Cross-shaped drone
            arm_len = np.random.randint(15, 30)
            arm_w = np.random.randint(2, 5)
            # Horizontal arm
            draw.rectangle([cx - arm_len, cy - arm_w, cx + arm_len, cy + arm_w], fill=color)
            # Vertical arm
            draw.rectangle([cx - arm_w, cy - arm_len, cx + arm_w, cy + arm_len], fill=color)
            # Center body
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(30, 30, 40))
            
        else:  # background — just terrain
            pass  # Already drawn
        
        # Add noise
        arr = np.array(img)
        noise = np.random.randint(-10, 10, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        
        save_path = target_dir / f"syn_{class_name}_{i:04d}.png"
        img.save(save_path)
        generated += 1
    
    return generated

# ============================================================
# Unified Download Orchestrator
# ============================================================

def download_all(args):
    """Download from all available sources for each class."""
    
    sources = {}
    
    for cls in CLASSES:
        if args.skip_synthetic and cls in ("tank", "armored_vehicle"):
            continue
            
        print(f"\n{'='*60}")
        print(f"Class: {cls}")
        print(f"{'='*60}")
        
        class_total = 0
        
        # 1. OpenImages
        if not args.skip_openimages:
            n = download_openimages(cls, limit=args.limit // len(CLASSES))
            class_total += n
            print(f"  OpenImages: +{n}")
        
        # 2. xView (only for vehicle/building classes)
        if not args.skip_xview and cls in ("tank", "armored_vehicle", "military_building"):
            n = download_xview(cls, limit=args.limit // len(CLASSES))
            class_total += n
            print(f"  xView: +{n}")
        
        # 3. VisDrone (only for background)
        if not args.skip_visdrone and cls == "background":
            n = download_visdrone_background(limit=args.limit // 2)
            class_total += n
            print(f"  VisDrone: +{n}")
        
        # 4. Synthetic generation (fill gap)
        real_count = len(list((RAW_DIR / cls).glob("*.jpg")) + list((RAW_DIR / cls).glob("*.png")))
        target = max(args.limit // len(CLASSES), TARGET_PER_CLASS)
        if real_count < target:
            need = target - real_count
            n = generate_synthetic(cls, count=need)
            class_total += n
            print(f"  Synthetic: +{n}")
        
        sources[cls] = {
            "total_raw": class_total,
            "real": real_count,
            "synthetic": class_total - real_count if class_total > real_count else 0
        }
    
    return sources

# ============================================================
# Post-Processing: Resize & Organize
# ============================================================

def process_dataset(limit_per_class=500):
    """Resize all raw images to 32x32, split into train/val/test."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("PIL/numpy required for processing. Install: pip install Pillow numpy")
        return False
    
    for cls in CLASSES:
        print(f"\nProcessing: {cls}")
        raw_dir = RAW_DIR / cls
        if not raw_dir.exists():
            continue
        
        # Collect all images
        images = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))
        if not images:
            print(f"  No images found!")
            continue
        
        # Shuffle
        np.random.seed(42)
        indices = np.random.permutation(len(images))
        images = [images[i] for i in indices[:limit_per_class]]
        
        # Split: 70% train, 15% val, 15% test
        n = len(images)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        
        for split_name, split_images in splits.items():
            split_dir = PROCESSED_DIR / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)
                    out_path = split_dir / f"{img_path.stem}.png"
                    img.save(out_path, "PNG")
                except Exception as e:
                    pass  # Skip corrupted
        
        print(f"  {cls}: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
              f"(train/val/test)")
    
    return True

# ============================================================
# Dataset Statistics
# ============================================================

def print_stats():
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total = 0
    for cls in CLASSES:
        raw_dir = RAW_DIR / cls
        if raw_dir.exists():
            raw_count = len(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))
        else:
            raw_count = 0
        
        proc_counts = {}
        for split in ("train", "val", "test"):
            split_dir = PROCESSED_DIR / split / cls
            if split_dir.exists():
                proc_counts[split] = len(list(split_dir.glob("*.png")))
            else:
                proc_counts[split] = 0
        
        proc_total = sum(proc_counts.values())
        total += proc_total
        
        print(f"\n  {cls}:")
        print(f"    Raw: {raw_count}")
        print(f"    Processed: train={proc_counts['train']}, "
              f"val={proc_counts['val']}, test={proc_counts['test']} "
              f"= {proc_total}")
    
    print(f"\n  TOTAL processed: {total}")
    
    # Check minimums
    for cls in CLASSES:
        proc_dir = PROCESSED_DIR / "train" / cls
        count = len(list(proc_dir.glob("*.png"))) if proc_dir.exists() else 0
        if count < TARGET_PER_CLASS:
            print(f"  ⚠ {cls}: {count}/{TARGET_PER_CLASS} — needs more data")

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="tinydrone dataset collection")
    parser.add_argument("--download", action="store_true", help="Download from online sources")
    parser.add_argument("--process", action="store_true", help="Resize and split dataset")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--limit", type=int, default=2500, help="Total images to download")
    parser.add_argument("--all", action="store_true", help="Download + process + stats")
    parser.add_argument("--skip-openimages", action="store_true")
    parser.add_argument("--skip-xview", action="store_true")
    parser.add_argument("--skip-visdrone", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-online", action="store_true", 
                        help="Skip online downloads, only generate synthetic")
    
    args = parser.parse_args()
    
    if args.all or args.download:
        if args.skip_online:
            print("Skipping online sources — generating synthetic data only")
            for cls in CLASSES:
                generate_synthetic(cls, count=args.limit // len(CLASSES))
        else:
            download_all(args)
    
    if args.all or args.process:
        process_dataset(limit_per_class=args.limit // len(CLASSES))
    
    if args.all or args.stats:
        print_stats()
    
    if not any([args.download, args.process, args.stats, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()
