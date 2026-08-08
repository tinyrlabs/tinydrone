#!/usr/bin/env python3
"""
Faz 2: TinyCNN PyTorch eğitim + C array weight export

TinyCNN Mimarisi:
  Conv2D(3, 16, 3x3, pad=1) → ReLU → MaxPool(2x2)  # 32→16, 16ch
  Conv2D(16, 32, 3x3, pad=1) → ReLU → MaxPool(2x2) # 16→8, 32ch
  Conv2D(32, 64, 3x3, pad=1) → ReLU → MaxPool(2x2) # 8→4, 64ch
  Flatten → 64*4*4 = 1024
  Dense(1024, 128) → ReLU → Dropout(0.5)
  Dense(128, N_CLASSES) → Softmax

Toplam parametre: ~100K (int8 quantize ile ~100KB)
Hedef doğruluk: >%80 (4 sınıf)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ============================================================
# Config
# ============================================================

DATASET_DIR = Path(__file__).parent / "dataset" / "processed"
OUTPUT_DIR = Path(__file__).parent / "output"
CLASSES = ["tank", "armored_vehicle", "drone_uav", "background"]
N_CLASSES = len(CLASSES)
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# ============================================================
# Dataset
# ============================================================

class TinydroneDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = DATASET_DIR / split / cls_name
            if not cls_dir.exists():
                print(f"  Warning: {cls_dir} not found, skipping {cls_name}")
                continue
            for img_path in cls_dir.glob("*.png"):
                self.samples.append(str(img_path))
                self.labels.append(cls_idx)
        
        print(f"  {split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ============================================================
# TinyCNN Model (PyTorch)
# ============================================================

class TinyCNN(nn.Module):
    """PyTorch implementation matching the tinycml CNN architecture."""
    
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)    # 32→32, 16ch
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # 16→16, 32ch
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)   # 8→8, 64ch
        self.pool = nn.MaxPool2d(2, 2)                   # halves spatial dim
        
        self.fc1 = nn.Linear(64 * 4 * 4, 128)           # 1024→128
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        # Block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # Block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data
    train_ds = TinydroneDataset("train")
    val_ds = TinydroneDataset("val")
    test_ds = TinydroneDataset("test")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Handle class imbalance with weighted loss
    class_counts = np.bincount(train_ds.labels, minlength=N_CLASSES)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    print(f"Class distribution: {dict(zip(CLASSES, class_counts))}")
    print(f"Class weights: {class_weights}")
    
    # Model
    model = TinyCNN(n_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training {EPOCHS} epochs...\n")
    
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * correct / total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_acc)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(OUTPUT_DIR / "best_model.pth"))
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")
    
    print(f"\nBest validation accuracy: {best_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load(str(OUTPUT_DIR / "best_model.pth")))
    
    # Test
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100.0 * correct / total
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = np.array(all_labels) == cls_idx
        if mask.sum() > 0:
            cls_acc = 100.0 * (np.array(all_preds)[mask] == cls_idx).sum() / mask.sum()
            print(f"  {cls_name}: {cls_acc:.2f}%")
    
    # Save history
    json.dump(history, open(str(OUTPUT_DIR / "training_history.json"), "w"), indent=2)
    
    return model, test_acc

# ============================================================
# Weight Export to C Array
# ============================================================

def export_weights(model):
    """Export PyTorch model weights as C header arrays for tinycml."""
    print("\nExporting weights to C arrays...")
    
    layers = []
    
    # Conv1
    w1 = model.conv1.weight.data.cpu().numpy()  # (16, 3, 3, 3)
    b1 = model.conv1.bias.data.cpu().numpy()    # (16,)
    layers.append(("conv1_weight", w1, "float"))
    layers.append(("conv1_bias", b1, "float"))
    
    # Conv2
    w2 = model.conv2.weight.data.cpu().numpy()  # (32, 16, 3, 3)
    b2 = model.conv2.bias.data.cpu().numpy()
    layers.append(("conv2_weight", w2, "float"))
    layers.append(("conv2_bias", b2, "float"))
    
    # Conv3
    w3 = model.conv3.weight.data.cpu().numpy()  # (64, 32, 3, 3)
    b3 = model.conv3.bias.data.cpu().numpy()
    layers.append(("conv3_weight", w3, "float"))
    layers.append(("conv3_bias", b3, "float"))
    
    # FC1
    w4 = model.fc1.weight.data.cpu().numpy()  # (128, 1024)
    b4 = model.fc1.bias.data.cpu().numpy()
    layers.append(("fc1_weight", w4, "float"))
    layers.append(("fc1_bias", b4, "float"))
    
    # FC2
    w5 = model.fc2.weight.data.cpu().numpy()  # (N_CLASSES, 128)
    b5 = model.fc2.bias.data.cpu().numpy()
    layers.append(("fc2_weight", w5, "float"))
    layers.append(("fc2_bias", b5, "float"))
    
    # Generate C header
    c_code = f"""/**
 * tinydrone_model.h — Auto-generated by train_cnn.py
 *
 * TinyCNN weights exported for tinycml.
 * Architecture: Conv(16)→Pool→Conv(32)→Pool→Conv(64)→Pool→FC(128)→FC({N_CLASSES})
 * Input: 32×32×3 RGB, normalized to [-1, 1]
 * Classes: {", ".join(CLASSES)}
 */

#ifndef TINYDRONE_MODEL_H
#define TINYDRONE_MODEL_H

#define TINYDRONE_N_CLASSES {N_CLASSES}
#define TINYDRONE_INPUT_H 32
#define TINYDRONE_INPUT_W 32
#define TINYDRONE_INPUT_C 3

"""
    
    for name, arr, dtype in layers:
        arr_flat = arr.flatten()
        c_code += f"static const float {name}[{len(arr_flat)}] = {{\n    "
        c_code += ", ".join(f"{v:.8f}f" for v in arr_flat)
        c_code += "\n};\n\n"
    
    c_code += "#endif /* TINYDRONE_MODEL_H */\n"
    
    header_path = OUTPUT_DIR / "tinydrone_model.h"
    header_path.write_text(c_code)
    
    size_kb = len(c_code) / 1024
    print(f"  Written: {header_path} ({size_kb:.1f} KB)")
    
    # Also save as numpy for verification
    np.savez(str(OUTPUT_DIR / "tinydrone_weights.npz"),
             conv1_weight=w1, conv1_bias=b1,
             conv2_weight=w2, conv2_bias=b2,
             conv3_weight=w3, conv3_bias=b3,
             fc1_weight=w4, fc1_bias=b4,
             fc2_weight=w5, fc2_bias=b5)
    
    return header_path

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TinyCNN training for tinydrone")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--export-only", action="store_true", 
                        help="Skip training, just export from saved model")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.export_only:
        model = TinyCNN(n_classes=N_CLASSES)
        model_path = OUTPUT_DIR / "best_model.pth"
        if model_path.exists():
            model.load_state_dict(torch.load(str(model_path)))
            export_weights(model)
        else:
            print(f"Model not found: {model_path}")
    else:
        model, test_acc = train(args)
        export_weights(model)
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Test accuracy: {test_acc:.2f}%")
        print(f"Model saved: {OUTPUT_DIR / 'best_model.pth'}")
        print(f"Weights exported: {OUTPUT_DIR / 'tinydrone_model.h'}")

if __name__ == "__main__":
    main()
