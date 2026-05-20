"""
OpenSeek — Image Model Training Script

Usage:
  python train/train_image.py --data_dir ./data --epochs 10 --batch_size 8

data_dir must contain two subfolders:
  data/real/   ← real/authentic images
  data/fake/   ← deepfake/AI-generated images

Outputs:
  weights/image_model.pt  ← auto-loaded by load_image_model() on next startup

Recommended datasets (free on Kaggle):
  - 140k Real and Fake Faces: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
  - DeepFake and Real Images:  https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

Tips:
  - Even 500 images per class gives a big accuracy boost
  - CPU training: ~30 min per epoch for 1000 images
  - GPU training: ~2 min per epoch
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ─── Dataset ─────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.samples = []
        self.transform = transform
        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")
        for path in os.listdir(real_dir):
            if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                self.samples.append((os.path.join(real_dir, path), 0.0))
        for path in os.listdir(fake_dir):
            if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                self.samples.append((os.path.join(fake_dir, path), 1.0))
        print(f"[Dataset] Real: {sum(1 for _, l in self.samples if l == 0)}, "
              f"Fake: {sum(1 for _, l in self.samples if l == 1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(data_dir: str, epochs: int = 10, batch_size: int = 8, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Device: {device}")

    # EfficientNet-B4 — best accuracy/speed tradeoff per DFDC research
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset   = DeepfakeDataset(data_dir, transform=transform)
    val_size  = max(1, int(0.15 * len(dataset)))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size]
    )
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "image_model.pt")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            correct    += ((preds > 0.5).float() == labels).sum().item()
            total      += imgs.size(0)
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                val_correct += ((preds > 0.5).float() == labels).sum().item()
                val_total   += imgs.size(0)

        train_acc = correct / total * 100
        val_acc   = val_correct / val_total * 100
        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"Loss: {train_loss/total:.4f} | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)
            print(f"  ✅ Saved best model ({val_acc:.1f}%) → {weights_path}")

    print(f"\n[Done] Best val accuracy: {best_val_acc:.1f}%")
    print(f"[Done] Weights saved to: {weights_path}")
    print("[Done] Restart the backend to use the new weights.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpenSeek image model")
    parser.add_argument("--data_dir", required=True, help="Folder with real/ and fake/ subfolders")
    parser.add_argument("--epochs",   type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr",       type=float, default=1e-4)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_dir, "real")):
        print("ERROR: data_dir must contain a 'real/' subfolder")
        sys.exit(1)
    if not os.path.isdir(os.path.join(args.data_dir, "fake")):
        print("ERROR: data_dir must contain a 'fake/' subfolder")
        sys.exit(1)

    train(args.data_dir, args.epochs, args.batch_size, args.lr)
