"""
OpenSeek — Unified Training Orchestrator
===========================================
Trains all 3 forensic branches (Spatial, Frequency, Noise) in sequence,
then summarises per-branch accuracy.

Usage:
    cd backend/
    python train/train_all.py --data_dir ./data --epochs 20 --batch_size 16

    # Quick smoke test (1 epoch, 100 images max):
    python train/train_all.py --data_dir ./data --epochs 1 --max_samples 100

    # Resume interrupted training:
    python train/train_all.py --data_dir ./data --epochs 20 --resume

Output weights auto-loaded by the ensemble on next server restart:
    weights/spatial_weights.pt
    weights/frequency_weights.pt
    weights/noise_weights.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure we can import from backend root when run from any cwd
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from models.forensics.spatial import SpatialBranch, get_spatial_transform
from models.forensics.frequency import FrequencyBranch
from models.forensics.noise import NoiseBranch

# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
WEIGHTS_DIR = os.path.join(_BACKEND_ROOT, "weights")


# ── Dataset ───────────────────────────────────────────────────────────────────

class ForensicDataset(Dataset):
    """
    Universal dataset for all 3 forensic branches.
    Expects:
        root/real/*.jpg   (label 0)
        root/fake/*.jpg   (label 1)
    """

    def __init__(self, root: str, branch: str, max_samples: int | None = None):
        self.branch = branch
        self.samples: list[tuple[str, float]] = []

        for sub, label in [("real", 0.0), ("fake", 1.0)]:
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                    self.samples.append((os.path.join(d, fname), label))

        if max_samples and len(self.samples) > max_samples:
            import random
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        n_real = sum(1 for _, l in self.samples if l == 0.0)
        n_fake = sum(1 for _, l in self.samples if l == 1.0)
        print(f"  [{branch}] Dataset — Real: {n_real:,}  Fake: {n_fake:,}  Total: {len(self.samples):,}")

        # Pre-compute augmentations
        self._spatial_aug = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.4),
            transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.5))], p=0.2),
            # Simulates JPEG compression artifact during training
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        t = torch.tensor([label], dtype=torch.float32)
        try:
            if self.branch == "spatial":
                img = Image.open(path).convert("RGB")
                return self._spatial_aug(img), t

            elif self.branch == "frequency":
                img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img_cv is None:
                    raise ValueError("cv2 read failed")
                img_cv = cv2.resize(img_cv, (224, 224))
                f = np.fft.fft2(img_cv)
                mag = 20 * np.log(np.abs(np.fft.fftshift(f)) + 1e-9)
                mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
                return torch.from_numpy(mag).float().unsqueeze(0), t

            elif self.branch == "noise":
                img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img_cv is None:
                    raise ValueError("cv2 read failed")
                img_cv = cv2.resize(img_cv, (224, 224))
                tensor = torch.from_numpy(img_cv).float().unsqueeze(0) / 255.0
                return tensor, t

        except (UnidentifiedImageError, ValueError, Exception):
            # Return a zero tensor for corrupt files — DataLoader won't crash
            if self.branch == "spatial":
                return torch.zeros(3, 224, 224), t
            elif self.branch == "frequency":
                return torch.zeros(1, 224, 224), t
            else:
                return torch.zeros(1, 224, 224), t


def _make_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Balance class sampling so each batch has ~50% real / 50% fake."""
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    n_real = labels.count(0.0)
    n_fake = labels.count(1.0)
    w_real = 1.0 / max(n_real, 1)
    w_fake = 1.0 / max(n_fake, 1)
    weights = [w_real if l == 0.0 else w_fake for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(preds, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ── Branch training loop ──────────────────────────────────────────────────────

def train_branch(
    name: str,
    model: nn.Module,
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    max_samples: int | None,
    resume: bool,
    patience: int = 5,
) -> float:
    """
    Train a single forensic branch. Returns best validation accuracy.
    Saves best weights to weights/{name}_weights.pt.
    """
    weight_path = os.path.join(WEIGHTS_DIR, f"{name}_weights.pt")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Resume from existing weights
    if resume and os.path.exists(weight_path):
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"  [{name}] Resumed from {weight_path}")
        except Exception as e:
            print(f"  [{name}] Resume failed ({e}), starting fresh.")

    full_ds = ForensicDataset(data_dir, branch=name, max_samples=max_samples)
    if len(full_ds) < 10:
        print(f"  [{name}] ❌ Not enough data to train ({len(full_ds)} samples). Skipping.")
        return 0.0

    val_size = max(1, int(0.15 * len(full_ds)))
    train_ds, val_ds = random_split(
        full_ds,
        [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    sampler = _make_weighted_sampler(full_ds)
    # Only apply sampler to training split indices
    train_sampler = WeightedRandomSampler(
        [sampler.weights[i] for i in train_ds.indices],
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=min(4, os.cpu_count() or 1), pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=min(4, os.cpu_count() or 1)
    )

    model = model.to(device)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_acc = 0.0
    no_improve = 0

    print(f"\n{'='*55}")
    print(f"  Training [{name.upper()}] branch — {epochs} epochs")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0
        bar = tqdm(train_loader, desc=f"  Ep {epoch:02d}/{epochs} [train]", leave=False)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += ((preds > 0.5).float() == y).sum().item()
            train_total += x.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        val_tp = val_tn = val_fp = val_fn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                pred_cls = (preds > 0.5).float()
                val_correct += (pred_cls == y).sum().item()
                val_total += x.size(0)
                val_tp += ((pred_cls == 1) & (y == 1)).sum().item()
                val_tn += ((pred_cls == 0) & (y == 0)).sum().item()
                val_fp += ((pred_cls == 1) & (y == 0)).sum().item()
                val_fn += ((pred_cls == 0) & (y == 1)).sum().item()

        train_acc = train_correct / max(train_total, 1) * 100
        val_acc   = val_correct   / max(val_total,   1) * 100
        elapsed   = time.time() - t0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            marker = "  ✅ best"
            no_improve = 0
        else:
            no_improve += 1

        precision = val_tp / max(val_tp + val_fp, 1) * 100
        recall    = val_tp / max(val_tp + val_fn, 1) * 100

        print(
            f"  [{name} Ep {epoch:02d}/{epochs}] "
            f"Loss: {train_loss/max(train_total,1):.4f} | "
            f"Train: {train_acc:.1f}% | "
            f"Val: {val_acc:.1f}% | "
            f"Prec: {precision:.1f}% | Rec: {recall:.1f}% | "
            f"{elapsed:.0f}s{marker}"
        )

        # Early stopping
        if no_improve >= patience:
            print(f"  [{name}] Early stopping — no improvement for {patience} epochs.")
            break

    print(f"  [{name}] ✅ Best val accuracy: {best_val_acc:.1f}% → {weight_path}")
    return best_val_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train all OpenSeek forensic branches")
    parser.add_argument("--data_dir",   required=True, help="Folder with real/ and fake/ subfolders")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=5,   help="Early stopping patience")
    parser.add_argument("--max_samples",type=int,   default=None, help="Cap total samples (for quick tests)")
    parser.add_argument("--resume",     action="store_true",      help="Resume from existing weights")
    parser.add_argument("--branches",   nargs="+", default=["spatial", "frequency", "noise"],
                        help="Which branches to train (default: all three)")
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_dir, "real")):
        print("ERROR: data_dir must contain a 'real/' subfolder")
        print("Run: python scripts/prepare_dataset.py --output_dir ./data")
        sys.exit(1)
    if not os.path.isdir(os.path.join(args.data_dir, "fake")):
        print("ERROR: data_dir must contain a 'fake/' subfolder")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[OpenSeek Trainer] Device: {device}")
    if device.type == "cpu":
        print("  ⚠ CPU training is slow. Expect ~30–60 min per epoch for 5000 images.")
        print("  Consider using a GPU (CUDA) machine for full training.")

    # Map branch names to model constructors
    branch_models = {
        "spatial":   SpatialBranch,
        "frequency": FrequencyBranch,
        "noise":     NoiseBranch,
    }

    results: dict[str, float] = {}

    for branch_name in args.branches:
        if branch_name not in branch_models:
            print(f"Unknown branch '{branch_name}'. Choose from: spatial, frequency, noise")
            continue

        model = branch_models[branch_name]()
        acc = train_branch(
            name=branch_name,
            model=model,
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            max_samples=args.max_samples,
            resume=args.resume,
            patience=args.patience,
        )
        results[branch_name] = acc
        del model  # Free memory before next branch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  Training Complete — Final Results")
    print("=" * 55)
    for branch, acc in results.items():
        status = "✅" if acc >= 75 else "⚠"
        print(f"  {status} {branch:12s}  {acc:.1f}% val accuracy")
    print()
    print("  Weights saved to ./weights/")
    print("  Restart the backend to load new weights:")
    print("  uvicorn main:app --reload --port 8000")
    print("=" * 55)


if __name__ == "__main__":
    main()
