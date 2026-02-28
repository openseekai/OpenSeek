"""
DeepShield — Production Image Training Script (v2)
====================================================
Model  : DualStreamImageDetector (EfficientNet-B4 + FFT Branch)
Loss   : CombinedLoss (FocalLoss + LabelSmoothingBCE)
Aug    : Albumentations (spatial, compression, noise, blur) + Mixup/CutMix + Freq masking
Optim  : AdamW + WarmupCosine scheduler
Metrics: AUC, F1, Precision, Recall, FPR, FNR, Confusion Matrix
TBoard : Full TensorBoard logging (loss, AUC, F1, confusion matrix, ROC curve)
CV     : Optional 5-fold cross-validation (--folds 5)
Target : AUC > 0.90

Usage:
    cd /path/to/backend
    source venv/bin/activate

    # Standard training:
    python training/image_train.py \\
        --data_dir ./data/images \\
        --extra_dirs ./data/diffusion_faces \\
        --epochs 40 --batch_size 16 \\
        --tensorboard_dir ./runs

    # 5-fold cross-validation:
    python training/image_train.py \\
        --data_dir ./data/images --folds 5

    # Cross-dataset evaluation (train FF++/CelebDF, test DFDC):
    python training/image_train.py \\
        --train_dirs ./data/ffpp ./data/celebdf \\
        --test_dirs ./data/dfdc \\
        --cross_dataset

    # Smoke test:
    python training/image_train.py --data_dir data_smoke --epochs 1 \\
        --max_samples 40 --batch_size 4 --num_workers 0

data_dir/ must contain:
    real/  ← authentic face images
    fake/  ← deepfake/AI-generated images
"""
from __future__ import annotations

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from training.models.image_model import DualStreamImageDetector
from training.dataset_loader import ImageLoader, KFoldDatasetBuilder
from training.utils.datasets import _collect_files, IMAGE_EXTS
from training.utils.losses import CombinedLoss
from training.metrics import (
    MetricsTracker, save_metrics_log, plot_roc_curve,
    print_confusion_matrix, log_to_tensorboard, get_writer, MetricAggregator,
    compute_cross_dataset_metrics,
)
from training.scheduler import get_scheduler
from training.augmentations import mixup_batch, cutmix_batch, random_frequency_mask
from training.utils.training_utils import (
    CheckpointManager, EarlyStopping, get_amp_scaler, clip_and_step, EpochTimer,
)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_np(arr):
    a = np.array(arr).squeeze()
    return np.atleast_1d(float(a)) if a.ndim == 0 else a.reshape(-1)


# ── Single-Split Training ──────────────────────────────────────────────────────

def run_one_epoch(model, loader, criterion, optimizer, scaler, device,
                   epoch, split, mixup_p=0.4, cutmix_p=0.3, freq_mask_p=0.2,
                   is_train=True):
    """Shared train/val loop — returns (avg_loss, MetricsTracker)."""
    model.train() if is_train else model.eval()
    tracker = MetricsTracker()
    total_loss = 0.0

    import contextlib
    ctx = torch.cuda.amp.autocast() if scaler else contextlib.nullcontext()

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)

        if is_train:
            # Mixup or CutMix
            if random.random() < mixup_p:
                imgs, labels = mixup_batch(imgs, labels, alpha=0.4)
            elif random.random() < cutmix_p:
                imgs, labels = cutmix_batch(imgs, labels, alpha=1.0)
            # Random frequency masking
            if random.random() < freq_mask_p:
                imgs = torch.stack([random_frequency_mask(im) for im in imgs])

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                clip_and_step(optimizer, model, scaler, max_norm=1.0)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                clip_and_step(optimizer, model, None, max_norm=1.0)
        else:
            with torch.no_grad():
                logits = model(imgs)
                loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits.detach()).cpu().numpy().squeeze()
        lbl   = labels.cpu().numpy().squeeze()
        tracker.update(_safe_np(probs), _safe_np(lbl))

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, tracker


def train_image(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ImageTrain] Device: {device}")

    criterion = CombinedLoss(focal_weight=0.7, smooth_weight=0.3, gamma=2.0, alpha=0.75)
    plot_dir  = os.path.join(args.checkpoint_dir, "image", "plots")

    # ── Cross-Dataset Mode ──────────────────────────────────────────────────
    if getattr(args, "cross_dataset", False):
        train_dirs = getattr(args, "train_dirs", [args.data_dir])
        test_dirs  = getattr(args, "test_dirs",  [])
        print(f"\n[ImageTrain] Cross-dataset mode")
        train_loader, test_loader = ImageLoader.cross_dataset(
            train_dirs=train_dirs, test_dirs=test_dirs,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        val_loader = test_loader
        # Run single training with cross-dataset test
        _train_loop(args, device, train_loader, val_loader, test_loader, criterion, plot_dir, fold=None)
        return

    # ── Build Dataset ────────────────────────────────────────────────────────
    extra_dirs = [args.extra_dirs] if isinstance(getattr(args, 'extra_dirs', None), str) else getattr(args, 'extra_dirs', None)
    train_loader, val_loader, test_loader = ImageLoader.build(
        primary_dir   = args.data_dir,
        extra_dirs    = extra_dirs,
        batch_size    = args.batch_size,
        image_size    = args.image_size,
        num_workers   = args.num_workers,
        use_albumentations = not getattr(args, 'no_albumentations', False),
    )

    # ── 5-Fold Cross-Validation Mode ─────────────────────────────────────────
    n_folds = getattr(args, "folds", 1)
    if n_folds > 1:
        print(f"\n[ImageTrain] 5-Fold Cross-Validation ({n_folds} folds)")
        # Collect all train samples for folding
        from training.subject_split import build_subject_split
        train_s, val_s, test_s = build_subject_split(
            args.data_dir, format="flat", extensions=IMAGE_EXTS,
        )
        all_train = train_s + val_s  # Fold within train+val; keep test held-out
        builder = KFoldDatasetBuilder(all_train, n_splits=n_folds)
        aggregator = MetricAggregator()

        for fold, fold_train, fold_val in builder.iter_folds(
            batch_size=args.batch_size, image_size=args.image_size, num_workers=args.num_workers
        ):
            print(f"\n  ── Fold {fold}/{n_folds} ──")
            val_m = _train_loop(args, device, fold_train, fold_val, test_loader,
                                criterion, plot_dir, fold=fold)
            if val_m:
                aggregator.add(val_m)

        aggregator.report()
        return

    # ── Standard Single-Split Training ────────────────────────────────────────
    _train_loop(args, device, train_loader, val_loader, test_loader, criterion, plot_dir, fold=None)


def _train_loop(args, device, train_loader, val_loader, test_loader,
                criterion, plot_dir, fold):
    """Core training loop (single split or one fold)."""
    tag   = f"image_fold{fold}" if fold else "image"
    ckpt  = CheckpointManager(args.checkpoint_dir, tag=tag, monitor="auc")
    stopper = EarlyStopping(patience=args.patience, monitor="auc")

    writer = get_writer(getattr(args, "tensorboard_dir", "runs"), tag)

    model = DualStreamImageDetector(pretrained=True).to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler("warmup_cosine", optimizer, epochs=args.epochs,
                               warmup=args.warmup_epochs, eta_min=args.lr * 0.01)
    scaler = get_amp_scaler(device)

    metrics_tracker = MetricsTracker()
    all_val_metrics = []
    best_val_m = None

    print(f"\n{'='*60}")
    print(f"  Training Image Model [{tag}] — {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        with EpochTimer(epoch, args.epochs, tag):
            train_loss, train_tracker = run_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                epoch, "train", is_train=True,
                mixup_p=getattr(args, 'mixup_prob', 0.4),
                cutmix_p=getattr(args, 'cutmix_prob', 0.3),
            )
            train_m = train_tracker.compute(epoch, "train", train_loss)
            print(train_m.summary())
            log_to_tensorboard(writer, train_m, tag=tag.capitalize())

            _, val_tracker = run_one_epoch(
                model, val_loader, criterion, optimizer, scaler, device,
                epoch, "val", is_train=False,
            )
            # Compute val loss separately
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs   = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True).view(-1, 1)
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                    val_loss += loss.item() * imgs.size(0)
                    probs = torch.sigmoid(logits).cpu().numpy().squeeze()
                    lbl   = labels.cpu().numpy().squeeze()
                    metrics_tracker.update(_safe_np(probs), _safe_np(lbl))

            avg_val_loss = val_loss / max(len(val_loader.dataset), 1)
            val_m = metrics_tracker.compute(epoch, "val", avg_val_loss)
            print(val_m.summary())
            log_to_tensorboard(writer, val_m, tag=tag.capitalize(), log_images=(epoch % 10 == 0))
            all_val_metrics.append(val_m)
            best_val_m = val_m

            # Log LR to TensorBoard
            if writer:
                writer.add_scalar(f"{tag}/lr", optimizer.param_groups[0]["lr"], epoch)

            if epoch % 5 == 0 or epoch == args.epochs:
                print_confusion_matrix(val_m)
                plot_roc_curve(val_m, plot_dir, tag=tag)

            metrics_dict = {"auc": val_m.auc, "f1": val_m.f1, "accuracy": val_m.accuracy, "epoch": epoch}
            ckpt.save(model, metrics_dict)
            scheduler.step()
            if stopper.step(metrics_dict):
                break

    # ── Final Test ────────────────────────────────────────────────────────────
    if test_loader:
        print(f"\n[{tag}] Loading best model for test evaluation...")
        ckpt.load_best(model, device)
        results = compute_cross_dataset_metrics(model, {"Test": test_loader}, device, criterion)
        test_m = results.get("Test")
        if test_m:
            print(f"\n{'='*60}")
            print(f"  ✅ FINAL TEST [{tag}]:")
            print(test_m.summary())
            print_confusion_matrix(test_m)
            print(f"{'='*60}")
            plot_roc_curve(test_m, plot_dir, tag=f"{tag}_test")

    save_metrics_log(all_val_metrics, os.path.join(args.checkpoint_dir, tag))
    if writer:
        writer.close()

    return best_val_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Image Deepfake Training v2")
    parser.add_argument("--data_dir",        default=None, help="Primary data dir (real/ + fake/)")
    parser.add_argument("--extra_dirs",      default=None, nargs="+", help="Extra dataset dirs (diffusion, etc.)")
    parser.add_argument("--train_dirs",      default=None, nargs="+", help="Cross-dataset train dirs")
    parser.add_argument("--test_dirs",       default=None, nargs="+", help="Cross-dataset test dirs")
    parser.add_argument("--cross_dataset",   action="store_true")
    parser.add_argument("--folds",           type=int, default=1, help="N-fold CV (1 = no CV)")
    parser.add_argument("--epochs",          type=int,   default=40)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--patience",        type=int,   default=7)
    parser.add_argument("--image_size",      type=int,   default=224)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--mixup_prob",      type=float, default=0.4)
    parser.add_argument("--cutmix_prob",     type=float, default=0.3)
    parser.add_argument("--max_samples",     type=int,   default=None)
    parser.add_argument("--no_albumentations", action="store_true")
    parser.add_argument("--checkpoint_dir",  default="training/checkpoints")
    parser.add_argument("--tensorboard_dir", default="runs")
    args = parser.parse_args()

    if args.data_dir and not os.path.isdir(args.data_dir):
        print(f"ERROR: data_dir '{args.data_dir}' not found"); sys.exit(1)

    train_image(args)
