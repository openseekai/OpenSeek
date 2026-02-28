"""
DeepShield — Production Audio Training Script (v2)
====================================================
Model  : AudioDeepfakeDetector (2D CNN + MultiHead Attention)
Loss   : CombinedLoss (FocalLoss + LabelSmoothing)
Aug    : SpecAugment (2 freq + 2 time masks), pitch/speed/noise
Optim  : AdamW + WarmupCosine
Metrics: AUC, F1, Precision, Recall, FPR, FNR, Confusion Matrix
TBoard : Full TensorBoard logging
Target : AUC > 0.85

Usage:
    python training/audio_train.py --data_dir ./data/audio --epochs 30 \\
        --batch_size 32 --tensorboard_dir ./runs
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

from training.models.audio_model import AudioDeepfakeDetector
from training.utils.datasets import build_audio_loaders
from training.utils.losses import CombinedLoss
from training.metrics import (
    MetricsTracker, save_metrics_log, plot_roc_curve,
    print_confusion_matrix, log_to_tensorboard, get_writer,
)
from training.scheduler import get_scheduler
from training.augmentations import specaugment
from training.utils.training_utils import (
    CheckpointManager, EarlyStopping, get_amp_scaler, clip_and_step, EpochTimer,
)


def _safe_np(arr):
    a = np.array(arr).squeeze()
    return np.atleast_1d(float(a)) if a.ndim == 0 else a.reshape(-1)


def train_audio(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[AudioTrain] Device: {device}")

    print("\n[AudioTrain] Building audio data loaders…")
    train_loader, val_loader, test_loader = build_audio_loaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        sr          = args.sample_rate,
        duration    = args.duration,
        num_workers = args.num_workers,
        max_samples = args.max_samples,
    )

    print("[AudioTrain] Building AudioDeepfakeDetector…")
    model = AudioDeepfakeDetector(embed_dim=256, num_attn_heads=8, dropout=0.3).to(device)

    criterion = CombinedLoss(focal_weight=0.7, smooth_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler("warmup_cosine", optimizer, epochs=args.epochs,
                               warmup=args.warmup_epochs, eta_min=args.lr * 0.01)
    scaler  = get_amp_scaler(device)
    ckpt    = CheckpointManager(args.checkpoint_dir, tag="audio", monitor="auc")
    stopper = EarlyStopping(patience=args.patience, monitor="auc")
    writer  = get_writer(args.tensorboard_dir, "audio")
    metrics_tracker = MetricsTracker()
    all_val_metrics = []
    plot_dir = os.path.join(args.checkpoint_dir, "audio", "plots")

    print(f"\n{'='*60}")
    print(f"  Training AudioDeepfakeDetector — {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        with EpochTimer(epoch, args.epochs, "Audio"):

            # ── Train ─────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            train_tracker = MetricsTracker()

            for specs, labels in train_loader:
                specs  = specs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).view(-1, 1)

                # SpecAugment (2 freq + 2 time masks)
                if random.random() < 0.6:
                    specs = specaugment(specs, freq_mask_param=20, time_mask_param=25,
                                        n_freq_masks=2, n_time_masks=2)

                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        logits = model(specs)
                        loss   = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    clip_and_step(optimizer, model, scaler)
                else:
                    logits = model(specs)
                    loss   = criterion(logits, labels)
                    loss.backward()
                    clip_and_step(optimizer, model, None)

                train_loss += loss.item() * specs.size(0)
                probs = torch.sigmoid(logits.detach()).cpu().numpy().squeeze()
                lbl   = labels.cpu().numpy().squeeze()
                train_tracker.update(_safe_np(probs), _safe_np(lbl))

            scheduler.step()
            train_m = train_tracker.compute(epoch, "train", train_loss / max(len(train_loader.dataset), 1))
            print(train_m.summary())
            log_to_tensorboard(writer, train_m, tag="Audio")

            # ── Validate ─────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for specs, labels in val_loader:
                    specs  = specs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True).view(-1, 1)
                    logits = model(specs)
                    loss   = criterion(logits, labels)
                    val_loss += loss.item() * specs.size(0)
                    probs = torch.sigmoid(logits).cpu().numpy().squeeze()
                    lbl   = labels.cpu().numpy().squeeze()
                    metrics_tracker.update(_safe_np(probs), _safe_np(lbl))

            val_m = metrics_tracker.compute(epoch, "val", val_loss / max(len(val_loader.dataset), 1))
            print(val_m.summary())
            log_to_tensorboard(writer, val_m, tag="Audio", log_images=(epoch % 10 == 0))
            all_val_metrics.append(val_m)

            if writer:
                writer.add_scalar("audio/lr", optimizer.param_groups[0]["lr"], epoch)

            if epoch % 5 == 0 or epoch == args.epochs:
                print_confusion_matrix(val_m)
                plot_roc_curve(val_m, plot_dir, tag="audio")

            d = {"auc": val_m.auc, "f1": val_m.f1, "accuracy": val_m.accuracy, "epoch": epoch}
            ckpt.save(model, d)
            if stopper.step(d):
                break

    # ── Test ─────────────────────────────────────────────────────────────────
    ckpt.load_best(model, device)
    model.eval()
    test_tracker = MetricsTracker()
    with torch.no_grad():
        for specs, labels in test_loader:
            specs  = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(specs)
            probs  = torch.sigmoid(logits).cpu().numpy().squeeze()
            test_tracker.update(_safe_np(probs), _safe_np(labels.cpu().numpy().squeeze()))
    test_m = test_tracker.compute(epoch, "test", 0.0)
    print(f"\n{'='*60}\n  ✅ TEST: {test_m.summary()}\n{'='*60}")
    print_confusion_matrix(test_m)
    save_metrics_log(all_val_metrics, os.path.join(args.checkpoint_dir, "audio"))
    if writer:
        log_to_tensorboard(writer, test_m, tag="Audio")
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Audio Training v2")
    parser.add_argument("--data_dir",       required=True)
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",  type=int,   default=5)
    parser.add_argument("--patience",       type=int,   default=7)
    parser.add_argument("--sample_rate",    type=int,   default=16000)
    parser.add_argument("--duration",       type=float, default=4.0)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--max_samples",    type=int,   default=None)
    parser.add_argument("--checkpoint_dir", default="training/checkpoints")
    parser.add_argument("--tensorboard_dir",default="runs")
    args = parser.parse_args()
    train_audio(args)
