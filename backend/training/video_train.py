"""
DeepShield — Production Video Training Script (v2)
====================================================
Model  : TemporalTransformerDetector (EfficientNet-B0 + 4-layer Transformer)
Loss   : CombinedLoss (FocalLoss + LabelSmoothing)
Aug    : Frame dropout, temporal jitter
Optim  : AdamW + WarmupCosine + gradient accumulation
Metrics: AUC, F1, Precision, Recall, FPR, FNR, Confusion Matrix
TBoard : Full TensorBoard logging
Target : AUC > 0.88

Usage:
    python training/video_train.py --data_dir ./data/videos --epochs 30 \\
        --batch_size 4 --accum_steps 4 --tensorboard_dir ./runs
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

from training.models.video_model import TemporalTransformerDetector
from training.utils.datasets import build_video_loaders
from training.utils.losses import CombinedLoss
from training.metrics import (
    MetricsTracker, save_metrics_log, plot_roc_curve,
    print_confusion_matrix, log_to_tensorboard, get_writer,
)
from training.scheduler import get_scheduler
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


def train_video(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[VideoTrain] Device: {device}")

    print("\n[VideoTrain] Building video data loaders…")
    train_loader, val_loader, test_loader = build_video_loaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_frames  = args.num_frames,
        image_size  = args.image_size,
        num_workers = args.num_workers,
        max_samples = args.max_samples,
    )

    print("[VideoTrain] Building TemporalTransformerDetector…")
    model = TemporalTransformerDetector(
        num_frames   = args.num_frames,
        d_model      = args.d_model,
        nhead        = args.nhead,
        num_layers   = args.num_layers,
        freeze_blocks= args.freeze_blocks,
        pretrained   = True,
    ).to(device)

    criterion = CombinedLoss(focal_weight=0.7, smooth_weight=0.3)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler("warmup_cosine", optimizer, epochs=args.epochs,
                               warmup=args.warmup_epochs, eta_min=args.lr * 0.01)
    scaler  = get_amp_scaler(device)
    ckpt    = CheckpointManager(args.checkpoint_dir, tag="video", monitor="auc")
    stopper = EarlyStopping(patience=args.patience, monitor="auc")
    writer  = get_writer(args.tensorboard_dir, "video")
    metrics_tracker = MetricsTracker()
    all_val_metrics = []
    plot_dir = os.path.join(args.checkpoint_dir, "video", "plots")
    accum = max(1, args.accum_steps)

    print(f"\n  Effective batch size: {args.batch_size * accum}")
    print(f"\n{'='*60}")
    print(f"  Training TemporalTransformerDetector — {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        with EpochTimer(epoch, args.epochs, "Video"):

            # ── Train ────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            train_tracker = MetricsTracker()
            optimizer.zero_grad()

            for step, (frames, labels) in enumerate(train_loader):
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).view(-1, 1)

                if scaler:
                    with torch.cuda.amp.autocast():
                        logits = model(frames)
                        loss = criterion(logits, labels) / accum
                    scaler.scale(loss).backward()
                else:
                    logits = model(frames)
                    loss = criterion(logits, labels) / accum
                    loss.backward()

                train_loss += loss.item() * accum * frames.size(0)
                probs = torch.sigmoid(logits.detach()).cpu().numpy().squeeze()
                lbl   = labels.cpu().numpy().squeeze()
                train_tracker.update(_safe_np(probs), _safe_np(lbl))

                if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                    clip_and_step(optimizer, model, scaler, max_norm=1.0)
                    optimizer.zero_grad()

            scheduler.step()
            avg_train_loss = train_loss / max(len(train_loader.dataset), 1)
            train_m = train_tracker.compute(epoch, "train", avg_train_loss)
            print(train_m.summary())
            log_to_tensorboard(writer, train_m, tag="Video")

            # ── Validate ─────────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for frames, labels in val_loader:
                    frames = frames.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True).view(-1, 1)
                    logits = model(frames)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * frames.size(0)
                    probs = torch.sigmoid(logits).cpu().numpy().squeeze()
                    lbl   = labels.cpu().numpy().squeeze()
                    metrics_tracker.update(_safe_np(probs), _safe_np(lbl))

            val_m = metrics_tracker.compute(epoch, "val", val_loss / max(len(val_loader.dataset), 1))
            print(val_m.summary())
            log_to_tensorboard(writer, val_m, tag="Video", log_images=(epoch % 10 == 0))
            all_val_metrics.append(val_m)

            if writer:
                writer.add_scalar("video/lr", optimizer.param_groups[0]["lr"], epoch)

            if epoch % 5 == 0 or epoch == args.epochs:
                print_confusion_matrix(val_m)
                plot_roc_curve(val_m, plot_dir, tag="video")

            d = {"auc": val_m.auc, "f1": val_m.f1, "accuracy": val_m.accuracy, "epoch": epoch}
            ckpt.save(model, d)
            if stopper.step(d):
                break

    # ── Test ──────────────────────────────────────────────────────────────────
    ckpt.load_best(model, device)
    model.eval()
    test_tracker = MetricsTracker()
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(frames)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            test_tracker.update(_safe_np(probs), _safe_np(labels.cpu().numpy().squeeze()))
    test_m = test_tracker.compute(epoch, "test", 0.0)
    print(f"\n{'='*60}\n  ✅ TEST: {test_m.summary()}\n{'='*60}")
    print_confusion_matrix(test_m)
    save_metrics_log(all_val_metrics, os.path.join(args.checkpoint_dir, "video"))
    if writer:
        log_to_tensorboard(writer, test_m, tag="Video")
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Video Training v2")
    parser.add_argument("--data_dir",       required=True)
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--accum_steps",    type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",  type=int,   default=5)
    parser.add_argument("--patience",       type=int,   default=7)
    parser.add_argument("--num_frames",     type=int,   default=16)
    parser.add_argument("--d_model",        type=int,   default=512)
    parser.add_argument("--nhead",          type=int,   default=8)
    parser.add_argument("--num_layers",     type=int,   default=4)
    parser.add_argument("--freeze_blocks",  type=int,   default=4)
    parser.add_argument("--image_size",     type=int,   default=224)
    parser.add_argument("--num_workers",    type=int,   default=2)
    parser.add_argument("--max_samples",    type=int,   default=None)
    parser.add_argument("--checkpoint_dir", default="training/checkpoints")
    parser.add_argument("--tensorboard_dir",default="runs")
    args = parser.parse_args()
    train_video(args)
