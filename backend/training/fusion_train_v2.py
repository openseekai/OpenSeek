"""
OpenSeek — Production Fusion Training Script (v2)
====================================================
Model   : MultiModalFusionDetector (Cross-Modal Attention)
Strategy: Freeze pretrained branches → train fusion head only
Loss    : Weighted multi-task (image + video + audio + fusion)
Metrics : AUC, F1, Precision, Recall, FPR, FNR
TBoard  : Full TensorBoard logging
Target  : AUC > 0.92

Usage:
    python training/fusion_train.py \\
        --image_data ./data/images \\
        --video_data ./data/videos \\
        --audio_data ./data/audio \\
        --image_ckpt ./training/checkpoints/image/best_model.pt \\
        --video_ckpt ./training/checkpoints/video/best_model.pt \\
        --audio_ckpt ./training/checkpoints/audio/best_model.pt \\
        --epochs 20 --tensorboard_dir ./runs
"""
from __future__ import annotations

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from training.models.image_model import DualStreamImageDetector
from training.models.video_model import TemporalTransformerDetector
from training.models.audio_model import AudioDeepfakeDetector
from training.models.fusion_model import MultiModalFusionDetector, load_branch_embedder
from training.utils.datasets import (
    _collect_files, _subject_aware_split, IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS,
    make_weighted_sampler,
)
from training.utils.losses import CombinedLoss
from training.metrics import (
    MetricsTracker, save_metrics_log, plot_roc_curve,
    print_confusion_matrix, log_to_tensorboard, get_writer,
)
from training.scheduler import get_scheduler
from training.utils.training_utils import (
    CheckpointManager, EarlyStopping, get_amp_scaler, clip_and_step, EpochTimer,
)
from training.fusion_train import TriModalDataset  # Reuse existing


def _safe_np(arr):
    a = np.array(arr).squeeze()
    return np.atleast_1d(float(a)) if a.ndim == 0 else a.reshape(-1)


def train_fusion(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[FusionTrain] Device: {device}")

    # ── Load Pretrained Branches ─────────────────────────────────────────────
    print("\n[FusionTrain] Loading pretrained branch models…")
    img_branch = load_branch_embedder(DualStreamImageDetector, args.image_ckpt, device, freeze=True)
    vid_branch = load_branch_embedder(
        lambda: TemporalTransformerDetector(num_frames=args.num_frames, d_model=args.d_model),
        args.video_ckpt, device, freeze=True,
    )
    aud_branch = load_branch_embedder(AudioDeepfakeDetector, args.audio_ckpt, device, freeze=True)

    fusion_model = MultiModalFusionDetector(
        image_embed_dim=2304, video_embed_dim=args.d_model, audio_embed_dim=256, shared_dim=256,
    ).to(device)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[FusionTrain] Building fusion datasets…")
    img_all = _collect_files(args.image_data, IMAGE_EXTS)
    vid_all = _collect_files(args.video_data, VIDEO_EXTS)
    aud_all = _collect_files(args.audio_data, AUDIO_EXTS)

    img_tr, img_val, _ = _subject_aware_split(img_all)
    vid_tr, vid_val, _ = _subject_aware_split(vid_all)
    aud_tr, aud_val, _ = _subject_aware_split(aud_all)

    train_ds = TriModalDataset(img_tr, vid_tr, aud_tr, "train", args.num_frames)
    val_ds   = TriModalDataset(img_val, vid_val, aud_val, "val",  args.num_frames)

    sampler = make_weighted_sampler([(s, l) for s, l in zip(range(len(train_ds.labels)), train_ds.labels)])
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=use_pin)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    criterion = CombinedLoss()
    optimizer = optim.AdamW(fusion_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_scheduler("warmup_cosine", optimizer, epochs=args.epochs,
                               warmup=3, eta_min=args.lr * 0.01)
    scaler  = get_amp_scaler(device)
    ckpt    = CheckpointManager(args.checkpoint_dir, tag="fusion", monitor="auc")
    stopper = EarlyStopping(patience=args.patience, monitor="auc")
    writer  = get_writer(args.tensorboard_dir, "fusion")
    metrics_tracker = MetricsTracker()
    all_val_metrics = []
    plot_dir = os.path.join(args.checkpoint_dir, "fusion", "plots")

    print(f"\n{'='*60}")
    print(f"  Training Fusion Module — {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        with EpochTimer(epoch, args.epochs, "Fusion"):

            # ── Train ─────────────────────────────────────────────────────
            fusion_model.train()
            img_branch.eval(); vid_branch.eval(); aud_branch.eval()
            train_loss = 0.0
            train_tracker = MetricsTracker()

            for imgs, vids, auds, labels in train_loader:
                imgs   = imgs.to(device, non_blocking=True)
                vids   = vids.to(device, non_blocking=True)
                auds   = auds.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).view(-1, 1)

                optimizer.zero_grad()
                with torch.no_grad():
                    img_emb = img_branch.get_embedding(imgs)
                    vid_emb = vid_branch.get_embedding(vids)
                    aud_emb = aud_branch.get_embedding(auds)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = fusion_model(img_emb, vid_emb, aud_emb)
                        loss, _ = fusion_model.compute_total_loss(outputs, labels, criterion)
                    scaler.scale(loss).backward()
                    clip_and_step(optimizer, fusion_model, scaler)
                else:
                    outputs = fusion_model(img_emb, vid_emb, aud_emb)
                    loss, _ = fusion_model.compute_total_loss(outputs, labels, criterion)
                    loss.backward()
                    clip_and_step(optimizer, fusion_model, None)

                train_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(outputs["fusion"].detach()).cpu().numpy().squeeze()
                lbl   = labels.cpu().numpy().squeeze()
                train_tracker.update(_safe_np(probs), _safe_np(lbl))

            scheduler.step()
            train_m = train_tracker.compute(epoch, "train", train_loss / max(len(train_loader.dataset), 1))
            print(train_m.summary())
            log_to_tensorboard(writer, train_m, tag="Fusion")

            # ── Validate ─────────────────────────────────────────────────
            fusion_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, vids, auds, labels in val_loader:
                    imgs   = imgs.to(device, non_blocking=True)
                    vids   = vids.to(device, non_blocking=True)
                    auds   = auds.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True).view(-1, 1)
                    img_emb = img_branch.get_embedding(imgs)
                    vid_emb = vid_branch.get_embedding(vids)
                    aud_emb = aud_branch.get_embedding(auds)
                    outputs = fusion_model(img_emb, vid_emb, aud_emb)
                    loss, _ = fusion_model.compute_total_loss(outputs, labels, criterion)
                    val_loss += loss.item() * imgs.size(0)
                    probs = torch.sigmoid(outputs["fusion"]).cpu().numpy().squeeze()
                    lbl   = labels.cpu().numpy().squeeze()
                    metrics_tracker.update(_safe_np(probs), _safe_np(lbl))

            val_m = metrics_tracker.compute(epoch, "val", val_loss / max(len(val_loader.dataset), 1))
            print(val_m.summary())
            log_to_tensorboard(writer, val_m, tag="Fusion", log_images=(epoch % 5 == 0))
            all_val_metrics.append(val_m)

            if writer:
                writer.add_scalar("fusion/lr", optimizer.param_groups[0]["lr"], epoch)

            if epoch % 5 == 0 or epoch == args.epochs:
                print_confusion_matrix(val_m)
                plot_roc_curve(val_m, plot_dir, tag="fusion")

            d = {"auc": val_m.auc, "f1": val_m.f1, "accuracy": val_m.accuracy, "epoch": epoch}
            ckpt.save(fusion_model, d)
            if stopper.step(d):
                break

    save_metrics_log(all_val_metrics, os.path.join(args.checkpoint_dir, "fusion"))
    if writer:
        writer.close()
    print(f"\n[FusionTrain] Complete. Best AUC: {ckpt.best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSeek Fusion Training v2")
    parser.add_argument("--image_data",    required=True)
    parser.add_argument("--video_data",    required=True)
    parser.add_argument("--audio_data",    required=True)
    parser.add_argument("--image_ckpt",    default="training/checkpoints/image/best_model.pt")
    parser.add_argument("--video_ckpt",    default="training/checkpoints/video/best_model.pt")
    parser.add_argument("--audio_ckpt",    default="training/checkpoints/audio/best_model.pt")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=5e-5)
    parser.add_argument("--patience",      type=int,   default=7)
    parser.add_argument("--num_frames",    type=int,   default=16)
    parser.add_argument("--d_model",       type=int,   default=512)
    parser.add_argument("--num_workers",   type=int,   default=2)
    parser.add_argument("--checkpoint_dir",default="training/checkpoints")
    parser.add_argument("--tensorboard_dir",default="runs")
    args = parser.parse_args()
    train_fusion(args)
