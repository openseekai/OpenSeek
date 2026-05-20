"""
OpenSeek — Fusion Training Script
======================================
Model  : MultiModalFusionDetector (Cross-Modal Attention)
Strategy: Load pretrained image/video/audio branches → freeze → train fusion head
Loss   : Weighted multi-task: image_loss + video_loss + audio_loss + 2× fusion_loss
Metrics: AUC, F1, Precision, Recall
Target : AUC > 0.92

Usage:
    cd /path/to/backend
    python training/fusion_train.py \\
        --image_data ./data/images \\
        --video_data ./data/videos \\
        --audio_data ./data/audio \\
        --image_ckpt ./training/checkpoints/image/best_model.pt \\
        --video_ckpt ./training/checkpoints/video/best_model.pt \\
        --audio_ckpt ./training/checkpoints/audio/best_model.pt \\
        --epochs 20 \\
        --batch_size 8
"""
from __future__ import annotations

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from training.models.image_model import DualStreamImageDetector
from training.models.video_model import TemporalTransformerDetector
from training.models.audio_model import AudioDeepfakeDetector
from training.models.fusion_model import MultiModalFusionDetector, load_branch_embedder
from training.utils.datasets import (
    build_image_loaders, build_video_loaders, build_audio_loaders,
    _collect_files, _subject_aware_split, IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS,
    make_weighted_sampler
)
from training.utils.losses import CombinedLoss
from training.utils.metrics import MetricsTracker, save_metrics_log, plot_roc_curve, print_confusion_matrix
from training.utils.training_utils import (
    WarmupCosineScheduler,
    CheckpointManager,
    EarlyStopping,
    get_amp_scaler,
    clip_and_step,
    EpochTimer,
)


def _safe_np(arr):
    a = np.array(arr).squeeze()
    return a.reshape(-1) if a.ndim >= 1 else np.array([float(a)])


class TriModalDataset(Dataset):
    """
    Dataset that returns (image, video_frames, audio_spec, label) for fusion training.
    
    Assumes a paired multimedia dataset where each sample ID has an image, a video clip,
    and an audio clip. For unpaired datasets, uses the same label but samples from
    each modality independently (common approach in practice).

    For simplicity, this implementation picks one image, one video frame sequence,
    and one audio spectrogram per sample from the respective datasets.
    Each is sampled from the same real/fake pool (label-matched but unpaired).
    """
    def __init__(self, img_samples, vid_samples, aud_samples, split="train", num_frames=16):
        # Group by label
        self.img_real = [s for s in img_samples if s[1] == 0]
        self.img_fake = [s for s in img_samples if s[1] == 1]
        self.vid_real = [s for s in vid_samples if s[1] == 0]
        self.vid_fake = [s for s in vid_samples if s[1] == 1]
        self.aud_real = [s for s in aud_samples if s[1] == 0]
        self.aud_fake = [s for s in aud_samples if s[1] == 1]

        self.split = split
        self.num_frames = num_frames
        self.length = min(
            len(img_samples), len(vid_samples), len(aud_samples)
        )
        self.labels = [0] * (self.length // 2) + [1] * (self.length // 2)
        random.shuffle(self.labels)

        from training.utils.augmentations import get_train_transform, get_val_transform
        self.img_tf = get_train_transform(224) if split == "train" else get_val_transform(224)

        from training.utils.datasets import AudioDeepfakeDataset
        # We'll load audio features inline using the AudioDeepfakeDataset logic
        self._aud_ds_real = None
        self._aud_ds_fake = None

        print(f"  [FusionDataset/{split}] {self.length} paired triplets")

    def _load_image(self, path):
        from PIL import Image as PILImage
        try:
            img = PILImage.open(path).convert("RGB")
            return self.img_tf(img)
        except Exception:
            return torch.zeros(3, 224, 224)

    def _load_video_frames(self, path):
        try:
            import cv2
            from PIL import Image as PILImage
            from training.utils.augmentations import get_val_transform
            tf = get_val_transform(224)
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return torch.zeros(self.num_frames, 3, 224, 224)
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    import cv2 as cv2_
                    frame_rgb = cv2_.cvtColor(frame, cv2_.COLOR_BGR2RGB)
                    pil = PILImage.fromarray(frame_rgb)
                    frames.append(tf(pil))
                else:
                    frames.append(torch.zeros(3, 224, 224))
            cap.release()
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
            return torch.stack(frames[:self.num_frames])
        except Exception:
            return torch.zeros(self.num_frames, 3, 224, 224)

    def _load_audio(self, path):
        try:
            import librosa
            sr = 16000
            y, _ = librosa.load(path, sr=sr, mono=True)
            target_len = sr * 4
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
            hop = 512
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=hop)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
            T = min(mel_db.shape[1], mfcc.shape[1], chroma.shape[1])
            features = np.concatenate([mel_db[:, :T], mfcc[:, :T], chroma[:, :T]], axis=0)
            features = (features - features.mean()) / (features.std() + 1e-9)
            return torch.from_numpy(features).float().unsqueeze(0)
        except Exception:
            return torch.zeros(1, 180, 128)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.labels[idx % len(self.labels)]
        pool_img = self.img_real if label == 0 else self.img_fake
        pool_vid = self.vid_real if label == 0 else self.vid_fake
        pool_aud = self.aud_real if label == 0 else self.aud_fake

        img_path = random.choice(pool_img)[0] if pool_img else None
        vid_path = random.choice(pool_vid)[0] if pool_vid else None
        aud_path = random.choice(pool_aud)[0] if pool_aud else None

        img_tensor = self._load_image(img_path)   if img_path else torch.zeros(3, 224, 224)
        vid_tensor = self._load_video_frames(vid_path) if vid_path else torch.zeros(self.num_frames, 3, 224, 224)
        aud_tensor = self._load_audio(aud_path)   if aud_path else torch.zeros(1, 180, 128)

        return img_tensor, vid_tensor, aud_tensor, torch.tensor(label, dtype=torch.float32)


def train_fusion(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[FusionTrain] Device: {device}")

    # ── Load Pretrained Branches ─────────────────────────────────────────────
    print("\n[FusionTrain] Loading pretrained branch models…")
    img_branch = load_branch_embedder(DualStreamImageDetector, args.image_ckpt, device, freeze=True)
    vid_branch = load_branch_embedder(TemporalTransformerDetector, args.video_ckpt, device, freeze=True)
    aud_branch = load_branch_embedder(AudioDeepfakeDetector, args.audio_ckpt, device, freeze=True)

    # ── Fusion Model ─────────────────────────────────────────────────────────
    fusion_model = MultiModalFusionDetector(
        image_embed_dim=2304,
        video_embed_dim=args.d_model,
        audio_embed_dim=256,
        shared_dim=256,
    ).to(device)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[FusionTrain] Building fusion datasets…")
    img_all = _collect_files(args.image_data, IMAGE_EXTS)
    vid_all = _collect_files(args.video_data, VIDEO_EXTS)
    aud_all = _collect_files(args.audio_data, AUDIO_EXTS)

    img_tr, img_val, img_test = _subject_aware_split(img_all)
    vid_tr, vid_val, vid_test = _subject_aware_split(vid_all)
    aud_tr, aud_val, aud_test = _subject_aware_split(aud_all)

    train_ds = TriModalDataset(img_tr, vid_tr, aud_tr, "train", num_frames=args.num_frames)
    val_ds   = TriModalDataset(img_val, vid_val, aud_val, "val", num_frames=args.num_frames)

    labels_tr = train_ds.labels
    sampler = make_weighted_sampler(labels_tr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # ── Optimizer (only fusion parameters) ───────────────────────────────────
    criterion = CombinedLoss()
    optimizer = optim.AdamW(fusion_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=3, max_epochs=args.epochs)
    scaler = get_amp_scaler(device)

    ckpt = CheckpointManager(args.checkpoint_dir, tag="fusion", monitor="auc")
    stopper = EarlyStopping(patience=args.patience, monitor="auc")
    metrics_tracker = MetricsTracker()
    all_val_metrics = []
    plot_dir = os.path.join(args.checkpoint_dir, "fusion", "plots")

    print(f"\n{'='*60}")
    print(f"  Training Fusion Module — {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        with EpochTimer(epoch, args.epochs, "Fusion"):

            # ── Train ────────────────────────────────────────────────────────
            fusion_model.train()
            img_branch.eval(); vid_branch.eval(); aud_branch.eval()
            train_loss = 0.0
            train_tracker = MetricsTracker()

            for imgs, vids, auds, labels in train_loader:
                imgs  = imgs.to(device, non_blocking=True)
                vids  = vids.to(device, non_blocking=True)
                auds  = auds.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).view(-1, 1)

                optimizer.zero_grad()

                with torch.no_grad():
                    img_emb = img_branch.get_embedding(imgs)
                    vid_emb = vid_branch.get_embedding(vids)
                    aud_emb = aud_branch.get_embedding(auds)

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = fusion_model(img_emb, vid_emb, aud_emb)
                        loss, breakdown = fusion_model.compute_total_loss(outputs, labels, criterion)
                    scaler.scale(loss).backward()
                    clip_and_step(optimizer, fusion_model, scaler)
                else:
                    outputs = fusion_model(img_emb, vid_emb, aud_emb)
                    loss, breakdown = fusion_model.compute_total_loss(outputs, labels, criterion)
                    loss.backward()
                    clip_and_step(optimizer, fusion_model, None)

                train_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(outputs["fusion"].detach()).cpu().numpy().squeeze()
                lbl   = labels.cpu().numpy().squeeze()
                train_tracker.update(_safe_np(probs), _safe_np(lbl))

            scheduler.step()
            avg_train_loss = train_loss / max(len(train_loader.dataset), 1)
            train_m = train_tracker.compute(epoch, "train", avg_train_loss)
            print(train_m.summary())

            # ── Validate ─────────────────────────────────────────────────────
            fusion_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, vids, auds, labels in val_loader:
                    imgs  = imgs.to(device, non_blocking=True)
                    vids  = vids.to(device, non_blocking=True)
                    auds  = auds.to(device, non_blocking=True)
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

            avg_val_loss = val_loss / max(len(val_loader.dataset), 1)
            val_m = metrics_tracker.compute(epoch, "val", avg_val_loss)
            print(val_m.summary())
            all_val_metrics.append(val_m)

            if epoch % 5 == 0 or epoch == args.epochs:
                print_confusion_matrix(val_m)
                plot_roc_curve(val_m, plot_dir, tag="fusion")

            metrics_dict = {"auc": val_m.auc, "f1": val_m.f1, "accuracy": val_m.accuracy, "epoch": epoch}
            ckpt.save(fusion_model, metrics_dict)
            if stopper.step(metrics_dict):
                break

    save_metrics_log(all_val_metrics, os.path.join(args.checkpoint_dir, "fusion"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSeek Fusion Training")
    parser.add_argument("--image_data",    required=True, help="Image dataset dir")
    parser.add_argument("--video_data",    required=True, help="Video dataset dir")
    parser.add_argument("--audio_data",    required=True, help="Audio dataset dir")
    parser.add_argument("--image_ckpt",    default="training/checkpoints/image/best_model.pt")
    parser.add_argument("--video_ckpt",    default="training/checkpoints/video/best_model.pt")
    parser.add_argument("--audio_ckpt",    default="training/checkpoints/audio/best_model.pt")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=5e-5)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--patience",      type=int,   default=7)
    parser.add_argument("--num_frames",    type=int,   default=16)
    parser.add_argument("--d_model",       type=int,   default=512)
    parser.add_argument("--num_workers",   type=int,   default=2)
    parser.add_argument("--checkpoint_dir", default="training/checkpoints")
    args = parser.parse_args()

    train_fusion(args)
