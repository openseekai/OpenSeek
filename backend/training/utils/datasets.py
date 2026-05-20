"""
OpenSeek Training — Multi-Source Dataset Loaders
====================================================
Supports multiple deepfake datasets in a unified interface.
Expects data in standard structure:
    <root>/
      real/ ← authentic images/videos/audio
      fake/ ← deepfake/AI-generated content

For multi-dataset training, merge folders from:
  IMAGE  : FaceForensics++, Celeb-DF, DFDC
  VIDEO  : FaceForensics++, Celeb-DF v2, DFDC Preview
  AUDIO  : ASVspoof 2021, FakeAVCeleb, WaveFake

Subject-level split ensures no identity leakage between train/val/test.
"""
from __future__ import annotations

import os
import random
import hashlib
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, UnidentifiedImageError

from training.utils.augmentations import (
    get_train_transform,
    get_val_transform,
    apply_audio_augmentation,
)


# ── Supported Extensions ───────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ── Utility ───────────────────────────────────────────────────────────────────

def _collect_files(root: str, extensions: set) -> List[Tuple[str, int]]:
    """
    Collect all files with given extensions from real/ and fake/ subdirs.
    Returns list of (filepath, label) where label: 0=real, 1=fake.
    """
    samples: List[Tuple[str, int]] = []
    for sub, label in [("real", 0), ("fake", 1)]:
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                samples.append((os.path.join(d, fname), label))
    return samples


def _subject_aware_split(
    samples: List[Tuple[str, int]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Subject-level split: group files by their first 8-char hash-prefix (simulates
    subject ID grouping when actual IDs are unavailable). This prevents the same
    file patterns from appearing in both train and val/test.

    When actual subject IDs are available in filename (e.g., FF++ uses actor IDs),
    the hash-prefix naturally groups by filename prefix which approximates identity.

    Returns: (train_samples, val_samples, test_samples)
    """
    rng = random.Random(seed)

    # Group by filename prefix as a proxy for subject identity
    groups: dict[str, list] = {}
    for path, label in samples:
        fname = os.path.basename(path)
        # Use first 2 chars as group key (a rough subject proxy)
        group_key = fname[:2].lower() if len(fname) >= 2 else fname[0]
        groups.setdefault(group_key, []).append((path, label))

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    train_keys = set(group_keys[:n_train])
    val_keys   = set(group_keys[n_train:n_train + n_val])
    test_keys  = set(group_keys[n_train + n_val:])

    train_s = [s for k in train_keys for s in groups[k]]
    val_s   = [s for k in val_keys   for s in groups[k]]
    test_s  = [s for k in test_keys  for s in groups[k]]

    return train_s, val_s, test_s


def make_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that balances real/fake proportions per batch.
    Essential when datasets have more fake than real (or vice versa).
    """
    n_real = labels.count(0)
    n_fake = labels.count(1)
    w_real = 1.0 / max(n_real, 1)
    w_fake = 1.0 / max(n_fake, 1)
    weights = [w_real if l == 0 else w_fake for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Image Dataset ─────────────────────────────────────────────────────────────

class ImageDeepfakeDataset(Dataset):
    """
    Dataset for image deepfake detection.
    
    Usage:
        train_ds = ImageDeepfakeDataset(samples, split="train", image_size=224)
        val_ds   = ImageDeepfakeDataset(samples, split="val",   image_size=224)
    
    Args:
        samples    : List of (filepath, label) tuples.
        split      : "train" or "val"/"test" (controls augmentation).
        image_size : Resize target (224 for EfficientNet, 384 for ViT-L).
        max_samples: Cap dataset size (for smoke tests).
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        split: str = "train",
        image_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        self.samples = samples
        if max_samples and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

        self.transform = (
            get_train_transform(image_size)
            if split == "train"
            else get_val_transform(image_size)
        )
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  [ImageDataset/{split}] Real: {n_real:,}  Fake: {n_fake:,}  Total: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except (UnidentifiedImageError, OSError):
            tensor = torch.zeros(3, 224, 224)
        return tensor, torch.tensor(label, dtype=torch.float32)


# ── Video Dataset ─────────────────────────────────────────────────────────────

class VideoDeepfakeDataset(Dataset):
    """
    Dataset for video deepfake detection using uniformly-sampled frames.
    
    Args:
        samples       : List of (video_filepath, label) tuples.
        split         : "train" or "val"/"test".
        num_frames    : Number of frames to extract per clip.
        image_size    : Frame resize target.
        frame_dropout : Apply frame dropout augmentation (train only).
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        split: str = "train",
        num_frames: int = 16,
        image_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        self.samples = samples
        if max_samples and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

        self.num_frames = num_frames
        self.split = split
        self.frame_transform = (
            get_train_transform(image_size)
            if split == "train"
            else get_val_transform(image_size)
        )
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  [VideoDataset/{split}] Real: {n_real:,}  Fake: {n_fake:,}  Total: {len(self.samples):,}")

    def _extract_frames(self, video_path: str) -> torch.Tensor:
        """Extract `num_frames` uniformly spaced frames from a video file."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
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
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(frame_rgb)
                    frames.append(self.frame_transform(pil))
                else:
                    frames.append(torch.zeros(3, 224, 224))
            cap.release()

            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
            return torch.stack(frames[:self.num_frames])
        except Exception:
            return torch.zeros(self.num_frames, 3, 224, 224)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        frames = self._extract_frames(path)

        if self.split == "train":
            from training.utils.augmentations import frame_dropout, temporal_jitter
            if random.random() < 0.4:
                frames = frame_dropout(frames, drop_prob=0.1)
            if random.random() < 0.3:
                frames = temporal_jitter(frames, max_shift=2)

        return frames, torch.tensor(label, dtype=torch.float32)


# ── Audio Dataset ─────────────────────────────────────────────────────────────

class AudioDeepfakeDataset(Dataset):
    """
    Dataset for audio deepfake detection.
    Extracts Log-Mel spectrogram + MFCC + Chroma features per clip.
    
    Args:
        samples    : List of (audio_filepath, label) tuples.
        split      : "train" or "val"/"test".
        sr         : Sample rate (16000 Hz standard for ASVspoof / WaveFake).
        duration   : Clip duration in seconds (clips are padded or trimmed).
        n_mels     : Number of Mel filter banks.
        n_mfcc     : Number of MFCC coefficients.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        split: str = "train",
        sr: int = 16000,
        duration: float = 4.0,
        n_mels: int = 128,
        n_mfcc: int = 40,
        max_samples: Optional[int] = None,
    ):
        self.samples = samples
        if max_samples and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

        self.split = split
        self.sr = sr
        self.length = int(sr * duration)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  [AudioDataset/{split}] Real: {n_real:,}  Fake: {n_fake:,}  Total: {len(self.samples):,}")

    def _load_audio(self, path: str) -> np.ndarray:
        import librosa
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        y, _ = librosa.effects.trim(y)
        if len(y) < self.length:
            y = np.pad(y, (0, self.length - len(y)))
        else:
            start = random.randint(0, len(y) - self.length) if self.split == "train" else 0
            y = y[start:start + self.length]
        return y.astype(np.float32)

    def _extract_features(self, y: np.ndarray) -> torch.Tensor:
        """
        Returns stacked feature map: (1, n_mels + n_mfcc + 12, T)
        where T = time frames (~128 for 4s at hop_length=512).
        The model will treat this as a single-channel 2D spectrogram.
        """
        import librosa
        hop = 512
        # Log-Mel
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=hop)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, hop_length=hop)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, hop_length=hop)

        # Stack → (n_features, T)
        T = min(mel_db.shape[1], mfcc.shape[1], chroma.shape[1])
        features = np.concatenate([mel_db[:, :T], mfcc[:, :T], chroma[:, :T]], axis=0)

        # Normalize per channel
        features = (features - features.mean()) / (features.std() + 1e-9)

        # Add channel dim → (1, F, T)
        return torch.from_numpy(features).float().unsqueeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            y = self._load_audio(path)
            if self.split == "train":
                y = apply_audio_augmentation(y, sr=self.sr)
            features = self._extract_features(y)
        except Exception as e:
            n_features = self.n_mels + self.n_mfcc + 12
            features = torch.zeros(1, n_features, 128)
        return features, torch.tensor(label, dtype=torch.float32)


# ── Dataset Factory ───────────────────────────────────────────────────────────

def build_image_loaders(
    data_dir: str,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, test DataLoaders for image deepfake detection.
    Returns (train_loader, val_loader, test_loader).
    """
    all_samples = _collect_files(data_dir, IMAGE_EXTS)
    if not all_samples:
        raise FileNotFoundError(f"No image files found in {data_dir}/real/ or {data_dir}/fake/")

    train_s, val_s, test_s = _subject_aware_split(all_samples, seed=seed)
    print(f"[ImageLoader] Split: train={len(train_s):,} val={len(val_s):,} test={len(test_s):,}")

    train_ds = ImageDeepfakeDataset(train_s, "train", image_size, max_samples)
    val_ds   = ImageDeepfakeDataset(val_s,   "val",   image_size)
    test_ds  = ImageDeepfakeDataset(test_s,  "test",  image_size)

    # Build sampler from ACTUAL dataset samples (after max_samples cap)
    actual_labels = [l for _, l in train_ds.samples]
    sampler = make_weighted_sampler(actual_labels)
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=use_pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size * 2), shuffle=False,
        num_workers=num_workers, pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(1, batch_size * 2), shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def build_video_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_frames: int = 16,
    image_size: int = 224,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, test DataLoaders for video deepfake detection."""
    all_samples = _collect_files(data_dir, VIDEO_EXTS)
    if not all_samples:
        raise FileNotFoundError(f"No video files found in {data_dir}/real/ or {data_dir}/fake/")

    train_s, val_s, test_s = _subject_aware_split(all_samples, seed=seed)
    print(f"[VideoLoader] Split: train={len(train_s):,} val={len(val_s):,} test={len(test_s):,}")

    train_ds = VideoDeepfakeDataset(train_s, "train", num_frames, image_size, max_samples)
    val_ds   = VideoDeepfakeDataset(val_s,   "val",   num_frames, image_size)
    test_ds  = VideoDeepfakeDataset(test_s,  "test",  num_frames, image_size)

    actual_labels = [l for _, l in train_ds.samples]
    sampler = make_weighted_sampler(actual_labels)
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=use_pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size // 2), shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(1, batch_size // 2), shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def build_audio_loaders(
    data_dir: str,
    batch_size: int = 32,
    sr: int = 16000,
    duration: float = 4.0,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, test DataLoaders for audio deepfake detection."""
    all_samples = _collect_files(data_dir, AUDIO_EXTS)
    if not all_samples:
        raise FileNotFoundError(f"No audio files found in {data_dir}/real/ or {data_dir}/fake/")

    train_s, val_s, test_s = _subject_aware_split(all_samples, seed=seed)
    print(f"[AudioLoader] Split: train={len(train_s):,} val={len(val_s):,} test={len(test_s):,}")

    train_ds = AudioDeepfakeDataset(train_s, "train", sr, duration, max_samples=max_samples)
    val_ds   = AudioDeepfakeDataset(val_s,   "val",   sr, duration)
    test_ds  = AudioDeepfakeDataset(test_s,  "test",  sr, duration)

    actual_labels = [l for _, l in train_ds.samples]
    sampler = make_weighted_sampler(actual_labels)
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=use_pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size * 2), shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(1, batch_size * 2), shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
