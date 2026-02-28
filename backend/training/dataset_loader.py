"""
DeepShield Training — Unified Dataset Loader
==============================================
Production-grade dataset loading for all modalities:
  - Combines multiple datasets (FF++, Celeb-DF, DFDC, ASVspoof, WaveFake)
  - Normalizes labels (real=0, fake=1)
  - Balances using weighted sampling
  - Removes duplicate identities
  - Standardizes to 224×224 for images/video frames
  - Supports diffusion-generated samples

Usage:
    from training.dataset_loader import (
        build_all_loaders, ImageLoader, VideoLoader, AudioLoader
    )

    # Unified image loader (all datasets merged):
    train_loader, val_loader, test_loader = ImageLoader.build(
        primary_dir     = "./data/images",
        extra_dirs      = ["./data/diffusion_faces"],  # diffusion!
        batch_size      = 16,
    )

    # Or use the full pipeline builder:
    loaders = build_all_loaders(config)
"""
from __future__ import annotations

import os
import random
import json
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from PIL import Image, UnidentifiedImageError

# ── Import from existing utils and new subject_split ─────────────────────────
import sys
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from training.utils.augmentations import (
    get_train_transform,
    get_val_transform,
    apply_audio_augmentation,
)
from training.subject_split import build_subject_split, load_identity_map, split_by_subject

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ── Helper ────────────────────────────────────────────────────────────────────

def _collect_flat(directory: str, extensions: set, label: int) -> List[Tuple[str, int]]:
    """Collect files from a single directory and assign a fixed label."""
    samples = []
    if not os.path.isdir(directory):
        return samples
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in extensions:
            samples.append((os.path.join(directory, fname), label))
    return samples


def merge_and_balance(
    sample_lists: List[List[Tuple[str, int]]],
    max_per_source: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[str, int]]:
    """
    Merge multiple sample lists and optionally cap per-source size.
    Used to combine FF++, Celeb-DF, DFDC without one dataset dominating.
    
    Args:
        sample_lists   : List of sample lists from different datasets.
        max_per_source : Cap samples per dataset (normalizes contributions).
        seed           : For reproducible subsampling.
    
    Returns:
        Merged, shuffled sample list.
    """
    rng = random.Random(seed)
    merged = []
    for lst in sample_lists:
        if max_per_source and len(lst) > max_per_source:
            lst = rng.sample(lst, max_per_source)
        merged.extend(lst)
    rng.shuffle(merged)
    return merged


def make_weighted_sampler(samples: List[Tuple[str, int]]) -> WeightedRandomSampler:
    """Balance real/fake classes via weighting."""
    labels = [l for _, l in samples]
    n_real = labels.count(0)
    n_fake = labels.count(1)
    w_real = 1.0 / max(n_real, 1)
    w_fake = 1.0 / max(n_fake, 1)
    weights = [w_real if l == 0 else w_fake for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Image Datasets ────────────────────────────────────────────────────────────

class DeepfakeImageDataset(Dataset):
    """
    Image deepfake detection dataset.
    
    Augmented in training mode, clean in val/test.
    JPEG compression simulation is built into the train transform.
    224×224 standardized output.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        split: str = "train",
        image_size: int = 224,
        use_albumentations: bool = True,
    ):
        self.samples = samples
        self.split = split

        if use_albumentations:
            try:
                from training.augmentations import AlbumentationsTransform
                self.transform = AlbumentationsTransform(split=split, image_size=image_size)
            except Exception:
                self.transform = (
                    get_train_transform(image_size) if split == "train"
                    else get_val_transform(image_size)
                )
        else:
            self.transform = (
                get_train_transform(image_size) if split == "train"
                else get_val_transform(image_size)
            )

        n_r = sum(1 for _, l in samples if l == 0)
        n_f = sum(1 for _, l in samples if l == 1)
        print(f"  [ImageDS/{split}] Real: {n_r:,}  Fake: {n_f:,}  Total: {len(samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, torch.tensor(label, dtype=torch.float32)


class DiffusionRobustnessDataset(Dataset):
    """
    Dataset for diffusion-generated samples with adversarial noise augmentation.
    Combines diffusion fakes with real images and applies:
      - High-frequency noise injection (simulate diffusion artifacts)
      - Patch inconsistency simulation
      - Frequency masking
    
    Expects directory structure: real/ and fake/ subdirectories.
    Fake = Stable Diffusion / Midjourney synthetic faces.
    """
    def __init__(self, samples: List[Tuple[str, int]], split: str = "train", image_size: int = 224):
        self.samples = samples
        self.split = split
        self.base_transform = (
            get_train_transform(image_size) if split == "train"
            else get_val_transform(image_size)
        )
        n_r = sum(1 for _, l in samples if l == 0)
        n_f = sum(1 for _, l in samples if l == 1)
        print(f"  [DiffusionDS/{split}] Real: {n_r:,}  Fake: {n_f:,}  Total: {len(samples):,}")

    def _add_adversarial_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Adversarial noise: high-frequency perturbation that simulates
        diffusion model checkerboard and noise patterns.
        """
        noise = torch.zeros_like(tensor)
        # Add checkerboard pattern (common in GAN/diffusion artifacts)
        H, W = tensor.shape[-2], tensor.shape[-1]
        xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        checker = ((xx + yy) % 2).float().unsqueeze(0) * 0.05
        noise += checker
        # Add random high-frequency component
        noise += torch.randn_like(tensor) * random.uniform(0.01, 0.04)
        return (tensor + noise).clamp(-3, 3)

    def _frequency_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mask random frequency range in Fourier domain."""
        freq = torch.fft.fft2(tensor, norm="ortho")
        H, W = freq.shape[-2], freq.shape[-1]
        f_h = random.randint(0, H // 4)
        f_w = random.randint(0, W // 4)
        freq[..., f_h:f_h + H // 8, f_w:f_w + W // 8] = 0
        return torch.fft.ifft2(freq, norm="ortho").real

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.base_transform(img)
            if self.split == "train":
                if random.random() < 0.5:
                    tensor = self._add_adversarial_noise(tensor)
                if random.random() < 0.3:
                    tensor = self._frequency_mask(tensor)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, torch.tensor(label, dtype=torch.float32)


# ── ImageLoader: Multi-Dataset Factory ───────────────────────────────────────

class ImageLoader:
    """
    Factory for building image DataLoaders from multiple datasets.
    
    Usage:
        train_loader, val_loader, test_loader = ImageLoader.build(
            primary_dir   = "./data/images",
            extra_dirs    = ["./data/dfdc_frames", "./data/diffusion_faces"],
            batch_size    = 16,
            format        = "flat",  # or "ffpp", "folder"
        )
    """
    @staticmethod
    def build(
        primary_dir: str,
        extra_dirs: Optional[List[str]] = None,
        batch_size: int = 16,
        image_size: int = 224,
        num_workers: int = 4,
        max_per_source: Optional[int] = None,
        format: str = "flat",
        subject_level_split: bool = True,
        use_albumentations: bool = True,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Args:
            primary_dir          : Main dataset directory (real/ + fake/).
            extra_dirs           : Additional dataset directories to merge.
            max_per_source       : Cap samples per dataset (avoids imbalance).
            format               : Subject identity format (flat/ffpp/folder/dfdc).
            subject_level_split  : Use proper subject-level splits.
            use_albumentations   : Use Albumentations instead of torchvision.
        """
        rng = random.Random(seed)

        # Build samples from primary dataset
        if subject_level_split:
            train_s, val_s, test_s = build_subject_split(
                primary_dir, format=format, extensions=IMAGE_EXTS, seed=seed
            )
        else:
            from training.utils.datasets import _collect_files, _subject_aware_split
            all_s = _collect_files(primary_dir, IMAGE_EXTS)
            train_s, val_s, test_s = _subject_aware_split(all_s, seed=seed)

        # Merge extra datasets into training set only (no leakage to val/test)
        if extra_dirs:
            extra_samples = []
            for d in (extra_dirs or []):
                if not os.path.isdir(d):
                    print(f"  [ImageLoader] Warning: {d} not found, skipping")
                    continue
                extra = (_collect_flat(os.path.join(d, "real"), IMAGE_EXTS, 0) +
                         _collect_flat(os.path.join(d, "fake"), IMAGE_EXTS, 1))
                if max_per_source and len(extra) > max_per_source:
                    extra = rng.sample(extra, max_per_source)
                extra_samples.extend(extra)
                print(f"  [ImageLoader] Extra source '{os.path.basename(d)}': {len(extra):,}")
            train_s = merge_and_balance([train_s, extra_samples], max_per_source=max_per_source, seed=seed)

        print(f"\n  [ImageLoader] Final split: train={len(train_s):,} val={len(val_s):,} test={len(test_s):,}")

        train_ds = DeepfakeImageDataset(train_s, "train", image_size, use_albumentations)
        val_ds   = DeepfakeImageDataset(val_s,   "val",   image_size, False)
        test_ds  = DeepfakeImageDataset(test_s,  "test",  image_size, False)

        sampler  = make_weighted_sampler(train_ds.samples)
        use_pin  = torch.cuda.is_available()

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=use_pin, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=use_pin,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader, test_loader

    @staticmethod
    def cross_dataset(
        train_dirs: List[str],
        test_dirs: List[str],
        batch_size: int = 16,
        image_size: int = 224,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Cross-dataset evaluation:
        Train on: FaceForensics++ + Celeb-DF
        Test on:  DFDC (unseen dataset)
        """
        from training.subject_split import cross_dataset_split, IMAGE_EXTS as _IEXTS
        train_s, test_s = cross_dataset_split(train_dirs, test_dirs, extensions=IMAGE_EXTS)
        train_ds = DeepfakeImageDataset(train_s, "train", image_size)
        test_ds  = DeepfakeImageDataset(test_s,  "test",  image_size, use_albumentations=False)
        sampler  = make_weighted_sampler(train_ds.samples)
        use_pin  = torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=use_pin)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2, shuffle=False,
                                  num_workers=num_workers)
        return train_loader, test_loader


# ── 5-Fold Cross-Validation Support ─────────────────────────────────────────

class KFoldDatasetBuilder:
    """
    5-fold cross-validation dataset builder.
    Each fold uses 80% for training, 20% for validation,
    with the final test set always held out separately.

    Usage:
        builder = KFoldDatasetBuilder(all_train_samples, n_splits=5)
        for fold, (train_loader, val_loader) in enumerate(builder.iter_folds(batch_size=16)):
            # train on this fold...
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        n_splits: int = 5,
        seed: int = 42,
    ):
        self.n_splits = n_splits
        rng = random.Random(seed)
        self.samples = list(samples)
        rng.shuffle(self.samples)
        self.fold_size = len(self.samples) // n_splits
        print(f"  [KFold] {n_splits} folds × ~{self.fold_size:,} samples each")

    def iter_folds(
        self,
        batch_size: int = 16,
        image_size: int = 224,
        num_workers: int = 4,
    ):
        """Yields (fold_idx, train_loader, val_loader) for each fold."""
        for fold in range(self.n_splits):
            val_start = fold * self.fold_size
            val_end   = val_start + self.fold_size

            val_s   = self.samples[val_start:val_end]
            train_s = self.samples[:val_start] + self.samples[val_end:]

            train_ds = DeepfakeImageDataset(train_s, "train", image_size)
            val_ds   = DeepfakeImageDataset(val_s,   "val",   image_size, use_albumentations=False)

            sampler  = make_weighted_sampler(train_ds.samples)
            use_pin  = torch.cuda.is_available()

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=use_pin, drop_last=False,
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size * 2, shuffle=False,
                num_workers=num_workers,
            )
            yield fold + 1, train_loader, val_loader


# ── Full Pipeline Config ──────────────────────────────────────────────────────

def build_all_loaders(config: Dict[str, Any]) -> Dict[str, Tuple]:
    """
    Build all modality loaders from a single config dict.
    
    Config keys:
        image_dir      : Image dataset root (required)
        video_dir      : Video dataset root
        audio_dir      : Audio dataset root
        extra_image_dirs: List of extra image dirs (diffusion, etc.)
        batch_size_image: Image batch size (default 16)
        batch_size_video: Video batch size (default 4)
        batch_size_audio: Audio batch size (default 32)
        num_workers    : DataLoader workers (default 4)
        image_size     : Resize (default 224)
        num_frames     : Video frames per clip (default 16)
        seed           : Random seed (default 42)
    
    Returns:
        dict with keys: 'image', 'video', 'audio' each a
        (train_loader, val_loader, test_loader) tuple.
    
    Example config:
        config = {
            "image_dir": "./data/images",
            "video_dir": "./data/videos",
            "audio_dir": "./data/audio",
            "extra_image_dirs": ["./data/diffusion_faces"],
            "batch_size_image": 16,
        }
        loaders = build_all_loaders(config)
        train_img, val_img, test_img = loaders["image"]
    """
    loaders = {}

    if config.get("image_dir") and os.path.isdir(config["image_dir"]):
        print("\n[DataLoader] Building IMAGE loaders...")
        loaders["image"] = ImageLoader.build(
            primary_dir        = config["image_dir"],
            extra_dirs         = config.get("extra_image_dirs"),
            batch_size         = config.get("batch_size_image", 16),
            image_size         = config.get("image_size", 224),
            num_workers        = config.get("num_workers", 4),
            max_per_source     = config.get("max_per_source"),
            use_albumentations = config.get("use_albumentations", True),
            seed               = config.get("seed", 42),
        )

    if config.get("video_dir") and os.path.isdir(config["video_dir"]):
        print("\n[DataLoader] Building VIDEO loaders...")
        from training.utils.datasets import build_video_loaders
        loaders["video"] = build_video_loaders(
            data_dir    = config["video_dir"],
            batch_size  = config.get("batch_size_video", 4),
            num_frames  = config.get("num_frames", 16),
            num_workers = config.get("num_workers", 4),
            seed        = config.get("seed", 42),
        )

    if config.get("audio_dir") and os.path.isdir(config["audio_dir"]):
        print("\n[DataLoader] Building AUDIO loaders...")
        from training.utils.datasets import build_audio_loaders
        loaders["audio"] = build_audio_loaders(
            data_dir    = config["audio_dir"],
            batch_size  = config.get("batch_size_audio", 32),
            num_workers = config.get("num_workers", 4),
            seed        = config.get("seed", 42),
        )

    return loaders
