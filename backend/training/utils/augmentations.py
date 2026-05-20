"""
OpenSeek Training — Data Augmentation
=========================================
Provides:
- ImageAugmentor   : Mixup, CutMix, JPEG compression, blur, noise, color jitter
- VideoAugmentor   : Frame dropout, temporal jitter
- AudioAugmentor   : Background noise, speed perturbation, pitch shifting
"""
from __future__ import annotations

import io
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


# ── Image Augmentation ────────────────────────────────────────────────────────

def jpeg_compress(pil_img: Image.Image, quality: int | None = None) -> Image.Image:
    """Simulate JPEG compression artifact at random quality 30–95."""
    q = quality or random.randint(30, 95)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()


def add_gaussian_noise(tensor: torch.Tensor, std_range=(0.01, 0.05)) -> torch.Tensor:
    """Add Gaussian noise to a normalized image tensor."""
    std = random.uniform(*std_range)
    return (tensor + torch.randn_like(tensor) * std).clamp(-3, 3)


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup augmentation: interpolates pairs of images and their labels.
    
    images : (B, C, H, W) float tensor
    labels : (B, 1) or (B,) float tensor
    Returns mixed images and labels.
    """
    if alpha <= 0:
        return images, labels

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    mixed_labels = lam * labels_a + (1 - lam) * labels_b
    return mixed_images, mixed_labels


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix augmentation: pastes a rectangular patch from one image into another.
    
    images : (B, C, H, W) float tensor
    labels : (B,) or (B,1) float tensor
    Returns augmented images and mixed labels.
    """
    if alpha <= 0:
        return images, labels

    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = images.shape
    index = torch.randperm(batch_size, device=images.device)

    # Sample random box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    actual_lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    labels_a = labels.view(-1)
    labels_b = labels[index].view(-1)
    mixed_labels = (actual_lam * labels_a + (1 - actual_lam) * labels_b).view_as(labels)

    return mixed, mixed_labels


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Standard image training transform with augmentations.
    Apply this transform to PIL images as part of the dataset.
    Mixup/CutMix are applied batch-level (see mixup_batch, cutmix_batch).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([
            transforms.Lambda(lambda t: add_gaussian_noise(t))
        ], p=0.3),
    ])


def get_jpeg_transform(image_size: int = 224) -> transforms.Compose:
    """Transform that also applies JPEG compression (for robustness to compression artifacts)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda img: jpeg_compress(img) if random.random() < 0.5 else img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Validation transform (no augmentation, only resize + normalize)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Video Augmentation ────────────────────────────────────────────────────────

def frame_dropout(
    frames: torch.Tensor,
    drop_prob: float = 0.1,
    fill_mode: str = "zero",
) -> torch.Tensor:
    """
    Randomly zero out frames in a video sequence to force temporal robustness.
    
    frames    : (T, C, H, W) tensor
    drop_prob : Probability of zeroing each frame.
    fill_mode : "zero" or "repeat" (repeat adjacent frame).
    Returns modified frames tensor.
    """
    T = frames.size(0)
    mask = torch.rand(T) > drop_prob
    result = frames.clone()
    for i in range(T):
        if not mask[i]:
            if fill_mode == "repeat" and i > 0:
                result[i] = result[i - 1]
            else:
                result[i] = torch.zeros_like(frames[i])
    return result


def temporal_jitter(frames: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
    """
    Simulate temporal misalignment by slightly shuffling frame order.
    Shift is small (max_shift frames) to preserve temporal coherence.
    """
    T = frames.size(0)
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return frames
    shifted = torch.roll(frames, shifts=shift, dims=0)
    return shifted


# ── Audio Augmentation ────────────────────────────────────────────────────────

def add_background_noise(waveform: np.ndarray, snr_db: float | None = None) -> np.ndarray:
    """
    Add Gaussian noise to waveform at a random SNR (10–30 dB).
    
    waveform : np.ndarray shape (T,)
    snr_db   : Signal-to-Noise ratio in dB. None → random [10, 30].
    """
    snr = snr_db if snr_db is not None else random.uniform(10, 30)
    signal_power = np.mean(waveform ** 2) + 1e-9
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.randn(len(waveform)) * np.sqrt(noise_power)
    return waveform + noise


def speed_perturbation(waveform: np.ndarray, sr: int, factor_range=(0.9, 1.1)) -> np.ndarray:
    """
    Speed perturbation via resampling. Changes duration without pitch shift.
    
    waveform     : np.ndarray (T,)
    sr           : Sample rate
    factor_range : (min_speed, max_speed) — factor 1.0 = unchanged.
    """
    try:
        import librosa
        factor = random.uniform(*factor_range)
        return librosa.resample(waveform, orig_sr=int(sr * factor), target_sr=sr)
    except Exception:
        return waveform


def pitch_shift_audio(waveform: np.ndarray, sr: int, semitones_range=(-2, 2)) -> np.ndarray:
    """
    Pitch shift without changing speed.
    
    waveform       : np.ndarray (T,)
    sr             : Sample rate
    semitones_range: Range for random pitch shift in semitones.
    """
    try:
        import librosa
        n_steps = random.randint(*semitones_range)
        if n_steps == 0:
            return waveform
        return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
    except Exception:
        return waveform


def apply_audio_augmentation(
    waveform: np.ndarray,
    sr: int = 16000,
    noise_prob: float = 0.3,
    speed_prob: float = 0.3,
    pitch_prob: float = 0.2,
) -> np.ndarray:
    """
    Applies a random combination of audio augmentations.
    Safe to use directly in a Dataset __getitem__ method.
    """
    if random.random() < noise_prob:
        waveform = add_background_noise(waveform)
    if random.random() < speed_prob:
        waveform = speed_perturbation(waveform, sr)
    if random.random() < pitch_prob:
        waveform = pitch_shift_audio(waveform, sr)
    return waveform
