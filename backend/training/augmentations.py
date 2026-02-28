"""
DeepShield Training — Production Augmentation Pipeline
========================================================
Uses Albumentations for maximum augmentation flexibility.
Falls back to torchvision if Albumentations is not installed.

Augmentations implemented:
  IMAGE:
    Spatial      : Random crop, horizontal flip, rotation ±10°, scale jitter
    Color        : Color jitter, random gamma, hue-sat shift
    Compression  : JPEG artifacts (q=30–90), downscale+upscale
    Noise        : Gaussian noise, salt & pepper, ISO noise
    Blur         : Motion blur, Gaussian blur, median blur
    Frequency    : Random frequency masking (FFT-space)
    Adversarial  : High-frequency noise injection (diffusion artifacts)
    Mixup/CutMix : Batch-level (torch-based, unchanged)

  VIDEO:
    Frame dropout : Zero out random frames
    Temporal jitter: Shuffle frame order

  AUDIO:
    Background noise injection
    Speed perturbation (0.9–1.1)
    Pitch shift
    SpecAugment masking (frequency + time masking)

Install Albumentations:
    pip install albumentations
"""
from __future__ import annotations

import io
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

# Try Albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALB = True
except ImportError:
    _ALB = False
    print("[Augment] albumentations not installed — falling back to torchvision. "
          "Install with: pip install albumentations")


# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Albumentations Transform Wrapper ──────────────────────────────────────────

class AlbumentationsTransform:
    """
    Albumentations-based augmentation pipeline for deepfake detection.
    
    Training augmentations:
      - Spatial: crop, flip, rotate, scale
      - Compression: JPEG artifacts (crucial for deepfake detection!)
      - Noise: Gaussian, ISO camera noise, salt & pepper
      - Blur: motion blur, Gaussian blur (simulate video compression)
      - Color: RGB shift, HSV shift, random gamma
    
    Handles both PIL Images and numpy arrays.
    Output: normalized torch.Tensor (C, H, W) in ImageNet range.
    """
    def __init__(self, split: str = "train", image_size: int = 224):
        self.split = split
        self.image_size = image_size

        if not _ALB:
            # Fallback to torchvision
            from training.utils.augmentations import get_train_transform, get_val_transform
            self._fallback = (
                get_train_transform(image_size) if split == "train"
                else get_val_transform(image_size)
            )
            self._use_fallback = True
        else:
            self._use_fallback = False
            if split == "train":
                self._transform = self._build_train(image_size)
            else:
                self._transform = self._build_val(image_size)

    @staticmethod
    def _build_train(size: int) -> "A.Compose":
        return A.Compose([
            # ── Spatial ─────────────────────────────────────────────────────
            A.RandomResizedCrop(height=size, width=size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),

            # ── Color / Photometric ─────────────────────────────────────────
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),

            # ── Compression simulation (KEY for deepfake detection!) ─────────
            A.OneOf([
                A.ImageCompression(quality_lower=30, quality_upper=90, p=1.0),
                A.Downscale(scale_min=0.5, scale_max=0.85,
                            interpolation={"downscale": 1, "upscale": 2}, p=1.0),
            ], p=0.5),

            # ── Noise ─────────────────────────────────────────────────────────
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.3),

            # ── Blur (simulate video compression artifacts) ────────────────
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),

            # ── Normalize + to tensor ──────────────────────────────────────
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    @staticmethod
    def _build_val(size: int) -> "A.Compose":
        return A.Compose([
            A.Resize(height=size, width=size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def __call__(self, img) -> torch.Tensor:
        if self._use_fallback:
            return self._fallback(img)

        # Accept PIL or numpy
        if isinstance(img, Image.Image):
            img_arr = np.array(img)
        else:
            img_arr = img

        result = self._transform(image=img_arr)
        return result["image"]   # Already a torch.Tensor with shape (C, H, W)


# ── Standalone torchvision transforms (always available) ───────────────────

def add_gaussian_noise(tensor: torch.Tensor, std_range=(0.01, 0.05)) -> torch.Tensor:
    std = random.uniform(*std_range)
    return (tensor + torch.randn_like(tensor) * std).clamp(-3, 3)


def jpeg_compress(pil_img: Image.Image, quality: Optional[int] = None) -> Image.Image:
    q = quality or random.randint(30, 95)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Torchvision fallback train transform."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        ], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.Lambda(lambda img: jpeg_compress(img) if random.random() < 0.4 else img),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomApply([transforms.Lambda(add_gaussian_noise)], p=0.3),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Frequency Masking (Image) ─────────────────────────────────────────────────

def random_frequency_mask(
    tensor: torch.Tensor,
    mask_ratio: float = 0.1,
) -> torch.Tensor:
    """
    Random frequency masking in Fourier space.
    Zeroes out a random rectangular region in the 2D FFT magnitude spectrum.
    Forces model to not rely on specific frequency patterns.
    
    tensor : (C, H, W) float image tensor.
    mask_ratio : Fraction of frequency components to mask (0.05–0.2 recommended).
    """
    C, H, W = tensor.shape
    freq = torch.fft.fft2(tensor.float(), norm="ortho")

    # Random rectangular region in frequency space
    h_mask = max(1, int(H * mask_ratio))
    w_mask = max(1, int(W * mask_ratio))
    h0 = random.randint(0, H - h_mask)
    w0 = random.randint(0, W - w_mask)
    freq[:, h0:h0 + h_mask, w0:w0 + w_mask] = 0

    result = torch.fft.ifft2(freq, norm="ortho").real
    return result.to(tensor.dtype)


# ── Adversarial / Diffusion Robustness Augmentation ──────────────────────────

def add_adversarial_noise(
    tensor: torch.Tensor,
    epsilon: float = 0.02,
    noise_type: str = "fgsm_style",
) -> torch.Tensor:
    """
    Add adversarial-style perturbations to improve robustness against:
      - Diffusion model artifacts (checkerboard, over-smooth skin)
      - High-frequency noise patterns typical of neural synthesizers
    
    tensor     : (C, H, W) normalized image tensor.
    epsilon    : Perturbation magnitude (0.01–0.05).
    noise_type : "fgsm_style" | "checkerboard" | "random_hf"
    """
    if noise_type == "fgsm_style":
        # Random gradient-like perturbation (sign of Gaussian noise)
        noise = torch.sign(torch.randn_like(tensor)) * epsilon
    elif noise_type == "checkerboard":
        # Checkerboard pattern (GAN/diffusion artifact simulation)
        H, W = tensor.shape[-2], tensor.shape[-1]
        xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        checker = ((xx + yy) % 2).float().unsqueeze(0).expand(3, -1, -1) * epsilon
        noise = checker
    else:  # random_hf
        # High-frequency random noise
        noise = torch.randn_like(tensor) * epsilon
        # Keep only high-frequency components
        freq = torch.fft.fft2(noise, norm="ortho")
        H, W = freq.shape[-2], freq.shape[-1]
        # Zero out low-frequency center
        cx, cy = H // 4, W // 4
        freq[:, H // 2 - cx:H // 2 + cx, W // 2 - cy:W // 2 + cy] = 0
        noise = torch.fft.ifft2(freq, norm="ortho").real

    return (tensor + noise).clamp(-3, 3)


# ── Batch-Level: Mixup and CutMix ────────────────────────────────────────────

def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup: interpolate pairs of images and labels."""
    if alpha <= 0:
        return images, labels
    lam = np.random.beta(alpha, alpha)
    B = images.size(0)
    idx = torch.randperm(B, device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, lam * labels.view(-1) + (1 - lam) * labels[idx].view(-1)


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix: paste a rectangular patch from one image into another."""
    if alpha <= 0:
        return images, labels
    lam = np.random.beta(alpha, alpha)
    B, _, H, W = images.shape
    idx  = torch.randperm(B, device=images.device)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    actual_lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    mixed_lbls = actual_lam * labels.view(-1) + (1 - actual_lam) * labels[idx].view(-1)
    return mixed, mixed_lbls


# ── Video Augmentations ────────────────────────────────────────────────────────

def frame_dropout(frames: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """Drop random frames, replace with zeros."""
    mask = torch.rand(frames.size(0)) > drop_prob
    result = frames.clone()
    for i in range(frames.size(0)):
        if not mask[i]:
            result[i] = torch.zeros_like(frames[i])
    return result


def temporal_jitter(frames: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
    """Slightly shuffle frame order to simulate temporal misalignment."""
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(frames, shifts=shift, dims=0) if shift != 0 else frames


# ── Audio Augmentations ────────────────────────────────────────────────────────

def add_background_noise(waveform: np.ndarray, snr_db: Optional[float] = None) -> np.ndarray:
    snr = snr_db or random.uniform(10, 30)
    pwr = np.mean(waveform ** 2) + 1e-9
    noise_pwr = pwr / (10 ** (snr / 10))
    return waveform + np.random.randn(len(waveform)) * np.sqrt(noise_pwr)


def speed_perturbation(waveform: np.ndarray, sr: int, factor_range=(0.9, 1.1)) -> np.ndarray:
    try:
        import librosa
        factor = random.uniform(*factor_range)
        return librosa.resample(waveform, orig_sr=int(sr * factor), target_sr=sr)
    except Exception:
        return waveform


def pitch_shift_audio(waveform: np.ndarray, sr: int, semitones_range=(-2, 2)) -> np.ndarray:
    try:
        import librosa
        n = random.randint(*semitones_range)
        return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n) if n != 0 else waveform
    except Exception:
        return waveform


def specaugment(
    spec: torch.Tensor,
    freq_mask_param: int = 20,
    time_mask_param: int = 25,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> torch.Tensor:
    """
    SpecAugment: mask frequency bands and time steps in spectrogram.
    More aggressive than what was in the old audio_train.py.
    
    spec : (1, F, T) or (F, T) spectrogram tensor.
    """
    result = spec.clone()
    F_dim = spec.shape[-2]
    T_dim = spec.shape[-1]

    for _ in range(n_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, F_dim - f))
        result[..., f0:f0 + f, :] = 0

    for _ in range(n_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, T_dim - t))
        result[..., :, t0:t0 + t] = 0

    return result


def apply_audio_augmentation(
    waveform: np.ndarray,
    sr: int = 16000,
    noise_prob: float = 0.35,
    speed_prob: float = 0.3,
    pitch_prob: float = 0.2,
) -> np.ndarray:
    if random.random() < noise_prob:
        waveform = add_background_noise(waveform)
    if random.random() < speed_prob:
        waveform = speed_perturbation(waveform, sr)
    if random.random() < pitch_prob:
        waveform = pitch_shift_audio(waveform, sr)
    return waveform
