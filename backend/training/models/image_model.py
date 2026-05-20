"""
OpenSeek Training — Image Detection Model
=============================================
Architecture: EfficientNet-B4 (spatial) + FFT Frequency Branch
Combined into a dual-stream classifier.

Target: AUC > 0.90
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── Spectral Attention ────────────────────────────────────────────────────────

class SpectralAttentionModule(nn.Module):
    """
    Frequency-domain channel attention. Concatenates spatial features with
    their 2D FFT magnitude and learns which channels to amplify.
    Catches GAN/diffusion checkerboard artifacts in frequency space.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        freq = torch.fft.fft2(x.float(), norm="ortho")
        freq_abs = torch.abs(freq).to(x.dtype)
        combined = torch.cat([x, freq_abs], dim=1)   # (B, 2C, H, W)
        mask = self.sigmoid(self.conv1x1(combined))   # (B, C, H, W)
        return self.bn(x * mask)


# ── Frequency Branch ──────────────────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Processes log-FFT magnitude spectrum as a 1-channel image through
    a lightweight EfficientNet-B0 to extract frequency artifacts.
    Output: 512-dim embedding.
    """
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Replace input conv to accept 1 channel (grayscale FFT)
        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Keep all features, replace pool + head
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(base.features[-1][0].out_channels, 512)

    def _compute_fft(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — compute per-image grayscale FFT
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B, H, W)
        f = torch.fft.fft2(gray, norm="ortho")
        mag = torch.log(torch.abs(f) + 1e-9)
        # Normalize
        mag = (mag - mag.mean(dim=(-2, -1), keepdim=True)) / (
            mag.std(dim=(-2, -1), keepdim=True) + 1e-9
        )
        return mag.unsqueeze(1)   # (B, 1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft_img = self._compute_fft(x)    # (B, 1, H, W)
        feats = self.features(fft_img)     # (B, C, h, w)
        pooled = self.pool(feats).flatten(1)
        return self.proj(pooled)           # (B, 512)


# ── Dual-Stream Image Detector ────────────────────────────────────────────────

class DualStreamImageDetector(nn.Module):
    """
    Two-stream deepfake detector:
      Stream A: EfficientNet-B4 with SpectralAttentionModule (spatial + frequency attention)
      Stream B: Frequency Branch (log-FFT processed by EfficientNet-B0)
    
    Combined:
      Concat [1792 + 512] → 2304
      FC(2304 → 512) → GELU → Dropout(0.4)
      FC(512 → 1)
    
    Logits are returned (apply sigmoid externally or use BCEWithLogitsLoss).
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── Stream A: Spatial (EfficientNet-B4) ──
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        eff_b4 = models.efficientnet_b4(weights=weights)

        self.spatial_features = eff_b4.features
        feat_dim_b4 = 1792  # EfficientNet-B4 output channels

        # Replace last conv block's last layer with spectral attention
        self.spectral_attn = SpectralAttentionModule(feat_dim_b4)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_drop = nn.Dropout(p=0.4)

        # ── Stream B: Frequency ──
        self.freq_branch = FrequencyBranch()

        # ── Fusion Head ──
        combined_dim = feat_dim_b4 + 512  # 2304
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stream A
        spatial = self.spatial_features(x)        # (B, 1792, h, w)
        spatial = self.spectral_attn(spatial)      # (B, 1792, h, w)
        spatial = self.spatial_pool(spatial).flatten(1)   # (B, 1792)
        spatial = self.spatial_drop(spatial)

        # Stream B
        freq = self.freq_branch(x)                 # (B, 512)

        # Fusion
        combined = torch.cat([spatial, freq], dim=1)   # (B, 2304)
        return self.head(combined)                  # (B, 1) logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated embedding for fusion training."""
        with torch.no_grad():
            spatial = self.spatial_features(x)
            spatial = self.spectral_attn(spatial)
            spatial = self.spatial_pool(spatial).flatten(1)
            freq = self.freq_branch(x)
        return torch.cat([spatial, freq], dim=1)   # (B, 2304)

    def freeze_backbone(self, num_blocks_to_freeze: int = 5):
        """Freeze first N feature blocks for fine-tuning only the head."""
        for i, block in enumerate(self.spatial_features):
            if i < num_blocks_to_freeze:
                for p in block.parameters():
                    p.requires_grad = False
