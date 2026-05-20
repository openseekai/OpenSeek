"""
OpenSeek Training — Audio Detection Model
=============================================
Architecture: 2D CNN Encoder on Log-Mel+MFCC+Chroma spectrogram + Multi-Head Attention

Processes audio as a 2D spectrogram image (F×T) rather than a 1D sequence,
enabling the model to detect subtle spectral artifacts from voice cloning.

Target: AUC > 0.85
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Frequency-Time Attention ──────────────────────────────────────────────────

class FreqTimeAttention(nn.Module):
    """
    Multi-head self-attention applied to the time dimension of a spectrogram feature map.
    After CNN encoding, treats each time step as a sequence token and computes
    contextual attention across the temporal axis.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) — CNN output after adaptive pooling over frequency axis.
        Treats T time steps as a sequence.
        Returns: (B, C) global summary via mean pooling over attended sequence.
        """
        x_t = x.permute(0, 2, 1)   # (B, T, C)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        attn_out = self.norm(x_t + self.drop(attn_out))
        return attn_out.mean(dim=1)  # (B, C)


# ── Spectrogram CNN Encoder ───────────────────────────────────────────────────

class SpectrogramCNNEncoder(nn.Module):
    """
    2D CNN that processes stacked spectrogram features (Log-Mel + MFCC + Chroma)
    as a 1-channel 2D image: (B, 1, F, T) where F = 128+40+12 = 180.
    
    Architecture:
        Conv(1→32) → BN → GELU → MaxPool(freq_axis)
        Conv(32→64) → BN → GELU → MaxPool(freq_axis)
        Conv(64→128) → BN → GELU → MaxPool(freq_axis)
        Conv(128→256) → BN → GELU
        AdaptiveAvgPool over freq → (B, 256, T)
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Halve frequency axis only

            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        # Collapse frequency axis, keep time axis
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))   # (B, 256, 1, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T)
        out = self.layers(x)             # (B, 256, F', T)
        out = self.freq_pool(out)        # (B, 256, 1, T)
        out = out.squeeze(2)             # (B, 256, T)
        return out


# ── Full Audio Deepfake Detector ─────────────────────────────────────────────

class AudioDeepfakeDetector(nn.Module):
    """
    Full audio deepfake detection model.
    
    Pipeline:
      1. SpectrogramCNNEncoder: 2D CNN on (1, F, T) spectrogram → (B, 256, T)
      2. FreqTimeAttention: Multi-head attention over T → (B, 256)
      3. Classification Head: Linear(256→64) → GELU → Dropout → Linear(64→1)
    
    Input shape: (B, 1, n_features, T) where n_features = n_mels + n_mfcc + 12
    Output: (B, 1) logits (apply sigmoid externally or use BCEWithLogitsLoss)
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_attn_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = SpectrogramCNNEncoder()
        self.attention = FreqTimeAttention(embed_dim=embed_dim, num_heads=num_attn_heads, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, F, T) — stacked spectrogram features
        Returns: (B, 1) logits
        """
        cnn_out = self.cnn(x)          # (B, 256, T)
        attn_emb = self.attention(cnn_out)  # (B, 256)
        return self.head(attn_emb)     # (B, 1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-dim embedding for fusion."""
        cnn_out = self.cnn(x)
        return self.attention(cnn_out)   # (B, 256)

    def predict_audio(self, features: torch.Tensor) -> float:
        """
        Inference interface for a single audio clip.
        features: (1, F, T) — already extracted spectrogram features.
        Returns: float probability (0=Real, 1=Fake).
        """
        self.eval()
        with torch.no_grad():
            x = features.unsqueeze(0).to(next(self.parameters()).device)
            logit = self.forward(x)
            return float(torch.sigmoid(logit).cpu().item())
