"""
OpenSeek Training — Video Detection Model
=============================================
Architecture: EfficientNet-B0 frame encoder + 4-layer Temporal Transformer

Replaces the old EfficientNet+LSTM with a Transformer that captures
long-range temporal dependencies (face morphing, flickering, texture drift).

Target: AUC > 0.88
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import timm


# ── Learnable Positional Encoding ──────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Learnable positional embedding for a fixed sequence length.
    Preferred over sinusoidal for short sequences (≤32 frames).
    """
    def __init__(self, d_model: int, max_seq_len: int = 32):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ── Temporal Transformer Detector ────────────────────────────────────────────

class TemporalTransformerDetector(nn.Module):
    """
    Temporal Transformer-based video deepfake detector.
    
    Pipeline:
      1. EfficientNet-B0 (pretrained, partial freeze) → per-frame features (1280-dim)
      2. Linear projection → d_model dimensions
      3. Learnable positional encoding
      4. 4-layer Transformer Encoder (8 attention heads)
      5. [CLS] token classification
      6. Binary classifier head
    
    Args:
        num_frames   : Sequence length (number of frames per clip).
        d_model      : Transformer embedding dimension.
        nhead        : Number of attention heads (d_model must be divisible by nhead).
        num_layers   : Number of Transformer encoder layers.
        pretrained   : Whether to use ImageNet-pretrained EfficientNet-B0.
        freeze_blocks: Number of EfficientNet feature blocks to freeze.
    
    Logits are returned (apply sigmoid externally or use BCEWithLogitsLoss).
    """
    def __init__(
        self,
        num_frames: int = 16,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_blocks: int = 4,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.d_model = d_model

        # ── Frame Feature Extractor ──
        self.frame_encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        frame_feat_dim = self.frame_encoder.num_features  # 1280 for EfficientNet-B0

        # Freeze early blocks for stability
        self._freeze_blocks(freeze_blocks)

        # ── Input Projection ──
        self.input_proj = nn.Sequential(
            nn.Linear(frame_feat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── CLS Token ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional Encoding ──
        self.pos_enc = PositionalEncoding(d_model, max_seq_len=num_frames + 1)

        # ── Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Classifier Head ──
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _freeze_blocks(self, n: int):
        """Freeze first n feature blocks of EfficientNet-B0."""
        count = 0
        for name, module in self.frame_encoder.named_children():
            if count < n:
                for p in module.parameters():
                    p.requires_grad = False
                count += 1

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) — batch of video clips
        Returns: (B, 1) logits
        """
        B, T, C, H, W = x.shape

        # Extract per-frame features
        x_flat = x.view(B * T, C, H, W)
        feats = self.frame_encoder(x_flat)          # (B*T, 1280)
        feats = feats.view(B, T, -1)                # (B, T, 1280)

        # Project to d_model
        feats = self.input_proj(feats)              # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        seq = torch.cat([cls, feats], dim=1)        # (B, T+1, d_model)

        # Positional encoding
        seq = self.pos_enc(seq)

        # Transformer
        out = self.transformer(seq)                  # (B, T+1, d_model)

        # CLS token output → classify
        cls_out = out[:, 0, :]                       # (B, d_model)
        return self.head(cls_out)                    # (B, 1) logits

    def predict_video(self, frames_tensor: torch.Tensor) -> float:
        """
        Inference interface for a single video.
        frames_tensor: (T, C, H, W)
        Returns: float probability (0=Real, 1=Fake).
        """
        self.eval()
        with torch.no_grad():
            x = frames_tensor.unsqueeze(0).to(next(self.parameters()).device)
            if next(self.parameters()).dtype == torch.float16:
                x = x.half()
            logit = self.forward(x)
            return float(torch.sigmoid(logit).cpu().item())

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS embedding for fusion training."""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.frame_encoder(x_flat).view(B, T, -1)
        feats = self.input_proj(feats)
        cls = self.cls_token.expand(B, -1, -1)
        seq = self.pos_enc(torch.cat([cls, feats], dim=1))
        out = self.transformer(seq)
        return out[:, 0, :]   # (B, d_model)
