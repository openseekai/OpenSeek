"""
OpenSeek Training — Multimodal Fusion Model
===============================================
Architecture: Cross-modal attention over image + video + audio embeddings
followed by a joint classifier.

Training strategy:
  1. Train each branch independently first (see image_train.py etc.)
  2. Load pretrained branches, freeze them
  3. Train ONLY the fusion head + cross-modal attention

Target: AUC > 0.92
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Cross-Modal Attention ──────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Soft attention that allows each modality to attend to information in the others.
    
    For each query modality q, computes attention over [key1, key2] context
    and produces an enriched query representation.
    
    Args:
        embed_dim : Dimension of each modality embedding (after projection).
        num_heads : Attention heads.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_img_to_ctx  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_vid_to_ctx  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_aud_to_ctx  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        img_emb: torch.Tensor,   # (B, D)
        vid_emb: torch.Tensor,   # (B, D)
        aud_emb: torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        """
        Returns: (B, 3*D) — concatenated attended modality embeddings.
        """
        # Add sequence dimension for attention: (B, 1, D)
        img_q = img_emb.unsqueeze(1)
        vid_q = vid_emb.unsqueeze(1)
        aud_q = aud_emb.unsqueeze(1)

        # Cross-context sequence for each query: [other_mod1, other_mod2] → (B, 2, D)
        img_ctx = torch.cat([vid_q, aud_q], dim=1)  # (B, 2, D)
        vid_ctx = torch.cat([img_q, aud_q], dim=1)
        aud_ctx = torch.cat([img_q, vid_q], dim=1)

        img_out, _ = self.attn_img_to_ctx(img_q, img_ctx, img_ctx)  # (B, 1, D)
        vid_out, _ = self.attn_vid_to_ctx(vid_q, vid_ctx, vid_ctx)
        aud_out, _ = self.attn_aud_to_ctx(aud_q, aud_ctx, aud_ctx)

        # Residual + norm
        img_final = self.norm(img_emb + self.drop(img_out.squeeze(1)))
        vid_final = self.norm(vid_emb + self.drop(vid_out.squeeze(1)))
        aud_final = self.norm(aud_emb + self.drop(aud_out.squeeze(1)))

        return torch.cat([img_final, vid_final, aud_final], dim=1)  # (B, 3*D)


# ── Modality Projection Heads ─────────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """
    Projects raw branch embeddings to a shared dimension for cross-modal attention.
    Also acts as an adapter so each branch can have different embedding sizes.
    """
    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ── Fusion Model ──────────────────────────────────────────────────────────────

class MultiModalFusionDetector(nn.Module):
    """
    Multimodal deepfake detector that fuses image, video, and audio branches.
    
    The branch models are loaded externally and passed in as `branch_models`.
    They are frozen during fusion training — only the fusion layers are updated.
    
    Architecture:
        image_branch  → ModalityProjector(2304→256) ──┐
        video_branch  → ModalityProjector(512→256)  ──┼→ CrossModalAttention → (B, 768)
        audio_branch  → ModalityProjector(256→256)  ──┘
            
        Fusion Head:
            Linear(768→512) → GELU → Dropout(0.3)
            Linear(512→128) → GELU → Dropout(0.2)
            Linear(128→1)   → logits
    
    Per-modality heads are also kept for auxiliary loss computation during training.
    
    Args:
        image_embed_dim : Image branch embedding dimension (2304 for dual-stream B4).
        video_embed_dim : Video branch embedding dimension (512 = d_model of transformer).
        audio_embed_dim : Audio branch embedding dimension (256 from CNN-attn).
        shared_dim      : Shared projection dimension for cross-modal attention.
    """

    def __init__(
        self,
        image_embed_dim: int = 2304,
        video_embed_dim: int = 512,
        audio_embed_dim: int = 256,
        shared_dim: int = 256,
        num_cross_attn_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Input Projection ──
        self.img_proj  = ModalityProjector(image_embed_dim, shared_dim, dropout)
        self.vid_proj  = ModalityProjector(video_embed_dim, shared_dim, dropout)
        self.aud_proj  = ModalityProjector(audio_embed_dim, shared_dim, dropout)

        # ── Per-modality Auxiliary Outputs (for auxiliary losses) ──
        self.img_aux_head = nn.Linear(shared_dim, 1)
        self.vid_aux_head = nn.Linear(shared_dim, 1)
        self.aud_aux_head = nn.Linear(shared_dim, 1)

        # ── Cross-Modal Attention ──
        self.cross_attn = CrossModalAttention(shared_dim, num_cross_attn_heads, dropout=0.1)

        # ── Fusion Classifier ──
        fusion_dim = shared_dim * 3   # 768
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        img_emb: torch.Tensor,    # (B, image_embed_dim)
        vid_emb: torch.Tensor,    # (B, video_embed_dim)
        aud_emb: torch.Tensor,    # (B, audio_embed_dim)
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with:
          'fusion'  : (B, 1) fusion logits
          'image'   : (B, 1) image auxiliary logits
          'video'   : (B, 1) video auxiliary logits
          'audio'   : (B, 1) audio auxiliary logits
        """
        # Project all modalities to shared dim
        img_p = self.img_proj(img_emb)   # (B, shared_dim)
        vid_p = self.vid_proj(vid_emb)
        aud_p = self.aud_proj(aud_emb)

        # Auxiliary per-modality outputs
        img_aux = self.img_aux_head(img_p)   # (B, 1)
        vid_aux = self.vid_aux_head(vid_p)
        aud_aux = self.aud_aux_head(aud_p)

        # Cross-modal attention
        fused = self.cross_attn(img_p, vid_p, aud_p)   # (B, 768)

        # Fusion prediction
        fusion_logit = self.fusion_head(fused)   # (B, 1)

        return {
            "fusion": fusion_logit,
            "image":  img_aux,
            "video":  vid_aux,
            "audio":  aud_aux,
        }

    def compute_total_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        loss_fn: nn.Module,
        weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute weighted multi-task loss.
        
        weights default: {"fusion": 0.5, "image": 0.2, "video": 0.2, "audio": 0.1}
        Returns (total_loss, loss_breakdown_dict).
        """
        w = weights or {"fusion": 0.5, "image": 0.2, "video": 0.2, "audio": 0.1}
        breakdown = {}
        total = torch.tensor(0.0, device=targets.device)

        for key, logit in outputs.items():
            l = loss_fn(logit, targets)
            breakdown[key] = l.item()
            total = total + w.get(key, 0.1) * l

        return total, breakdown


# ── Checkpoint Loading Helpers ────────────────────────────────────────────────

def load_branch_embedder(
    branch_class,
    checkpoint_path: str,
    device: torch.device,
    freeze: bool = True,
):
    """
    Load a pretrained branch model and optionally freeze all its parameters.
    Returns the model ready for embedding extraction.
    """
    model = branch_class()
    if os.path.exists(checkpoint_path):
        try:
            sd = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(sd)
            print(f"  [Fusion] Loaded {branch_class.__name__} from {checkpoint_path}")
        except Exception as e:
            print(f"  [Fusion] Warning: could not load {checkpoint_path}: {e}")
    else:
        print(f"  [Fusion] No checkpoint at {checkpoint_path} — using random weights")

    model = model.to(device).eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        print(f"  [Fusion] {branch_class.__name__} frozen.")
    return model
