"""
OpenSeek Training — Loss Functions
=====================================
- FocalLoss         : Handles class imbalance (gamma=2, alpha=0.75)
- LabelSmoothingBCE : Prevents overconfident binary predictions (smoothing=0.1)
- CombinedLoss      : Weighted sum of both for training stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Reduces the relative loss for well-classified examples, focusing on hard ones.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma (float): Focusing parameter. Higher = more focus on hard examples.
        alpha (float): Weighting factor for positive class (fake samples).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds: raw logits or probabilities (B, 1) or (B,)
        # targets: (B, 1) or (B,) float in {0, 1}
        preds = preds.view(-1)
        targets = targets.view(-1)

        bce = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        # Convert logits → probs for focal weight
        preds_prob = torch.sigmoid(preds)
        p_t = preds_prob * targets + (1 - preds_prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross Entropy with label smoothing.
    Smooths hard 0/1 targets to [smoothing/2, 1-smoothing/2].
    Prevents overconfident predictions and improves calibration.
    
    Args:
        smoothing (float): Label smoothing factor (0.1 recommended).
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)
        # Smooth labels: 0 → eps/2, 1 → 1 - eps/2
        eps = self.smoothing
        smooth_targets = targets * (1 - eps) + (1 - targets) * eps / 2
        return F.binary_cross_entropy_with_logits(preds, smooth_targets)


class CombinedLoss(nn.Module):
    """
    Weighted combination of FocalLoss + LabelSmoothingBCE.
    Gives the best of both: handles imbalance + prevents overconfidence.
    
    Args:
        focal_weight   : Weight for FocalLoss contribution.
        smooth_weight  : Weight for LabelSmoothingBCE contribution.
        gamma          : Focal loss gamma parameter.
        alpha          : Focal loss alpha parameter.
        smoothing      : Label smoothing factor.
    """
    def __init__(
        self,
        focal_weight: float = 0.7,
        smooth_weight: float = 0.3,
        gamma: float = 2.0,
        alpha: float = 0.75,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.smooth = LabelSmoothingBCE(smoothing=smoothing)
        self.fw = focal_weight
        self.sw = smooth_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(preds, targets) + self.sw * self.smooth(preds, targets)


class MultiTaskLoss(nn.Module):
    """
    For fusion training: weighted sum of per-modality losses + fusion loss.
    
    total = img_w * img_loss + vid_w * vid_loss + aud_w * aud_loss + fusion_w * fusion_loss
    """
    def __init__(
        self,
        image_weight: float = 0.25,
        video_weight: float = 0.25,
        audio_weight: float = 0.25,
        fusion_weight: float = 0.25,
    ):
        super().__init__()
        self.base_loss = CombinedLoss()
        self.iw = image_weight
        self.vw = video_weight
        self.aw = audio_weight
        self.fw = fusion_weight

    def forward(
        self,
        image_pred: torch.Tensor,
        video_pred: torch.Tensor,
        audio_pred: torch.Tensor,
        fusion_pred: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.iw * self.base_loss(image_pred, targets)
            + self.vw * self.base_loss(video_pred, targets)
            + self.aw * self.base_loss(audio_pred, targets)
            + self.fw * self.base_loss(fusion_pred, targets)
        )
