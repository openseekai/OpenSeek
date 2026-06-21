import torch
import torch.nn as nn
from torchvision import models, transforms


class SpatialBranch(nn.Module):
    """
    Forensic Branch: Spatial Analysis using Swin-Base Transformer.
    Advanced training backbone for deep visual artifacts.
    """
    def __init__(self):
        super().__init__()
        # Upgrade to Swin-Base for significantly higher feature depth (1024 channels)
        backbone = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.norm = backbone.norm
        self.avgpool = backbone.avgpool

        # Binary output: Real (0) vs AI (1)
        # Swin-Base feature dim is 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # x: (Batch, 3, 224, 224)
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if return_features:
            return x
        return self.classifier(x)

def get_spatial_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
