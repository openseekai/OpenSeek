import numpy as np
import torch
import torch.nn as nn


class NoiseBranch(nn.Module):
    """
    Forensic Branch: Noise Residual Analysis via SRM (Spatial Rich Model).
    Isolates sensor-level noise to detect unnatural sharpening or smoothing.
    """
    def __init__(self):
        super().__init__()
        # SRM Kernels (fixed, non-trainable)
        self.srm_conv = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self._init_srm_kernels()

        # Classification backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _init_srm_kernels(self):
        """Standard SRM kernels for noise extraction."""
        # 1. First-order
        k1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        # 2. Second-order
        k2 = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]
        ])
        # 3. Edge
        k3 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ])

        kernels = torch.from_numpy(np.stack([k1, k2, k3])).float().unsqueeze(1)
        self.srm_conv.weight.data = kernels
        self.srm_conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is grayscale image (B, 1, H, W)
        with torch.no_grad():
            x = self.srm_conv(x)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
