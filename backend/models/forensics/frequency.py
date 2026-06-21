import cv2
import numpy as np
import torch
import torch.nn as nn


class FrequencyBranch(nn.Module):
    """
    Forensic Branch: Frequency Analysis via 2D FFT.
    Detects periodic artifacts from GAN upsampling/Diffusion noise.
    """
    def __init__(self):
        super().__init__()
        # Simple CNN to classify the log-magnitude spectrum
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the log-magnitude spectrum batch (B, 1, H, W)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def extract_fft_magnitude(img_path: str, size=(224, 224)) -> torch.Tensor:
    """Extract Log-Magnitude spectrum from an image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return torch.zeros((1, 1, size[0], size[1]))

    img = cv2.resize(img, size)

    # 2D FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Log Magnitude Spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
    # Normalize
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum) + 1e-9)

    return torch.from_numpy(magnitude_spectrum).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
