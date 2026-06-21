import cv2
import numpy as np
import torch
from models.forensics.spatial import SpatialBranch, get_spatial_transform


class PatchScanner:
    """
    Forensic Tool: Patch-Level Heatmap Analysis.
    Splits image into 224x224 patches to identify local manipulations.
    """
    def __init__(self, model: SpatialBranch, device: torch.device):
        self.model = model
        self.device = device
        self.transform = get_spatial_transform()

    def generate_heatmap(self, img: np.ndarray, stride=224) -> np.ndarray:
        """
        Generate a suspiciousness heatmap.
        Optimized for speed: Uses non-overlapping patches and sampled scanning.
        """
        import PIL.Image as PILImage
        if img is None: return np.zeros((1, 1))

        h, w, _ = img.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        patch_size = 224

        # If image is very large, only scan a limited number of patches for speed
        max_patches = 16
        y_coords = range(0, h - patch_size + 1, stride)
        x_coords = range(0, w - patch_size + 1, stride)

        all_coords = [(y, x) for y in y_coords for x in x_coords]

        # If too many patches, sample them (center is most important)
        if len(all_coords) > max_patches:
            # Sort by distance to center
            cy, cx = h // 2, w // 2
            all_coords.sort(key=lambda c: (c[0] + patch_size//2 - cy)**2 + (c[1] + patch_size//2 - cx)**2)
            all_coords = all_coords[:max_patches]

        # Inference Loop
        with torch.no_grad():
            for y, x in all_coords:
                patch = img[y:y+patch_size, x:x+patch_size]
                pil_patch = PILImage.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                tensor = self.transform(pil_patch).unsqueeze(0).to(self.device)
                score = self.model(tensor).item()

                # Accumulate
                heatmap[y:y+patch_size, x:x+patch_size] += score
                counts[y:y+patch_size, x:x+patch_size] += 1

        # Average
        safe_counts = np.where(counts > 0, counts, 1)
        heatmap /= safe_counts

        # Rescale for visualization (0-255)
        heatmap_norm = (heatmap * 255).astype(np.uint8)
        return heatmap_norm

    def get_max_patch_score(self, heatmap: np.ndarray) -> float:
        """Return the highest score found in any region."""
        return float(np.max(heatmap) / 255.0) if heatmap.size > 0 else 0.0
