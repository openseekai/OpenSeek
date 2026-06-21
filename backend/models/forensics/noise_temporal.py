import cv2
import numpy as np
from scipy.stats import pearsonr


class NoiseConsistencyAnalyzer:
    """
    Analyzes sequences of frames for consistent camera sensor noise or compression artifacts.
    Real videos generally exhibit stable noise patterns frame-to-frame.
    AI videos often generate inconsistent noise, or surfaces are artificially over-smoothed.
    """
    def __init__(self, filter_size=5):
        self.filter_size = filter_size

    def _extract_noise_residual(self, frame_gray):
        """
        Applies a median filter to isolate high-frequency pixel noise.
        """
        # Smooth the image
        smoothed = cv2.medianBlur(frame_gray, self.filter_size)

        # Calculate residual (Noise = Original - Smoothed)
        # Using int16 to allow negative residuals
        residual = cv2.subtract(frame_gray.astype(np.int16), smoothed.astype(np.int16))

        return residual

    def analyze_sequence(self, frames):
        """
        Frames: list of numpy arrays (BGR format)
        Returns: float score (0 to 1), where 1 = Highly Unnatural/AI, 0 = Real
        """
        if len(frames) < 2:
            return 0.5 # Not enough frames

        noise_residuals = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            residual = self._extract_noise_residual(gray)
            noise_residuals.append(residual)

        # Compare correlations between adjacent frames
        correlations = []
        for i in range(1, len(noise_residuals)):
            res1 = noise_residuals[i-1].flatten()
            res2 = noise_residuals[i].flatten()

            # Sub-sample for speed if images are very large (optional, but requested target <3s)
            # We'll take a random stride of 10
            stride = 10
            res1_sub = res1[::stride]
            res2_sub = res2[::stride]

            # Pearson correlation
            corr, _ = pearsonr(res1_sub, res2_sub)

            # Handle NaNs from perfectly flat regions
            if np.isnan(corr):
                corr = 0.0

            correlations.append(corr)

        avg_correlation = np.mean(correlations)

        # Real videos typically have SOME positive correlation in their noise (e.g., fixed sensor patterns)
        # AI videos often have 0 correlation (completely random noise every frame)
        # Or if they are fully smoothed, they might have high correlation but very low variance in the residual itself.

        # We need to penalize *lack* of correlation
        # Average correlation might range from e.g., 0.1 to 0.5 in real video
        score = 0.0

        if avg_correlation < 0.05:
            # Noise is completely disconnected between frames (AI hallmark)
            score += 0.8
        elif avg_correlation < 0.15:
            # Suspiciously low correlation
            score += 0.4

        # Also check variance - is the video totally devoid of texture/grain?
        var_list = [np.var(res) for res in noise_residuals]
        mean_var = np.mean(var_list)

        if mean_var < 2.0:
            # Unnaturally smooth (AI diffusion hallmark)
            score += 0.5

        return min(1.0, score)
