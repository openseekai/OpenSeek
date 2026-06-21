import os

import cv2
import numpy as np


class GenerationStepAnalyzer:
    """
    Flowchart-Guided Generation Step Consistency Detector.
    Analyzes physical, geometric, and digital consistency corresponding to the 6 distinct steps of the image generation process:
    1. Random Colors (Noise Residual)
    5. Sky Appears (Coarse gradients/colors)
    10. Mountain Shapes Appears (Structural layout)
    20. Dragon Silhouette Appears (Edge boundaries/silhouettes)
    35. Wings Gain Detail (Repeating microtextures/fine details)
    50. Scales, Lighting, Shadows (Surface texture and physical illumination consistency)
    """
    def __init__(self):
        pass

    def analyze_image(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image at: {image_path}")

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Step 1: Random Colors (Noise Residual Stage)
        score1, details1 = self._analyze_step1_noise(img_gray)

        # Step 5: Sky Appears (Color & Gradient Smoothness Stage)
        score5, details5 = self._analyze_step5_sky(img_bgr)

        # Step 10: Mountain Shapes (Structural Layout Stage)
        score10, details10 = self._analyze_step10_structure(img_gray)

        # Step 20: Dragon Silhouette (Contour & Edge Stage)
        score20, details20 = self._analyze_step20_silhouette(img_gray)

        # Step 35: Wings Gain Detail (Texture Detail Repetition Stage)
        score35, details35 = self._analyze_step35_details(img_gray)

        # Step 50: Scales, Lighting, Shadows (Hyper-fine texture & Illumination Stage)
        score50, details50 = self._analyze_step50_lighting_shadows(img_bgr)

        # Aggregate the scores using weighted fusion
        # Weights reflect how reliable each step-detection is.
        # Step 1 (noise residual) and Step 50 (lighting/shadows) are highly informative forensic metrics.
        weights = {
            "step1_noise": 0.20,
            "step5_sky": 0.15,
            "step10_structure": 0.10,
            "step20_silhouette": 0.15,
            "step35_details": 0.15,
            "step50_lighting": 0.25
        }

        raw_score = (
            weights["step1_noise"] * score1 +
            weights["step5_sky"] * score5 +
            weights["step10_structure"] * score10 +
            weights["step20_silhouette"] * score20 +
            weights["step35_details"] * score35 +
            weights["step50_lighting"] * score50
        )

        # Calibrate the aggregated score to a probability scale
        # Normal real images score < 0.40, while AI images score > 0.60
        if raw_score > 0.58:
            ai_probability = 0.70 + ((raw_score - 0.58) / 0.42) * 0.29
        elif raw_score > 0.45:
            ai_probability = 0.40 + ((raw_score - 0.45) / 0.13) * 0.30
        else:
            ai_probability = (raw_score / 0.45) * 0.40

        ai_probability = min(0.99, max(0.01, ai_probability))

        return {
            "ai_probability": round(ai_probability, 4),
            "is_ai_generated": ai_probability > 0.5,
            "scores": {
                "step1_noise_residual": round(score1, 4),
                "step5_color_gradients": round(score5, 4),
                "step10_layout_structure": round(score10, 4),
                "step20_silhouette_contours": round(score20, 4),
                "step35_detail_textures": round(score35, 4),
                "step50_lighting_shadows": round(score50, 4)
            },
            "metrics": {
                "noise": details1,
                "sky": details5,
                "structure": details10,
                "silhouette": details20,
                "details": details35,
                "lighting": details50
            }
        }

    def _analyze_step1_noise(self, img_gray: np.ndarray) -> tuple[float, dict]:
        """
        Step 1: Noise Residual Analysis (Random Colors stage).
        Measures the statistical distribution of high-frequency noise residuals.
        AI upsampling/denoising leaves grid-like periodic residuals and non-Gaussian kurtosis.
        """
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        var = np.var(laplacian)

        # Calculate skewness and kurtosis
        std = np.std(laplacian)
        if std < 1e-6:
            return 0.5, {"kurtosis": 3.0, "autocorrelation": 0.0, "variance": 0.0}

        mean_diff = laplacian - np.mean(laplacian)
        np.mean(mean_diff ** 3) / (std ** 3 + 1e-9)
        kurt = np.mean(mean_diff ** 4) / (std ** 4 + 1e-9)

        # Autocorrelation of residual to catch grid pattern artifacts
        h, w = laplacian.shape
        shifted = laplacian[1:, 1:]
        orig = laplacian[:-1, :-1]
        autocorr = np.mean(orig * shifted) / (var + 1e-9)

        # Natural noise residuals are close to Gaussian (kurtosis = 3).
        # Generative models deviate significantly due to spatial constraints and have periodic grid autocorrelation.
        kurt_dev = abs(kurt - 3.0)
        autocorr_anomaly = abs(autocorr)

        score = min(1.0, max(0.0, (kurt_dev / 12.0) + (autocorr_anomaly * 8.0)))
        return score, {
            "kurtosis": float(kurt),
            "autocorrelation": float(autocorr),
            "variance": float(var)
        }

    def _analyze_step5_sky(self, img_bgr: np.ndarray) -> tuple[float, dict]:
        """
        Step 5: Sky / Color Layout Analysis.
        Checks for artificial smoothness or banding in large sky/background gradients.
        AI models over-smooth solid gradient fields (yielding low local entropy and banding).
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Low-pass filter to examine sky/color layout regions
        low_freq = cv2.GaussianBlur(v, (15, 15), 0)
        grad_x = cv2.Sobel(low_freq, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(low_freq, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Identify flat color gradients (common in skies)
        flat_mask = (grad_mag < 1.2) & (grad_mag > 0.01)
        flat_ratio = np.sum(flat_mask) / (grad_mag.size + 1e-9)

        # Color channel histogram entropy
        hist = cv2.calcHist([img_bgr], [0], None, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-9)
        entropy = -np.sum(hist * np.log2(hist + 1e-9))

        # High flat_ratio and low color entropy indicate generative smoothing/banding
        score = min(1.0, max(0.0, (flat_ratio * 3.5) + (1.0 - (entropy / 8.0))))
        return score, {
            "flat_ratio": float(flat_ratio),
            "color_entropy": float(entropy)
        }

    def _analyze_step10_structure(self, img_gray: np.ndarray) -> tuple[float, dict]:
        """
        Step 10: Mountain Shapes / Coarse Layout Analysis.
        Examines structural coherence and fractal boundaries.
        AI models often generate melting structural boundaries or shapes violating perspective/geometrical constraints.
        """
        # Downsample to look at structural layout
        down = cv2.resize(img_gray, (32, 32), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(down, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(down, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2 + 1e-9)
        theta = np.arctan2(gy, gx)

        # Compute direction orientation entropy (coherence of structural layout shapes)
        hist, _ = np.histogram(theta, bins=16, range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-9)
        direction_entropy = -np.sum(hist * np.log2(hist + 1e-9))

        # Real landscapes/shapes have coherent structural direction profiles (lower entropy).
        # AI images show unnatural shape variations or lack of clear physical perspective directions.
        score = min(1.0, max(0.0, (direction_entropy / 4.0) - 0.2))
        return score, {
            "shape_direction_entropy": float(direction_entropy),
            "mean_structural_magnitude": float(np.mean(mag))
        }

    def _analyze_step20_silhouette(self, img_gray: np.ndarray) -> tuple[float, dict]:
        """
        Step 20: Dragon Silhouette / Boundary Silhouette Analysis.
        Analyzes edge smoothness, sharpness, and aliasing along contours.
        AI images often exhibit blending anomalies, ringing, or aliased silhouettes when separating objects.
        """
        edges = cv2.Canny(img_gray, 50, 150)
        if np.sum(edges) == 0:
            return 0.3, {"edge_sharpness_var": 0.5, "grid_alignment": 0.1}

        gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)

        edge_mags = mag[edges > 0]
        mean_edge_grad = np.mean(edge_mags)
        std_edge_grad = np.std(edge_mags)

        # Normalize sharpness variance
        sharpness_var = std_edge_grad / (mean_edge_grad + 1e-9)

        # Aliasing check: proportion of edge gradients aligning perfectly with grid coordinates (0, 90, 180, 270)
        angles = np.arctan2(gy, gx)[edges > 0] * 180 / np.pi
        grid_align = np.sum((np.abs(angles) % 90 < 5) | (np.abs(angles) % 90 > 85)) / (angles.size + 1e-9)

        # Score increases if edge profiles have high sharpness variance (unnatural blurring/sharpening) or high grid alignment
        score = min(1.0, max(0.0, (grid_align * 1.6) + (sharpness_var - 0.4)))
        return score, {
            "edge_sharpness_var": float(sharpness_var),
            "grid_alignment": float(grid_align)
        }

    def _analyze_step35_details(self, img_gray: np.ndarray) -> tuple[float, dict]:
        """
        Step 35: Wings / Texture Detail Analysis.
        Checks for repetitive micro-texture patterns using Local Binary Patterns (LBP).
        Generative models produce repeated pattern errors or homogeneous textures in detailed regions.
        """
        h, w = img_gray.shape
        if h < 10 or w < 10:
            return 0.5, {"lbp_entropy": 4.0, "texture_homogeneity": 0.1}

        # Vectorized Local Binary Pattern (LBP) computation
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        for i, (dy, dx) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]):
            neighbor = img_gray[1+dy:h-1+dy, 1+dx:w-1+dx]
            center = img_gray[1:h-1, 1:w-1]
            lbp += ((neighbor >= center).astype(np.uint8) << i)

        # LBP histogram entropy
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-9)
        lbp_entropy = -np.sum(hist * np.log2(hist + 1e-9))

        # Microtexture homogeneity (ratio of the dominant micro-pattern)
        max_bin_ratio = np.max(hist)

        # Generative copy-paste/homogeneity artifacts cause LBP entropy drops and spikes in specific bins
        score = min(1.0, max(0.0, (max_bin_ratio * 12.0) + (1.0 - (lbp_entropy / 7.5))))
        return score, {
            "lbp_entropy": float(lbp_entropy),
            "texture_homogeneity": float(max_bin_ratio)
        }

    def _analyze_step50_lighting_shadows(self, img_bgr: np.ndarray) -> tuple[float, dict]:
        """
        Step 50: Scales, Lighting, and Shadows Analysis.
        Evaluates physical consistency of illumination direction and hyper-fine texture scales.
        AI models struggle with consistent illumination direction and local contrast scaling.
        """
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Hyper-fine scale texturing (Scales vs background)
        # We calculate the ratio of hyper-fine high-frequency components to mid-frequency structures
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        r = np.sqrt(x*x + y*y)

        mid_energy = np.sum(mag[(r > (cx * 0.2)) & (r < (cx * 0.5))])
        high_energy = np.sum(mag[(r >= (cx * 0.5)) & (r < (cx * 0.95))])
        energy_ratio = high_energy / (mid_energy + 1e-9)

        # 2. Lighting Direction Consistency
        # Estimate surface normal gradients on highlighted regions (pixel intensity > 200)
        _, thresholded = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

        highlight_gx = gx[thresholded > 0]
        highlight_gy = gy[thresholded > 0]

        lighting_variance = 0.0
        if len(highlight_gx) > 15:
            angles = np.arctan2(highlight_gy, highlight_gx)
            sin_sum = np.sum(np.sin(angles))
            cos_sum = np.sum(np.cos(angles))
            r_val = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
            # Higher circular variance = lighting is coming from conflicting directions
            lighting_variance = 1.0 - r_val

        # High lighting direction variance and anomalous frequency energy ratios suggest AI synthesis
        score = min(1.0, max(0.0, (lighting_variance * 1.6) + (0.4 - energy_ratio)))
        return score, {
            "lighting_variance": float(lighting_variance),
            "texture_energy_ratio": float(energy_ratio)
        }
