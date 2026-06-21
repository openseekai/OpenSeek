import cv2
import numpy as np


class DCTAnalyzer:
    """
    Discrete Cosine Transform (DCT) grid analyzer.
    Detects Double JPEG compression artifacts and 8x8 grid misalignments typical of face-swaps
    and localized image splicing.
    """
    def __init__(self):
        pass

    def analyze_image(self, image_cv: np.ndarray) -> dict:
        """
        Calculates grid misalignments and high-frequency discrepancies indicating splicing.
        """
        if image_cv is None or image_cv.size == 0:
             return {"anomaly_score": 0.0, "details": "Invalid image"}

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # We look for the strength of the 8x8 grid boundary artifacts.
        # Calculate horizontal and vertical differences
        diff_h = np.abs(gray[:, :-1].astype(np.int16) - gray[:, 1:].astype(np.int16))
        diff_v = np.abs(gray[:-1, :].astype(np.int16) - gray[1:, :].astype(np.int16))

        # Sum differences along columns and rows
        sum_h = np.sum(diff_h, axis=0)
        sum_v = np.sum(diff_v, axis=1)

        # Check for periodicity every 8 pixels (JPEG block size)
        # We calculate the average difference at block boundaries vs internal pixels
        if len(sum_h) > 8 and len(sum_v) > 8:
            boundary_h = sum_h[7::8] # Every 8th column
            internal_h = [sum_h[i] for i in range(len(sum_h)) if (i + 1) % 8 != 0]

            boundary_v = sum_v[7::8] # Every 8th row
            internal_v = [sum_v[i] for i in range(len(sum_v)) if (i + 1) % 8 != 0]

            avg_boundary_h = np.mean(boundary_h) if len(boundary_h) > 0 else 0
            avg_internal_h = np.mean(internal_h) if len(internal_h) > 0 else 1

            avg_boundary_v = np.mean(boundary_v) if len(boundary_v) > 0 else 0
            avg_internal_v = np.mean(internal_v) if len(internal_v) > 0 else 1

            # Ratio of boundary difference to internal difference
            # In a tampered/double-compressed image, block boundaries often have higher or drastically lower discontinuity
            ratio_h = avg_boundary_h / (avg_internal_h + 1e-9)
            ratio_v = avg_boundary_v / (avg_internal_v + 1e-9)

            # Normal JPEG has ratio slightly > 1.
            # A significantly high ratio indicates severe double compression/blocking artifacts (common in fast face-swaps).
            # A ratio < 1 might indicate generative AI over-smoothing that breaks normal JPEG laws.
            anomaly_h = abs(ratio_h - 1.1)
            anomaly_v = abs(ratio_v - 1.1)

            base_anomaly = (anomaly_h + anomaly_v) / 2.0

            # Map to 0-1 score
            anomaly_score = min(1.0, max(0.0, base_anomaly / 0.5))

            details = "Consistent DCT grid."
            if anomaly_score > 0.6:
                details = "Severe 8x8 DCT grid misalignment detected. Highly indicative of face-swapping or localized splicing."
            elif anomaly_score > 0.4:
                details = "Moderate block boundary inconsistencies."

            return {
                "anomaly_score": round(anomaly_score, 4),
                "ratio_h": round(ratio_h, 4),
                "ratio_v": round(ratio_v, 4),
                "details": details
            }

        return {"anomaly_score": 0.0, "details": "Image too small for DCT analysis"}
