
import cv2
import numpy as np
from PIL import Image


class AdversarialPurifier:
    """
    Adversarial Evasion Defense (Anti-Cloaking).
    Deepfakes often contain adversarial noise (e.g., from Glaze or Nightshade)
    designed to trick neural networks into predicting "Real".
    This purifier applies non-differentiable transformations to wash off adversarial
    perturbations while preserving the underlying deepfake spatial/frequency artifacts.
    """
    def __init__(self):
        pass

    def purify(self, image_pil: Image.Image) -> Image.Image:
        """
        Applies a carefully tuned sequence of defenses:
        1. Bit-depth reduction / Quantization
        2. WebP/JPEG re-compression cycle
        3. Median filtering (removes high-frequency adversarial spikes)
        """
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # 1. Median Filtering
        # Highly effective against salt-and-pepper adversarial pixel attacks
        purified_cv = cv2.medianBlur(img_cv, 3)

        # 2. Bit-depth reduction (Quantization)
        # Reduces the color space to wash out microscopic gradient adversarial noise
        quantization_factor = 32
        purified_cv = (purified_cv // quantization_factor) * quantization_factor

        # 3. WebP Compression Cycle (Non-differentiable step)
        # Adversarial noise is usually highly sensitive to compression.
        _, encoded_img = cv2.imencode('.webp', purified_cv, [cv2.IMWRITE_WEBP_QUALITY, 90])
        decoded_cv = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

        # Convert back to PIL
        purified_pil = Image.fromarray(cv2.cvtColor(decoded_cv, cv2.COLOR_BGR2RGB))
        return purified_pil

    def extract_adversarial_noise(self, original_pil: Image.Image, purified_pil: Image.Image) -> float:
        """
        Calculates how much the image changed during purification.
        A very high change indicates the presence of aggressive adversarial cloaking.
        """
        orig_arr = np.array(original_pil).astype(np.float32)
        purified_arr = np.array(purified_pil).astype(np.float32)

        # Mean Absolute Error
        diff = np.abs(orig_arr - purified_arr)
        mae = np.mean(diff)

        # Map to anomaly score
        # Natural images change slightly (~2-4 MAE). Adversarial noise changes drastically.
        anomaly_score = min(1.0, max(0.0, (mae - 3.0) / 10.0))
        return round(anomaly_score, 4)
