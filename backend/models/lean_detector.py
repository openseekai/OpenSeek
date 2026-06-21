"""Lean, low-RAM detector for small CPU hosts (e.g. Railway).

Loads ONLY the benchmarked-best image classifier (haywoodsloan, AUC 0.94) plus
cheap OpenCV/PIL forensics for the heatmap. No CLIP, mediapipe, timm, or
torchvision ensembles — so it fits ~1 GB and needs no GPU. Accuracy is the same
as the full engine, whose probability came entirely from this model anyway.

Returns the SAME response dict shape as AdvancedForensicEnsemble.forward_analyze
so main.py and the UI work unchanged.
"""
import base64
import io
import logging
import os
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageChops

from models.label_mapping import ai_prob_from_classifier

logger = logging.getLogger("openseek.lean")


class LeanDetector:
    def __init__(self, device="cpu"):
        from transformers import pipeline
        self.primary_id = os.environ.get(
            "OPENSEEK_DETECTOR_MODEL", "haywoodsloan/ai-image-detector-deploy"
        )
        dev = 0 if str(device) not in ("cpu", "cpu:0") and "cuda" in str(device) else -1
        self.hf_model = pipeline(
            "image-classification", model=self.primary_id, top_k=None, device=dev
        )
        logger.info(f"[OpenSeek Lean] Loaded primary AI detector: {self.primary_id}")

    # ── cheap forensics (no torch) ────────────────────────────────────────────
    def _ela(self, img: Image.Image):
        """Error-Level-Analysis heatmap + a rough tamper score (PIL only)."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                name = tmp.name
            try:
                img.save(name, "JPEG", quality=90)
                resaved = Image.open(name)
                diff = ImageChops.difference(img, resaved)
                extrema = diff.getextrema()
                max_diff = max((ex[1] for ex in extrema), default=1) or 1
                arr = np.clip(np.array(diff) * (255.0 / max_diff), 0, 255).astype(np.uint8)
                enhanced = Image.fromarray(arr)
                score = float(min(1.0, np.std(np.array(enhanced.convert("L"))) / 48.0))
                buf = io.BytesIO()
                enhanced.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return score, f"data:image/jpeg;base64,{b64}"
            finally:
                if os.path.exists(name):
                    os.remove(name)
        except Exception as e:
            logger.warning(f"[OpenSeek Lean] ELA failed: {e}")
            return 0.0, None

    def _has_face(self, img: Image.Image) -> bool:
        try:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            return len(cascade.detectMultiScale(gray, 1.3, 5)) > 0
        except Exception:
            return False

    # ── main entry (matches AdvancedForensicEnsemble.forward_analyze) ─────────
    def forward_analyze(self, image_path: str, fast: bool = True) -> dict:
        img = Image.open(image_path).convert("RGB")

        prob = ai_prob_from_classifier(self.hf_model(img))
        if prob is None:
            prob = 0.5
        prob = round(float(min(0.99, max(0.01, prob))), 4)
        is_ai = prob > 0.5

        risk = "Low" if prob <= 0.40 else ("Medium" if prob <= 0.65 else "High")
        confidence = round(min(0.99, 0.60 + abs(prob - 0.5)), 4)
        if confidence < 0.4:
            risk = "Uncertain"

        ela_score, heatmap = self._ela(img)
        has_face = self._has_face(img)

        if is_ai:
            report = ("🚨 **Likely AI-generated.** The image classifier flagged "
                      f"generative artifacts (P(AI)={prob:.2f}).")
        else:
            report = (f"✅ **Likely authentic.** No strong AI artifacts detected "
                      f"(P(AI)={prob:.2f}).")

        return {
            "ai_probability": prob,
            "is_ai_generated": is_ai,
            "content_type": "AI Generated Image" if is_ai else "Photograph",
            "predicted_class": "AI" if is_ai else "Real",
            "confidence_score": confidence,
            "risk_level": risk,
            "manipulated_regions_heatmap": heatmap,
            "patch_manipulated_count": int(ela_score * 10) if is_ai else 0,
            "embedding_anomaly_score": 0.0,
            "face_detected": has_face,
            "facial_ai_probability": None,
            "invisible_face_anomaly": None,
            "flowchart_analysis": None,
            "pipeline": f"Lean Detector ({self.primary_id})",
            "forensic_report": report,
            "dct_anomaly_score": 0.0,
            "adversarial_noise_score": 0.0,
            "attributed_generator": "Unknown",
        }
