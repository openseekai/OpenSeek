"""
OpenSeek — MediaPipe-based face detector.
"""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

_mp_face = mp.solutions.face_detection


class FaceDetector:
    """Lightweight, CPU-friendly face detector backed by MediaPipe."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self._detector = _mp_face.FaceDetection(
            model_selection=0,           # 0 = short-range (≤2 m), fast
            min_detection_confidence=min_confidence,
        )

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR image.

        Returns
        -------
        list[dict]
            Each dict has keys: ``bbox`` (x, y, w, h) and ``confidence``.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        faces: list[dict] = []

        if not results.detections:
            return faces

        h, w = image_bgr.shape[:2]
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = int(bb.xmin * w)
            y = int(bb.ymin * h)
            bw = int(bb.width  * w)
            bh = int(bb.height * h)
            faces.append({
                "bbox": (max(x, 0), max(y, 0), bw, bh),
                "confidence": round(det.score[0], 4),
            })

        return faces

    def close(self) -> None:
        self._detector.close()


# Module-level singleton (created lazily)
_detector_instance: FaceDetector | None = None


def get_face_detector() -> FaceDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector()
    return _detector_instance
