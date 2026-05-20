"""
OpenSeek — POST /analyze-image  +  POST /analyze-image-data
"""
import os
import base64
import tempfile
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import get_risk_level, RATE_LIMIT
from database import log_detection
from models.image_model import analyze_image
from utils.download import download_media
from utils.validators import MediaRequest
from utils.face_detector import get_face_detector

router  = APIRouter(tags=["Image Analysis"])
limiter = Limiter(key_func=get_remote_address)


# ── Shared analysis helper ────────────────────────────────────────────────────
def _run_analysis(tmp_path: str, source_url: str = "base64-upload") -> dict:
    """Run face detection + model inference on a saved image file."""
    import cv2
    img = cv2.imread(tmp_path)
    face_detected = False
    if img is not None:
        detector     = get_face_detector()
        faces        = detector.detect(img)
        face_detected = len(faces) > 0

    result     = analyze_image(tmp_path)
    auth_score = result["authenticity_score"]
    risk_level = get_risk_level(auth_score)

    return {
        "type":               "image",
        "authenticity_score": auth_score,
        "risk_level":         risk_level,
        "face_detected":      face_detected,
        "source":             source_url,
        "analysis": {
            "facial_inconsistency": result["facial_inconsistency"],
            "lighting_mismatch":    result["lighting_mismatch"],
            "gan_artifacts":        result["gan_artifacts"],
        },
    }


# ── POST /analyze-image (URL-based) ──────────────────────────────────────────
@router.post("/analyze-image", summary="Detect deepfakes in an image (by URL)")
@limiter.limit(RATE_LIMIT)
async def analyze_image_endpoint(payload: MediaRequest, request: Request) -> dict:
    """Download an image from the given URL and run the deepfake detector."""
    tmp_path: Optional[str] = None
    try:
        try:
            tmp_path = await download_media(payload.url, "image")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Download failed: {exc}")

        try:
            result = _run_analysis(tmp_path, payload.url)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Analysis error: {exc}")

        await log_detection("image", payload.url, result["authenticity_score"], result["risk_level"])
        return result

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── POST /analyze-image-data (base64 — for Chrome Extension canvas capture) ──

class ImageDataRequest(BaseModel):
    image_data: str   # data:image/png;base64,<data>  OR  raw base64
    source_url: str = "canvas-capture"


@router.post("/analyze-image-data", summary="Detect deepfakes in a base64 image")
@limiter.limit(RATE_LIMIT)
async def analyze_image_data_endpoint(payload: ImageDataRequest, request: Request) -> dict:
    """
    Accept a base64-encoded image (e.g. from HTML Canvas video frame capture)
    and run the deepfake detector. Used by Chrome Extension for Reels/TikTok
    where the video URL is a blob: or auth-gated CDN URL.
    """
    tmp_path: Optional[str] = None
    try:
        # Strip data URI prefix if present
        raw = payload.image_data
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(raw)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image data exceeds 10 MB limit")

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.flush()
        tmp.close()
        tmp_path = tmp.name

        try:
            result = _run_analysis(tmp_path, payload.source_url)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Analysis error: {exc}")

        await log_detection("image", payload.source_url, result["authenticity_score"], result["risk_level"])
        return result

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
