"""
OpenSeek Configuration

All runtime settings are environment-driven so the same image can be promoted
across dev / staging / prod without code changes (12-factor).
"""
import logging
import os

# ─── Server ───────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO").upper()

_LOGGING_CONFIGURED = False

def setup_logging() -> logging.Logger:
    """Configure root logging once and return the app logger.

    Honours LOG_LEVEL. Safe to call multiple times (idempotent).
    """
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        _LOGGING_CONFIGURED = True
    return logging.getLogger("openseek")

# ─── Security / CORS ──────────────────────────────────────────────────────────
# Comma-separated allowlist of origins, e.g.
#   ALLOWED_ORIGINS="https://open-seek-ai.vercel.app,chrome-extension://<id>"
# Defaults to "*" so local dev keeps working; ALWAYS set an explicit list in prod.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
] or ["*"]

# ─── Rate Limiting ────────────────────────────────────────────────────────────
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

# ─── Download Limits ──────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB   = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_VIDEO_SIZE_MB   = 100
MAX_AUDIO_SIZE_MB   = 20
DOWNLOAD_TIMEOUT_S  = 30   # seconds per request

# ─── Risk Thresholds ─────────────────────────────────────────────────────────
def get_risk_level(score: float) -> str:
    """Map authenticity score (0–100) to risk label (higher score → higher risk)."""
    if score <= 15:
        return "Low"
    elif score <= 55:
        return "Medium"
    else:
        return "High"

# ─── Allowed MIME types ───────────────────────────────────────────────────────
ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/mpeg", "video/webm", "video/quicktime",
    "video/x-msvideo", "video/x-matroska",
}
ALLOWED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/ogg", "audio/flac", "audio/aac", "audio/webm",
}

# ─── Model Settings ───────────────────────────────────────────────────────────
IMAGE_SIZE         = 224          # input resolution for CNN
MAX_VIDEO_FRAMES   = 60           # max frames to analyse per video
VIDEO_AUDIO_WEIGHT = 0.6          # weight for video track score
AUDIO_WEIGHT       = 0.4          # weight for audio track score

# ─── Private / Reserved IP ranges (CIDR-ish prefix list) ─────────────────────
BLOCKED_IP_PREFIXES = (
    "10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
    "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
    "127.", "0.", "169.254.", "::1", "fc", "fd",
)

# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "openseek.db")
