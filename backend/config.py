"""
OpenSeek Configuration
"""
import os

# ─── Server ───────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ─── Rate Limiting ────────────────────────────────────────────────────────────
RATE_LIMIT = "30/minute"

# ─── Download Limits ──────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB   = 10
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
