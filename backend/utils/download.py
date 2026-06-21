"""
OpenSeek — Safe async media downloader.
Enforces size limits and Content-Type validation before saving.
"""
import os
import socket
import tempfile
from urllib.parse import urlparse

import httpx
from config import (
    ALLOWED_AUDIO_TYPES,
    ALLOWED_IMAGE_TYPES,
    ALLOWED_VIDEO_TYPES,
    BLOCKED_IP_PREFIXES,
    DOWNLOAD_TIMEOUT_S,
    MAX_AUDIO_SIZE_MB,
    MAX_IMAGE_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
)

_SIZE_MAP = {
    "image": MAX_IMAGE_SIZE_MB * 1024 * 1024,
    "video": MAX_VIDEO_SIZE_MB * 1024 * 1024,
    "audio": MAX_AUDIO_SIZE_MB * 1024 * 1024,
}

_MIME_MAP = {
    "image": ALLOWED_IMAGE_TYPES,
    "video": ALLOWED_VIDEO_TYPES,
    "audio": ALLOWED_AUDIO_TYPES,
}

# Extension-to-suffix map (for tempfile)
_EXT_SUFFIX = {
    "image": ".jpg",
    "video": ".mp4",
    "audio": ".mp3",
}


def _resolve_host(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    try:
        return socket.gethostbyname(host)
    except Exception:
        return host


def _is_blocked_ip(ip: str) -> bool:
    for prefix in BLOCKED_IP_PREFIXES:
        if ip.startswith(prefix):
            return True
    return False


async def download_media(url: str, media_type: str) -> str:
    """
    Download media from *url*, validate Content-Type and size.

    Returns
    -------
    str
        Absolute path to the saved temporary file.

    Raises
    ------
    ValueError
        On SSRF-prone URLs, wrong MIME type, or oversize content.
    httpx.HTTPError
        On network-level errors.
    """
    # 1. SSRF guard
    resolved_ip = _resolve_host(url)
    if _is_blocked_ip(resolved_ip):
        raise ValueError(f"Blocked URL (resolves to private/reserved IP: {resolved_ip})")

    max_bytes   = _SIZE_MAP[media_type]
    allowed     = _MIME_MAP[media_type]
    suffix      = _EXT_SUFFIX[media_type]

    timeout = httpx.Timeout(DOWNLOAD_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()

            # 2. MIME check (Content-Type header)
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in allowed:
                raise ValueError(
                    f"Unsupported Content-Type '{content_type}' for {media_type} analysis. "
                    f"Allowed: {allowed}"
                )

            # 3. Content-Length pre-check (informational; not authoritative)
            cl = resp.headers.get("content-length")
            if cl and int(cl) > max_bytes:
                raise ValueError(
                    f"Content-Length {int(cl) // (1024*1024)} MB exceeds limit "
                    f"{max_bytes // (1024*1024)} MB"
                )

            # 4. Stream download with hard byte cap
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            downloaded = 0
            try:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        tmp.close()
                        os.unlink(tmp.name)
                        raise ValueError(
                            f"Download exceeded maximum size {max_bytes // (1024*1024)} MB"
                        )
                    tmp.write(chunk)
                tmp.flush()
            finally:
                tmp.close()

    return tmp.name
