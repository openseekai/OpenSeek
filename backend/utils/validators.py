"""
OpenSeek — Input validation helpers.
"""
import re
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator

_SAFE_SCHEME = re.compile(r"^https?$", re.I)


class MediaRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https scheme.")
        if not parsed.netloc:
            raise ValueError("URL is missing a network location (host).")
        # Reject obviously malformed URLs
        if " " in v or "\n" in v or "\r" in v:
            raise ValueError("URL contains invalid characters.")
        return v


def sanitize_url(url: str) -> str:
    """Return a cleaned URL string (strip whitespace)."""
    return url.strip()


def save_upload_limited(upload_file, dest_path: str, max_bytes: int) -> int:
    """Stream an UploadFile to `dest_path`, enforcing a hard size cap.

    Streams in 1 MiB chunks so a huge upload never sits fully in memory, and
    aborts (removing the partial file) once `max_bytes` is exceeded.

    Raises fastapi.HTTPException(413) if the file is too large.
    Returns the number of bytes written.
    """
    import os

    from fastapi import HTTPException

    written = 0
    chunk = 1024 * 1024
    with open(dest_path, "wb") as out:
        while True:
            data = upload_file.file.read(chunk)
            if not data:
                break
            written += len(data)
            if written > max_bytes:
                out.close()
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (limit {max_bytes // (1024 * 1024)} MB).",
                )
            out.write(data)
    return written


def validate_image_type(content_type: str | None, allowed: set) -> None:
    """Raise HTTPException(415) if the declared content type isn't an allowed image type."""
    from fastapi import HTTPException

    if content_type and content_type.split(";")[0].strip().lower() not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Allowed: {sorted(allowed)}",
        )
