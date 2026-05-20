"""
OpenSeek — Input validation helpers.
"""
import re
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator, HttpUrl


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
