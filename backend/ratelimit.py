"""
Rate limiting for OpenSeek.

Uses slowapi when available. If slowapi is not installed (or we're running the
test suite), `rate_limit()` degrades to a no-op decorator so the app still
imports and serves — production gets real limits, dev/CI never breaks.

Usage in main.py:

    from ratelimit import limiter, rate_limit, install_rate_limiting
    install_rate_limiting(app)        # registers state + error handler

    @app.post("/detect-image")
    @rate_limit()                     # uses config.RATE_LIMIT
    async def detect_image(request: Request, ...):
        ...
"""
import os
import sys

from config import RATE_LIMIT

_TESTING = "pytest" in sys.modules or os.getenv("TESTING") == "1"

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    def _client_ip(request) -> str:
        """Prefer the real client IP behind a proxy/load balancer (Cloud Run)."""
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return get_remote_address(request)

    limiter = Limiter(key_func=_client_ip, enabled=not _TESTING)
    HAS_SLOWAPI = True
except Exception:  # slowapi missing — degrade gracefully
    limiter = None
    RateLimitExceeded = None
    _rate_limit_exceeded_handler = None
    HAS_SLOWAPI = False


def rate_limit(limit_value: str = RATE_LIMIT):
    """Decorator that applies `limit_value` when slowapi is present, else no-op."""
    if HAS_SLOWAPI and limiter is not None:
        return limiter.limit(limit_value)

    def _noop(func):
        return func

    return _noop


def install_rate_limiting(app) -> None:
    """Wire the limiter into the FastAPI app (safe no-op without slowapi)."""
    if HAS_SLOWAPI and limiter is not None:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
