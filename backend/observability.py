"""Optional error tracking (Sentry).

Enabled only when SENTRY_DSN is set AND sentry-sdk is installed — otherwise a
no-op, so dev/CI never depend on it. Pair with the global exception handler in
main.py, which logs every unhandled error (and Sentry auto-captures it).
"""
import logging
import os

logger = logging.getLogger("openseek")


def init_sentry() -> bool:
    """Initialise Sentry if configured. Returns True if enabled."""
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("ENVIRONMENT", "production"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
            send_default_pii=False,
        )
        logger.info("[OpenSeek] Sentry error tracking enabled")
        return True
    except Exception as e:  # sentry-sdk missing or bad DSN — never block startup
        logger.warning(f"[OpenSeek] Sentry not enabled: {e}")
        return False
