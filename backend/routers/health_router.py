"""
OpenSeek — GET /health
"""
from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Health check")
async def health(request: Request) -> dict:
    """Returns server and model status."""
    models_loaded: bool = getattr(request.app.state, "models_ready", False)
    return {
        "status": "ok",
        "models_loaded": models_loaded,
    }
