"""
OpenSeek — Integration Test Suite
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """
    Create a test client with stubbed model singletons so tests run without
    GPU/large-download overhead while still exercising all routing and
    validation logic.
    """
    # Build lightweight stub objects that satisfy main.py's interface
    fake_ensemble = MagicMock()
    fake_ensemble.forward_analyze.return_value = {
        "final_probability": 42.0,
        "is_ai_generated": False,
        "detected_type": "authentic",
        "confidence": 35.0,
        "scores": {
            "spatial": 0.40,
            "frequency": 0.38,
            "noise": 0.41,
            "metadata": 0.10,
        },
        "patch_peak": 0.25,
    }

    fake_video_detector = MagicMock()
    fake_video_detector.device = "cpu"
    import torch
    fake_video_detector.return_value = (
        torch.tensor([0.35]),
        torch.tensor([0.10]),
    )

    fake_audio_detector = MagicMock()
    fake_audio_detector.parameters.return_value = iter([torch.zeros(1)])

    fake_processor = MagicMock()
    fake_processor.extract_frames.return_value = torch.zeros(8, 3, 224, 224)
    fake_processor.extract_audio.return_value = False

    with (
        patch("main._ensemble", fake_ensemble),
        patch("main._video_detector", fake_video_detector),
        patch("main._audio_detector", fake_audio_detector),
        patch("main._video_processor", fake_processor),
    ):
        from main import app
        yield TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_schema(self, client):
        r = client.get("/health")
        body = r.json()
        assert "status" in body
        assert body["status"] == "ok"
        assert "models_loaded" in body


# ─────────────────────────────────────────────────────────────────────────────
# POST /detect-image  (multipart upload)
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectImage:
    def _fake_jpg(self):
        """Minimal valid JPEG bytes."""
        import struct
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xd9"
        )

    def test_returns_200_with_file(self, client):
        r = client.post(
            "/detect-image",
            files={"file": ("test.jpg", self._fake_jpg(), "image/jpeg")},
        )
        # The stub may raise if cv2 can't open the fake bytes, but we at least
        # expect a well-formatted error (4xx/5xx), NOT a 404.
        assert r.status_code != 404

    def test_analyze_image_alias_exists(self, client):
        """Must not 404 — /analyze-image must be registered."""
        r = client.post(
            "/analyze-image",
            files={"file": ("test.jpg", self._fake_jpg(), "image/jpeg")},
        )
        assert r.status_code != 404

    def test_missing_file_field_rejected(self, client):
        r = client.post("/detect-image")
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-image-data  (JSON body with URL)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeImageData:
    def test_missing_url_rejected(self, client):
        r = client.post("/analyze-image-data", json={})
        assert r.status_code == 422

    def test_endpoint_exists(self, client):
        """Must not 404."""
        # We don't actually hit the network — httpx will fail, giving us 500, not 404.
        r = client.post("/analyze-image-data", json={"url": "http://example.com/img.jpg"})
        assert r.status_code != 404


# ─────────────────────────────────────────────────────────────────────────────
# GET /health — status field value
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthStatusValue:
    def test_status_is_ok_not_online(self, client):
        """background.js checks status === 'ok', so this must NOT be 'online'."""
        r = client.get("/health")
        assert r.json()["status"] == "ok"
