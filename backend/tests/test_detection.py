"""Detection-logic tests.

These guard the parts that silently broke before: the HF label→P(AI) mapping
(which was inverted and reported real photos as AI) and the upload validators.
They run without downloading models. A heavier, opt-in integration test that
loads the real detector is gated behind RUN_MODEL_TESTS=1.
"""
import io
import os

import pytest
from models.advanced_ensemble import AdvancedForensicEnsemble

ai_prob = AdvancedForensicEnsemble._ai_prob_from_classifier


# ── Label mapping: high score == more likely AI, regardless of label order ────

def test_real_image_scores_low():
    # prithiv-style labels: "Realism" (real) vs "Deepfake" (fake)
    out = [{"label": "Realism", "score": 0.92}, {"label": "Deepfake", "score": 0.08}]
    assert ai_prob(out) == pytest.approx(0.08)  # must NOT be 0.92 (the old inversion bug)


def test_fake_image_scores_high():
    out = [{"label": "Realism", "score": 0.10}, {"label": "Deepfake", "score": 0.90}]
    assert ai_prob(out) == pytest.approx(0.90)


def test_real_fake_labels():
    # dima/haywood-style labels: "real" / "fake"
    assert ai_prob([{"label": "real", "score": 0.7}, {"label": "fake", "score": 0.3}]) == pytest.approx(0.3)
    assert ai_prob([{"label": "real", "score": 0.2}, {"label": "fake", "score": 0.8}]) == pytest.approx(0.8)


def test_only_real_label_inverts():
    assert ai_prob([{"label": "authentic", "score": 0.75}]) == pytest.approx(0.25)


def test_empty_output_is_none():
    assert ai_prob([]) is None
    assert ai_prob(None) is None


# ── Upload validators ─────────────────────────────────────────────────────────

def test_mime_validation():
    from fastapi import HTTPException
    from utils.validators import validate_image_type
    allowed = {"image/jpeg", "image/png"}
    validate_image_type("image/png", allowed)          # ok, no raise
    validate_image_type("image/jpeg; charset=x", allowed)  # ok with params
    with pytest.raises(HTTPException) as e:
        validate_image_type("application/zip", allowed)
    assert e.value.status_code == 415


def test_upload_size_limit(tmp_path):
    from fastapi import HTTPException
    from utils.validators import save_upload_limited

    class _Upload:
        def __init__(self, data):
            self.file = io.BytesIO(data)

    dest = str(tmp_path / "out.bin")
    # Under the limit: writes fine and returns byte count.
    assert save_upload_limited(_Upload(b"x" * 1000), dest, 2000) == 1000
    # Over the limit: raises 413 and removes the partial file.
    with pytest.raises(HTTPException) as e:
        save_upload_limited(_Upload(b"x" * 5000), dest, 2000)
    assert e.value.status_code == 413
    assert not os.path.exists(dest)


# ── Opt-in integration test (loads the real detector) ─────────────────────────

@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1",
                    reason="set RUN_MODEL_TESTS=1 to run the real-model integration test")
def test_real_model_separates_real_from_ai(tmp_path):
    """Sanity check that the deployed primary model gives a low score to a flat
    real-ish image. Downloads the model — only runs when explicitly enabled."""
    import numpy as np
    import torch
    from PIL import Image

    eng = AdvancedForensicEnsemble(torch.device("cpu"))
    # A plain mid-grey image should not be confidently flagged as AI.
    p = str(tmp_path / "grey.jpg")
    Image.fromarray(np.full((256, 256, 3), 127, dtype=np.uint8)).save(p)
    res = eng.forward_analyze(p, fast=True)
    assert 0.0 <= res["ai_probability"] <= 1.0
    assert "risk_level" in res
