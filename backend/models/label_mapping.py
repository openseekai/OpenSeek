"""Shared label→P(AI) mapping. No heavy deps so both the lean and full engines
can import it cheaply.

Maps an image-classification output (list of {"label", "score"}) to P(AI/fake):
picks the fake/deepfake label score; falls back to 1 - P(real) if only a real
label is present. Keeps every model on the same convention (high == more AI),
instead of guessing per-model label ordering (which caused an inversion bug).
"""

_FAKE_KEYS = ("fake", "deepfake", "synthetic", "gan", "manipulated", "generated")
_REAL_KEYS = ("real", "authentic", "genuine", "pristine", "natural")


def ai_prob_from_classifier(outputs):
    if not outputs:
        return None
    fake_score = real_score = None
    for r in outputs:
        label = str(r.get("label", "")).lower()
        score = float(r.get("score", 0.0))
        if any(k in label for k in _FAKE_KEYS):
            fake_score = score if fake_score is None else max(fake_score, score)
        elif any(k in label for k in _REAL_KEYS):
            real_score = score if real_score is None else max(real_score, score)
    if fake_score is not None:
        return fake_score
    if real_score is not None:
        return 1.0 - real_score
    return None
