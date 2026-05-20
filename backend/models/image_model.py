"""
OpenSeek — Image Deepfake Detection Model (HuggingFace ViT + Heuristic Ensemble)

Detection pipeline (in order of priority):
  1. HuggingFace pre-trained ViT  — Various models (Wvolf/ViT, dima806/deepfake_vs_real)
  2. Custom fine-tuned weights    — weights/image_model.pt (EfficientNet-B4)
  3. Heuristic ensemble           — secondary signals

Final score = weighted blend of (AI Model) + (Heuristics)
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image as PILImage

from config import IMAGE_SIZE

# ─── Model IDs ───────────────────────────────────────────────────────────────

_HF_MODELS = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "dima806/deepfake_vs_real_image_detection",
]
_CUSTOM_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "image_model.pt")

_classifiers = {}  # Dict of transformers pipelines {name: pipe}
_eff_model   = None  # fallback/custom model
_device     = None

# ─── Fallback EfficientNet-B4 model ──────────────────────────────────────────

class SpectralAttentionModule(nn.Module):
    """
    Research-Grade: Learns attention weights in the frequency domain.
    Makes the model sensitive to high-frequency GAN/Diffusion artifacts.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Spatial Features (x)
        # 2. Spectral Features (FFT)
        freq = torch.fft.fft2(x, norm='ortho')
        freq_abs = torch.abs(freq)
        
        # Concatenate spatial + spectral
        combined = torch.cat([x, freq_abs], dim=1)
        
        # Learn frequency-aware masks
        mask = self.sigmoid(self.conv1x1(combined))
        return x * mask

class DeepfakeImageDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        
        # Inject Spectral Attention before the classifier
        self.features = backbone.features
        self.spectral_attn = SpectralAttentionModule(1792) # EfficientNet-B4 features channel count
        self.avgpool = backbone.avgpool
        
        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.features(x)
        x = self.spectral_attn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if return_features:
            return x
            
        return self.classifier(x)

_eff_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Heuristic helpers ────────────────────────────────────────────────────────

def _dct_artifact_score(gray: np.ndarray) -> float:
    f32 = np.float32(gray)
    dct = cv2.dct(f32)
    h, w = dct.shape
    total = np.sum(dct ** 2) + 1e-9
    hf = np.sum(dct[h // 2:, w // 2:] ** 2)
    return float(np.clip((hf / total) * 3.5, 0.0, 1.0))

def _lbp_entropy_score(gray: np.ndarray) -> float:
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp |= (gray >= shifted).astype(np.uint8)
        lbp  = np.left_shift(lbp, 1) & 0xFF
    hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0,256), density=True)
    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    return float(np.clip(1.0 - entropy / 8.0 + 0.2, 0.0, 1.0))

def _sharpness_asymmetry(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    _, w = lap.shape
    lv, rv = float(np.var(lap[:, :w//2])), float(np.var(lap[:, w//2:]))
    return float(np.clip(abs(lv - rv) / (max(lv, rv) + 1e-9) * 2.5, 0.0, 1.0))

def _noise_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    noise = gray - cv2.GaussianBlur(gray, (5,5), 0)
    std = float(np.std(noise))
    if std < 1.5: return 0.85
    return float(np.clip(1.0 - (std - 1.5) / 13.5, 0.0, 1.0))

def _is_human_signature(img_bgr: np.ndarray) -> bool:
    """Robust Human Identity Check: Must find a face AND eyes to trigger the 'Realism' Veto."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Tight detection to avoid false positives on bear/art textures
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 1: # Human face with eyes confirmed
                return True
        return False
    except:
        return True # Fallback to safe mode (Trust realism) if check fails

def _heuristic_score(img_bgr: np.ndarray) -> float:
    img_bgr = cv2.resize(img_bgr, (512, 512))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(0.35*_dct_artifact_score(gray) + 0.25*_lbp_entropy_score(gray) + 0.40*_noise_variance(img_bgr))

def _calibrate(raw: float) -> float:
    """
    Refined Barrier Calibration:
    1. Real Zone (< 0.45): Safe suppression.
    2. Artificial Zone (> 0.45): Immediate jump to High Risk (60%+).
    """
    if raw < 0.45:
        # Scale 0.45 raw into 0-15% range for ultra-low real scores
        return (raw / 0.45) * 0.15
    else:
        # CROSS THE BARRIER: Jump to 60%
        # Scale remaining 0.55 raw into 60%-100% range
        x = (raw - 0.45) / 0.55
        return 0.60 + (x ** 0.5) * 0.40

# ─── Model loader ─────────────────────────────────────────────────────────────

def load_image_model() -> None:
    global _classifiers, _eff_model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _classifiers = {}

    # 1. Load HuggingFace pipelines for Expert Consensus
    from transformers import pipeline
    for hf_id in _HF_MODELS:
        try:
            name = hf_id.split('/')[-1]
            print(f"[OpenSeek] Loading {name} detector…")
            pipe = pipeline(
                "image-classification",
                model=hf_id,
                top_k=None,
                device=-1 if _device.type == 'cpu' else 0
            )
            _classifiers[name] = pipe
            print(f"[OpenSeek] ✅ {name} added to ensemble")
        except Exception as e:
            print(f"[OpenSeek] Failed to load {hf_id}: {e}")

    # 2. Load Internal Spectral model
    try:
        print("[OpenSeek] Initializing Spectral-Aware Backbone…")
        _eff_model = DeepfakeImageDetector().to(_device)
        _eff_model.eval()
        print("[OpenSeek] ✅ Master Spectral Algorithm Ready")
    except Exception as e:
        print(f"[OpenSeek] Spectral init failed: {e}")
        _eff_model = None
    if os.path.exists(_CUSTOM_WEIGHTS):
        try:
            _eff_model.load_state_dict(torch.load(_CUSTOM_WEIGHTS, map_location=_device))
            print(f"[OpenSeek] ✅ Custom weights loaded from {_CUSTOM_WEIGHTS}")
        except:
            print("[OpenSeek] ⚠️ Custom weights failed, using ImageNet fallback")
    else:
        print("[OpenSeek] ⚠️ No deepfake model loaded. Run train/train_image.py for accuracy.")
    _eff_model.eval()

# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_image(image_path: str) -> dict:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: raise ValueError("Could not decode image")
    
    heuristic = _heuristic_score(img_bgr)
    pil_img = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
    # MASTER IMAGE ALGORITHM
    ai_score = 0.5
    
    # ── Branch A: Expert Ensemble (HuggingFace) ──
    m_results = {}
    if _classifiers:
        for name, pipe in _classifiers.items():
            try:
                out = pipe(pil_img) # Changed img_pil to pil_img
                fake_res = next((r for r in out if any(l in r['label'].lower() for l in ["fake", "deepfake", "synthetic"])), None)
                if fake_res: m_results[name] = float(fake_res['score'])
            except Exception as e:
                print(f"[OpenSeek] Expert sub-model error: {e}")
        
    # ── Branch B: Internal Spectral Model (FFT-Aware) ──
    spectral_score = 0.5
    if _eff_model:
        try:
            inp = _eff_transform(pil_img).unsqueeze(0).to(_device) # Changed img_pil to pil_img
            with torch.no_grad():
                spectral_score = float(_eff_model(inp).cpu().item())
        except Exception as e:
            print(f"[OpenSeek] Spectral inference error: {e}")

    # ── Branch C: Master Consensus Logic ──
    if m_results:
        s_lead = m_results.get("Deep-Fake-Detector-v2-Model", 0.5)
        s_check = m_results.get("deepfake_vs_real_image_detection", 0.5)
        
        # 1. BIOMETRIC GUARD: Strong Realism Veto for humans
        if _is_human_signature(img_bgr) and (s_check < 0.15 or s_lead < 0.15):
            # If a human face is found and any expert sees realism, trust it heavily
            ai_score = min(s_lead, s_check, spectral_score)
        # 2. SPECTRAL DOMINANCE: Only trust internal model if it's confident
        elif spectral_score > 0.7:
            ai_score = max(spectral_score, s_lead)
        # 3. EXPERT DOMINANCE: Trust the 2024 model for art/fakes
        elif s_lead > 0.15:
            # Shift subtle signals above the 0.45 barrier
            ai_score = 0.46 + s_lead * 0.54
        # 4. DEFAULT: Weighted average
        else:
            ai_score = 0.7 * s_lead + 0.3 * s_check
    else:
        # No experts, rely on spectral if it's confident
        ai_score = spectral_score if spectral_score > 0.6 or spectral_score < 0.4 else 0.4

    # Blend and calibrate
    # 'Perfect Accuracy' Mode: 90% AI Algorithm + 10% Forensic Heuristics
    raw = 0.90 * ai_score + 0.10 * heuristic
    final_score = _calibrate(raw)
    
    gray = cv2.cvtColor(cv2.resize(img_bgr, (512, 512)), cv2.COLOR_BGR2GRAY)
    is_human = _is_human_signature(img_bgr)
    
    return {
        "authenticity_score": round(final_score * 100, 2),
        "is_ai_generated": final_score > 0.55, # High risk threshold
        "spectral_score": round(spectral_score, 4),
        "expert_score": round(max(m_results.values()) if m_results else spectral_score, 4),
        "heuristic_score": round(heuristic, 4),
        "face_detected": is_human,
        "facial_inconsistency": round(_sharpness_asymmetry(img_bgr), 4),
        "lighting_mismatch": round(np.std([ch.mean() for ch in cv2.split(img_bgr)]) / (img_bgr.mean() + 1e-9) * 5.0, 4),
        "gan_artifacts": round(_dct_artifact_score(gray), 4),
    }
