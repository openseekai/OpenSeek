"""Pre-download the model weights the backend ACTUALLY loads, so a deployed
image has them cached (no cold-start download).

Engine-aware (kept in sync with main.py / the engine modules):
  OPENSEEK_ENGINE=lean (default) → only the primary detector (small, CPU, Railway)
  OPENSEEK_ENGINE=full           → the full ensemble's models too (CLIP, etc.)
"""
import os

from transformers import pipeline

print("[OpenSeek Build] Pre-downloading model weights to cache...")

os.makedirs(os.environ.get("TORCH_HOME", "/app/cache/torch"), exist_ok=True)
os.makedirs(os.environ.get("HF_HOME", "/app/cache/huggingface"), exist_ok=True)

engine_mode = os.environ.get("OPENSEEK_ENGINE", "lean").lower()
primary = os.environ.get("OPENSEEK_DETECTOR_MODEL", "haywoodsloan/ai-image-detector-deploy")

# Primary detector — needed in BOTH modes.
print(f"[OpenSeek Build] Downloading primary AI detector: {primary} ...")
try:
    pipeline("image-classification", model=primary)
except Exception as e:
    print(f"[OpenSeek Build] Warning: failed to pre-download {primary}: {e}")

if engine_mode == "full":
    # Heavy ensemble extras: facial detector + torchvision backbones + CLIP.
    import torchvision.models as models
    from transformers import CLIPModel, CLIPProcessor

    print("[OpenSeek Build] (full) Downloading secondary facial detector...")
    try:
        pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
    except Exception as e:
        print(f"[OpenSeek Build] Warning: failed facial detector: {e}")

    print("[OpenSeek Build] (full) Downloading MobileNetV3 Small + EfficientNetV2-S...")
    models.mobilenet_v3_small(weights="DEFAULT")
    models.efficientnet_v2_s(weights="IMAGENET1K_V1")

    print("[OpenSeek Build] (full) Downloading CLIP (openai/clip-vit-base-patch32)...")
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(f"[OpenSeek Build] Model weights pre-downloaded (engine={engine_mode}).")
