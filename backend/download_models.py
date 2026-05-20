import os
import torch
import torchvision.models as models
from transformers import CLIPProcessor, CLIPModel

print("[OpenSeek Build] Pre-downloading model weights to cache...")

# Create directories
os.makedirs(os.environ.get("TORCH_HOME", "/app/cache/torch"), exist_ok=True)
os.makedirs(os.environ.get("HF_HOME", "/app/cache/huggingface"), exist_ok=True)

# 1. ContentTypeClassifier: mobilenet_v3_small
print("[OpenSeek Build] Downloading MobileNetV3 Small...")
models.mobilenet_v3_small(weights="DEFAULT")

# 2. DiffusionDetector: efficientnet_b2
print("[OpenSeek Build] Downloading EfficientNet B2...")
models.efficientnet_b2(weights="IMAGENET1K_V1")

# 3. CLIPEmbeddingAnalyzer: openai/clip-vit-base-patch32
print("[OpenSeek Build] Downloading CLIP (openai/clip-vit-base-patch32)...")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 4. AdvancedForensicEnsemble: efficientnet_b0
print("[OpenSeek Build] Downloading EfficientNet B0...")
try:
    models.efficientnet_b0(weights=None)
except Exception:
    models.efficientnet_b0(pretrained=False)

print("[OpenSeek Build] All model weights pre-downloaded successfully!")

