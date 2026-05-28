import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import io
import uuid
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from transformers import CLIPProcessor, CLIPModel, pipeline
import mediapipe as mp

def _sanitize_numpy(val):
    """Recursively convert NumPy data types to standard Python types."""
    if isinstance(val, dict):
        return {k: _sanitize_numpy(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_sanitize_numpy(v) for v in val]
    elif isinstance(val, tuple):
        return tuple(_sanitize_numpy(v) for v in val)
    elif isinstance(val, np.ndarray):
        return _sanitize_numpy(val.tolist())
    elif isinstance(val, np.generic):
        return val.item()
    return val


# Initialize FastAPI App
app = FastAPI(title="OpenSeek Colab GPU Inference Server")

# ─── Face Detector Class ──────────────────────────────────────────────────────
_mp_face = mp.solutions.face_detection

class FaceDetector:
    def __init__(self, min_confidence: float = 0.5) -> None:
        self._detector = _mp_face.FaceDetection(
            model_selection=0,           # 0 = short-range (<=2 m), fast
            min_detection_confidence=min_confidence,
        )

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        faces: list[dict] = []

        if not results.detections:
            return faces

        h, w = image_bgr.shape[:2]
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = int(bb.xmin * w)
            y = int(bb.ymin * h)
            bw = int(bb.width  * w)
            bh = int(bb.height * h)
            faces.append({
                "bbox": (max(x, 0), max(y, 0), bw, bh),
                "confidence": round(det.score[0], 4),
            })

        return faces

    def close(self) -> None:
        self._detector.close()

# ─── Model Class Definitions ──────────────────────────────────────────────────

class ContentTypeClassifier(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.model = models.mobilenet_v3_small(weights="DEFAULT") 
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, 3)
        self.to(self.device)
        self.eval()

    def forward(self, x):
        logits = self.model(x)
        return torch.softmax(logits, dim=1)

    def classify(self, x):
        probs = self.forward(x)
        pred_class = torch.argmax(probs, dim=1).item()
        classes = ["Photograph", "Digital Illustration", "3D Render"]
        return classes[pred_class]


class DiffusionDetector(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.classes = ["Real", "Diffusion_AI"]
        self.model = models.efficientnet_b2(weights="IMAGENET1K_V1") 
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)
        self.to(self.device)
        self.eval()

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        pred_idx = np.argmax(probs)
        predicted_class = self.classes[pred_idx]
        
        return {
            "predicted_class": predicted_class,
            "probability_distribution": {
                "Real": float(probs[0]),
                "Diffusion_AI": float(probs[1])
            },
            "ai_probability": float(probs[1]) 
        }


class CLIPEmbeddingAnalyzer(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.has_clip = False
        
        try:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
            self.has_clip = True
            
            # Precalculated Clustering Centroids (consistent seed)
            np.random.seed(42)
            self.real_centroid = nn.Parameter(
                F.normalize(torch.randn(1, 512).to(self.device), p=2, dim=1), requires_grad=False
            )
            self.diffusion_centroid = nn.Parameter(
                F.normalize(torch.randn(1, 512).to(self.device) + 0.1, p=2, dim=1), requires_grad=False
            )
        except Exception as e:
            print(f"[OpenSeek CLIP] Warning: CLIP loading failed: {e}")

    def get_embedding(self, pil_image: Image.Image):
        if not self.has_clip:
            return torch.zeros(1, 512).to(self.device)
            
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            if hasattr(outputs, 'image_embeds'):
                features = outputs.image_embeds
            elif hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            else:
                features = outputs
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def analyze_anomaly(self, pil_image: Image.Image):
        if not self.has_clip:
            return 0.0
            
        features = self.get_embedding(pil_image)
        with torch.no_grad():
            similarity_to_real = F.cosine_similarity(features, self.real_centroid).item()
            similarity_to_ai = F.cosine_similarity(features, self.diffusion_centroid).item()
            
        diff = similarity_to_ai - similarity_to_real
        anomaly_score = torch.sigmoid(torch.tensor(diff * 10)).item() 
        return anomaly_score


class TemperatureScaling(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
    
    def forward(self, logits):
        return logits / self.temperature


class FrequencyCNN(nn.Module):
    def __init__(self, input_dim=112):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AdvancedForensicEnsemble(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # 0. Pre-Classifier (Photo vs Illustration) via MobileNet
        self.content_type_classifier = ContentTypeClassifier(device=self.device)

        # 1. Spatial Model (EfficientNet-B0)
        self.spatial_model = models.efficientnet_b0(pretrained=False)
        self.spatial_model.classifier[1] = nn.Linear(self.spatial_model.classifier[1].in_features, 1)
        self.spatial_model.to(self.device).eval()

        # Grad-CAM hooks
        self.gradients = None
        self.activations = None
        self.target_layer = self.spatial_model.features[-1]
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

        # 2a. Photograph Frequency Model
        self.freq_model = FrequencyCNN()
        self.freq_model.to(self.device).eval()
        
        # 2b. Diffusion Specific Detector (3-class Outputs)
        self.diffusion_detector = DiffusionDetector(device=self.device)
        
        # 3. Residual Noise Model (Photographs only)
        self.residual_model = ResidualCNN()
        self.residual_model.to(self.device).eval()

        # 4. CLIP Embedding Analyzer (Clustering)
        self.embedding_analyzer = CLIPEmbeddingAnalyzer(device=self.device)
        
        # 5. Hugging Face Expert Classifier (accurate pre-trained deepfake model)
        self.hf_model = None
        if os.environ.get("LOW_MEMORY") == "1":
            print("[OpenSeek] ℹ️ Running in LOW_MEMORY mode: skipped HuggingFace Deepfake ViT model")
        else:
            try:
                self.hf_model = pipeline(
                    "image-classification",
                    model="prithivMLmods/Deep-Fake-Detector-v2-Model",
                    top_k=None,
                    device=-1 if self.device == 'cpu' else 0
                )
                print("[OpenSeek] ✅ Loaded pre-trained HuggingFace Deepfake ViT model")
            except Exception as e:
                print(f"[OpenSeek] Warning: Failed to load HuggingFace Expert model: {e}")
        
        # Calibration
        self.calibrator = TemperatureScaling()
        self.calibrator.to(self.device)

        # Generator Attribution
        self.attribution_classifier = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        ).to(self.device).eval()
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _get_fft_magnitude(self, original_img_cv):
        gray = cv2.cvtColor(original_img_cv, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray, (224, 224))
        f = np.fft.fft2(img_resized)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift) ** 2
        
        h, w = magnitude_spectrum.shape
        center = (w//2, h//2)
        y, x = np.indices((h, w))
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)
        radial_profile = radial_profile[:112]
        
        radial_profile = np.log1p(radial_profile)
        radial_profile = (radial_profile - np.min(radial_profile)) / (np.max(radial_profile) - np.min(radial_profile) + 1e-9)
        
        tensor = torch.from_numpy(radial_profile).float().unsqueeze(0)
        return tensor.to(self.device)
        
    def _get_noise_residual(self, original_img_cv):
        img_resized = cv2.resize(original_img_cv, (224, 224))
        blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
        residual = cv2.absdiff(img_resized, blurred)
        
        pil_residual = Image.fromarray(cv2.cvtColor(residual, cv2.COLOR_BGR2RGB))
        tensor = T.ToTensor()(pil_residual).unsqueeze(0)
        tensor = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(tensor)
        return tensor.to(self.device)
        
    def _run_patch_analysis(self, original_img_cv):
        h, w = original_img_cv.shape[:2]
        patch_h, patch_w = h // 8, w // 8
        if patch_h < 10 or patch_w < 10:
            return 0.0, 0
            
        patches = []
        for i in range(8):
            for j in range(8):
                y1, y2 = i * patch_h, (i + 1) * patch_h
                x1, x2 = j * patch_w, (j + 1) * patch_w
                patch = original_img_cv[y1:y2, x1:x2]
                
                pil_patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                tensor = self.transform(pil_patch)
                patches.append(tensor)
                
        patches_tensor = torch.stack(patches).to(self.device)
        is_cuda = self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda")
        from torch.cuda.amp import autocast
        with torch.no_grad():
            with autocast(enabled=is_cuda):
                logits = self.spatial_model(patches_tensor)
                probs = torch.sigmoid(self.calibrator(logits)).squeeze().cpu().numpy()
            
        patch_variance = np.var(probs)
        manipulated_count = int(np.sum(probs > 0.6))
        anomaly_score = min(1.0, (patch_variance * 10) + (manipulated_count / 64.0))
        return anomaly_score, manipulated_count

    def calculate_gradcam(self, input_tensor, original_img):
        self.spatial_model.zero_grad()
        output = self.spatial_model(input_tensor)
        output.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-9)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_img = heatmap + np.float32(original_img) / 255
        cam_img = cam_img / np.max(cam_img)
        
        cam_img_uint8 = np.uint8(255 * cam_img)
        _, buffer = cv2.imencode('.jpg', cam_img_uint8)
        base64_heatmap = base64.b64encode(buffer).decode('utf-8')
        return base64_heatmap

    def forward_analyze(self, image_path: str, fast: bool = True):
        img = Image.open(image_path).convert("RGB")
        original_img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        is_cuda = self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda")
        from torch.cuda.amp import autocast
        
        # 0. Route Classification via MobileNetV3
        with torch.no_grad():
            with autocast(enabled=is_cuda):
                content_type = self.content_type_classifier.classify(input_tensor)
        is_illustration = (content_type in ["Digital Illustration", "3D Render"])
        
        # 1. Spatial Model
        if fast:
            with torch.no_grad():
                with autocast(enabled=is_cuda):
                    spatial_logit = self.spatial_model(input_tensor)
                    spatial_prob = torch.sigmoid(self.calibrator(spatial_logit)).item()
            heatmap_base64 = None
        else:
            input_tensor.requires_grad = True
            with autocast(enabled=is_cuda):
                spatial_logit = self.spatial_model(input_tensor)
                spatial_prob = torch.sigmoid(self.calibrator(spatial_logit)).item()
            # Calculate Grad-CAM in float32 for gradient stability
            heatmap_base64 = self.calculate_gradcam(input_tensor, original_img_cv)
            input_tensor.requires_grad = False

        # Run Hugging Face expert model inference
        hf_probability = None
        if self.hf_model:
            try:
                with torch.no_grad():
                    out = self.hf_model(img)
                fake_res = next((r for r in out if any(l in r['label'].lower() for l in ["fake", "deepfake", "synthetic"])), None)
                if fake_res:
                    hf_probability = float(fake_res['score'])
            except Exception as e:
                print(f"[OpenSeek HF] Inference error: {e}")

        with torch.no_grad():
            with autocast(enabled=is_cuda):
                freq_tensor = self._get_fft_magnitude(original_img_cv)
                embedding_dist_score = self.embedding_analyzer.analyze_anomaly(img)
                
                if not is_illustration:
                    # ── PHOTOGRAPH FORENSIC PIPELINE ──
                    freq_logit = self.freq_model(freq_tensor)
                    freq_prob = torch.sigmoid(self.calibrator(freq_logit)).item()
                    
                    res_tensor = self._get_noise_residual(original_img_cv)
                    res_logit = self.residual_model(res_tensor)
                    authentic_prnu_prob = torch.sigmoid(self.calibrator(res_logit)).item()
                    
                    if not fast:
                        patch_prob, manipulated_count = self._run_patch_analysis(original_img_cv)
                    else:
                        patch_prob, manipulated_count = 0.0, 0
    
                    # Rebalanced Formula: Photograph
                    if hf_probability is not None:
                        base_score = (0.75 * hf_probability) + (0.15 * freq_prob) + (0.10 * embedding_dist_score)
                    else:
                        base_score = (0.50 * spatial_prob) + (0.30 * freq_prob) + (0.20 * embedding_dist_score)
                    ai_probability = max(0.0, base_score - (0.40 * authentic_prnu_prob))
                    
                    model_probs = [hf_probability if hf_probability is not None else spatial_prob, freq_prob, (1.0 - authentic_prnu_prob), embedding_dist_score]
                    variance = np.var(model_probs)
                    confidence_score = max(0.0, 1.0 - (variance * 4))
                    
                    if ai_probability > 0.5:
                        predicted_class = "Deepfake_AI"
                    else:
                        predicted_class = "Real"
                else:
                    # ── ILLUSTRATION / DIFFUSION PIPELINE ──
                    diff_data = self.diffusion_detector.predict(input_tensor)
                    diffusion_score = diff_data["ai_probability"]
                    predicted_class = diff_data["predicted_class"]
                    
                    freq_logit = self.freq_model(freq_tensor)
                    freq_prob = torch.sigmoid(self.calibrator(freq_logit)).item()
                    
                    if not fast:
                        patch_prob, manipulated_count = self._run_patch_analysis(original_img_cv)
                    else:
                        patch_prob, manipulated_count = 0.0, 0
                    
                    # Rebalanced Formula: Illustration
                    if hf_probability is not None:
                        ai_probability = (0.75 * hf_probability) + (0.15 * embedding_dist_score) + (0.10 * freq_prob)
                        if ai_probability > 0.5:
                            predicted_class = "Diffusion_AI"
                        else:
                            predicted_class = "Real"
                    else:
                        ai_probability = (0.50 * diffusion_score) + (0.25 * embedding_dist_score) + (0.25 * freq_prob)
                    
                    model_probs = [hf_probability if hf_probability is not None else diffusion_score, freq_prob, embedding_dist_score]
                    variance = np.var(model_probs)
                    confidence_score = max(0.0, 1.0 - (variance * 3))
            
        # Risk Categorization
        if ai_probability <= 0.40:
            risk_level = "Low"
        elif ai_probability <= 0.65:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        if confidence_score < 0.4:
            risk_level = "Uncertain"
            
        return {
            "content_type": content_type,
            "ai_probability": round(ai_probability, 4),
            "predicted_class": predicted_class,
            "confidence_score": round(confidence_score, 4),
            "risk_level": risk_level,
            "manipulated_regions_heatmap": f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_base64 else None,
            "patch_manipulated_count": manipulated_count,
            "embedding_anomaly_score": round(embedding_dist_score, 4)
        }

# Global instances loaded at startup
ensemble = None
face_detector = None

@app.on_event("startup")
def load_models():
    global ensemble, face_detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Starting OpenSeek Server on device: {device}")
    
    ensemble = AdvancedForensicEnsemble(device=device)
    face_detector = FaceDetector()
    
    if torch.cuda.is_available():
        print("[*] Tesla T4 GPU Detected! Activating cuDNN auto-tuner benchmarks for maximum speed...")
        torch.backends.cudnn.benchmark = True
        
    print("[*] OpenSeek Engine and Face Detector Loaded Successfully")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Save the file temporarily
    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Colab has a GPU, so we run with fast=False to generate heatmaps & detailed patch scans!
        full_res = ensemble.forward_analyze(temp_path, fast=False)
        
        # Detect faces in BGR image
        img_cv = cv2.imread(temp_path)
        faces = face_detector.detect(img_cv)
        
        final_probability = full_res["ai_probability"]
        
        # If faces found, boost probability slightly (5% boost)
        if faces:
            final_probability = min(1.0, full_res["ai_probability"] * 1.05)
            
        # Re-calc risk level logic
        if final_probability <= 0.40:
            risk = "Low"
        elif final_probability <= 0.65:
            risk = "Medium"
        else:
            risk = "High"
            
        content_type = full_res.get("content_type", "Photograph")
        predicted_class = full_res.get("predicted_class", "Real")
        embedding_score = full_res.get("embedding_anomaly_score", 0.0)
            
        response_data = {
            "is_ai_generated": final_probability > 0.5,
            "ai_probability": round(final_probability, 4),
            "content_type": content_type,
            "predicted_class": predicted_class,
            "confidence_score": full_res["confidence_score"],
            "risk_level": risk,
            "manipulated_regions_heatmap": full_res["manipulated_regions_heatmap"],
            "patch_manipulated_count": full_res["patch_manipulated_count"],
            "embedding_anomaly_score": embedding_score,
            "face_detected": len(faces) > 0
        }
        
        if full_res["confidence_score"] < 0.4:
            response_data["risk_level"] = "Uncertain"
            response_data["flag"] = "Low Confidence Detection"
            
        return _sanitize_numpy(response_data)

    except Exception as e:
        print(f"[Error] Failed to analyze file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "ensemble_loaded": ensemble is not None,
        "face_detector_loaded": face_detector is not None
    }

if __name__ == "__main__":
    try:
        # Detect if running in an interactive notebook (Jupyter / Colab)
        shell = get_ipython().__class__.__name__
        import nest_asyncio
        import threading
        import subprocess
        import time
        import re
        nest_asyncio.apply()
        threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000), daemon=True).start()
        print("[*] API Server started in background on port 8000 (Notebook mode).")
        
        # Start Cloudflare Tunnel
        print("[*] Starting Cloudflare Tunnel, please wait...")
        subprocess.Popen("cloudflared tunnel --url http://127.0.0.1:8000 > cloudflare.log 2>&1", shell=True)
        time.sleep(8)
        
        with open("cloudflare.log", "r") as f:
            logs = f.read()
            
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", logs)
        if match:
            print("\n🎉 SUCCESS! Copy this URL and set it in Railway:")
            print(match.group(0))
        else:
            print("\n❌ Tunnel URL not found yet. Please view cloudflare.log or run again.")
    except NameError:
        # Standard CLI execution
        uvicorn.run(app, host="127.0.0.1", port=8000)
