import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import timm
import numpy as np
import cv2
import base64
from PIL import Image
from utils.forensics import MetadataAnalyzer

# Explicit modular components (Phase 9 Restructuring)
from models.content_type_classifier import ContentTypeClassifier
from models.diffusion_detector import DiffusionDetector
from models.embedding_analyzer import CLIPEmbeddingAnalyzer

class TemperatureScaling(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
    
    def forward(self, logits):
        return logits / self.temperature

class FrequencyCNN(nn.Module):
    """
    Advanced Frequency Module taking Radial Power Spectrum (1D) features.
    Designed to catch over-optimized color harmonics and smooth gradients.
    """
    def __init__(self, input_dim=112): # For 224x224 image, max radius is ~112
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
    """
    Analyzes the sensor pattern noise (PRNU) residual to verify real cameras.
    Outputs high probability if PRNU (Real Camera) pattern is authentic.
    """
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

        # Grad-CAM hooks (only used when fast=False)
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
        import os
        if os.environ.get("LOW_MEMORY") == "1":
            print("[OpenSeek] ℹ️ Running in LOW_MEMORY mode: skipped HuggingFace Deepfake ViT model")
        else:
            try:
                from transformers import pipeline
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
        """Converts image to grayscale, applies FFT, extracts Radial Power Spectrum."""
        gray = cv2.cvtColor(original_img_cv, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray, (224, 224))
        f = np.fft.fft2(img_resized)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift) ** 2
        
        # Calculate radial profile
        h, w = magnitude_spectrum.shape
        center = (w//2, h//2)
        y, x = np.indices((h, w))
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)
        
        # We need a fixed length, e.g., 112
        radial_profile = radial_profile[:112]
        
        # Normalize
        radial_profile = np.log1p(radial_profile)
        radial_profile = (radial_profile - np.min(radial_profile)) / (np.max(radial_profile) - np.min(radial_profile) + 1e-9)
        
        tensor = torch.from_numpy(radial_profile).float().unsqueeze(0)
        return tensor.to(self.device)
        
    def _get_noise_residual(self, original_img_cv):
        """Extracts PRNU sensor pattern noise using a denoising filter."""
        img_resized = cv2.resize(original_img_cv, (224, 224))
        # Apply Gaussian Blur (Denoising)
        blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
        # Subtract from original to isolate noise residual
        residual = cv2.absdiff(img_resized, blurred)
        
        # Convert to tensor
        pil_residual = Image.fromarray(cv2.cvtColor(residual, cv2.COLOR_BGR2RGB))
        tensor = T.ToTensor()(pil_residual).unsqueeze(0)
        # Normalize
        tensor = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(tensor)
        return tensor.to(self.device)
        
    def _run_patch_analysis(self, original_img_cv):
        """Divides image into 8x8 patches and detects local inconsistencies using the spatial model."""
        h, w = original_img_cv.shape[:2]
        patch_h, patch_w = h // 8, w // 8
        
        if patch_h < 10 or patch_w < 10:
            return 0.0, 0 # Image too small for meaningful patches
            
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
                # Batch inference on all 64 patches
                logits = self.spatial_model(patches_tensor)
                probs = torch.sigmoid(self.calibrator(logits)).squeeze().cpu().numpy()
            
        patch_variance = np.var(probs)
        manipulated_count = int(np.sum(probs > 0.6))
        
        # Calculate a normalized anomaly score based on variance and high-scoring patches
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
        
        # 1. Spatial Model (skip Grad-CAM in fast mode for speed)
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
            heatmap_base64 = self.calculate_gradcam(input_tensor, original_img_cv)
            input_tensor.requires_grad = False

        # Run Hugging Face expert model inference (if loaded)
        hf_probability = None
        if self.hf_model:
            try:
                with torch.no_grad():
                    out = self.hf_model(img)
                # Since the model config labels are inverted:
                # 'Realism' represents actual Deepfake
                # 'Deepfake' represents actual Realism
                fake_res = next((r for r in out if any(l in r['label'].lower() for l in ["realism", "real"])), None)
                if fake_res:
                    hf_probability = float(fake_res['score'])
            except Exception as e:
                print(f"[OpenSeek] HuggingFace expert model inference error: {e}")

        with torch.no_grad():
            with autocast(enabled=is_cuda):
                freq_tensor = self._get_fft_magnitude(original_img_cv)
                
                # Extract cluster embedding anomaly score
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
                    # Sensor verify reduces AI score if authentic PRNU found
                    ai_probability = max(0.0, base_score - (0.40 * authentic_prnu_prob))
                    
                    model_probs = [hf_probability if hf_probability is not None else spatial_prob, freq_prob, (1.0 - authentic_prnu_prob), embedding_dist_score]
                    variance = np.var(model_probs)
                    confidence_score = max(0.0, 1.0 - (variance * 4))
                    
                    # Predict source from photograph pipeline
                    if ai_probability > 0.5:
                        predicted_class = "Deepfake_AI"
                    else:
                        predicted_class = "Real"
                        
                else:
                    # ── ILLUSTRATION / DIFFUSION PIPELINE ──
                    diff_data = self.diffusion_detector.predict(input_tensor)
                    diffusion_score = diff_data["ai_probability"]
                    predicted_class = diff_data["predicted_class"]
                    
                    freq_logit = self.freq_model(freq_tensor) # Basic FFT artifacts
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
            
        # --- Risk Categorization ---
        if ai_probability <= 0.40:
            risk_level = "Low"
        elif ai_probability <= 0.65:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        # Heavy model disagreement triggers uncertainty flag
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
