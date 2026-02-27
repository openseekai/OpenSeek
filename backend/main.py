import os
import io
import shutil
import uuid
import hashlib
import json
import sqlite3
import cv2
import numpy as np
from typing import Optional
import torch
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.advanced_ensemble import AdvancedForensicEnsemble
from models.audio_detection import AudioCNNLSTM, extract_advanced_audio_features
from utils.video_utils import VideoProcessor
from utils.face_detector import get_face_detector
from models.temporal_video_detector import TemporalVideoDetector
from utils.face_temporal import FaceTemporalAnalyzer
from models.forensics.noise_temporal import NoiseConsistencyAnalyzer

# ── Database Cache Initialization ───────────────────────────────────────────
DB_PATH = "deepshield_cache.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scan_cache 
                 (hash TEXT PRIMARY KEY, response TEXT)''')
    conn.commit()
    conn.close()

def get_cached_result(file_hash):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT response FROM scan_cache WHERE hash=?', (file_hash,))
    row = c.fetchone()
    conn.close()
    if row: return json.loads(row[0])
    return None

def set_cached_result(file_hash, response):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO scan_cache (hash, response) VALUES (?, ?)', 
              (file_hash, json.dumps(response)))
    conn.commit()
    conn.close()

def compute_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

class MediaUrlRequest(BaseModel):
    url: str

app = FastAPI(title="DeepShield Ultimate Forensic Service")

_ensemble = None
_audio_detector = None
_video_processor = None
_face_detector = None
_temporal_video = None
_face_temporal = None
_noise_analyzer = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global _ensemble, _audio_detector, _video_processor, _face_detector
    init_db()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DeepShield API] Loading Advanced Ensemble Pipeline on {device}…")
    
    _ensemble = AdvancedForensicEnsemble(device)
    _audio_detector = AudioCNNLSTM().to(device)
    
    # New Video Upgrades
    _video_processor = VideoProcessor(frame_count=12) # 12 uniformly spaced frames
    _temporal_video = TemporalVideoDetector().to(device)
    _face_temporal = FaceTemporalAnalyzer()
    _noise_analyzer = NoiseConsistencyAnalyzer()
    
    _face_detector = get_face_detector()
    
    # FP16 Optimization
    if torch.cuda.is_available():
        print("[DeepShield API] Optimizing Models for FP16 Inference...")
        _ensemble.half()
        _audio_detector.half()
        _temporal_video.half()
    
    print("[DeepShield API] 🟢 Research-Grade Multi-Modal Engine Ready")

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    """Advanced Image Deepfake Detection (Spatial + ViT + Freq + Face)"""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_hash = compute_hash(temp_path)
    cached = get_cached_result(file_hash)
    if cached:
        os.remove(temp_path)
        return cached

    try:
        # 1. Full Image Analysis
        full_res = _ensemble.forward_analyze(temp_path)
        
        # 2. Face-Focused Layer
        img_cv = cv2.imread(temp_path)
        faces = _face_detector.detect(img_cv)
        
        final_probability = full_res["ai_probability"]
        
        if faces:
            # Get largest face bbox
            best_face = max(faces, key=lambda f: f["bbox"][2]*f["bbox"][3])
            x, y, w, h = best_face["bbox"]
            margin = int(0.1 * max(w, h))
            
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img_cv.shape[1], x + w + margin)
            y2 = min(img_cv.shape[0], y + h + margin)
            
            face_crop = img_cv[y1:y2, x1:x2]
            face_path = os.path.join(temp_dir, f"face_{uuid.uuid4()}.jpg")
            cv2.imwrite(face_path, face_crop)
            
            face_res = _ensemble.forward_analyze(face_path)
            os.remove(face_path)
            
            # Combine core score with face score (70/30 weighting)
            final_probability = (0.7 * full_res["ai_probability"]) + (0.3 * face_res["ai_probability"])
        
        # Re-calc risk level logic
        if final_probability <= 0.40: risk = "Low"
        elif final_probability <= 0.65: risk = "Medium"
        else: risk = "High"
        
        # Pull new phase 9 architectural attributes
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

        set_cached_result(file_hash, response_data)
        return response_data

    except Exception as e:
        print(f"[DeepShield API] Forensic Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """Advanced Temporal Video Deepfake Detection (Multi-Modal Multi-Frame)"""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    video_id = str(uuid.uuid4())
    temp_video = os.path.join(temp_dir, f"{video_id}_{file.filename}")
    temp_audio = os.path.join(temp_dir, f"{video_id}_audio.wav")

    try:
        with open(temp_video, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_hash = compute_hash(temp_video)
        cached = get_cached_result(file_hash)
        if cached: return cached

        # 1. Extract 12 uniformly spaced frames
        frames_tensor = _video_processor.extract_frames(temp_video)
        
        frame_scores = []
        raw_bgr_frames = [] # for face/noise temporal
        
        for i in range(len(frames_tensor)):
            f_path = os.path.join(temp_dir, f"frame_{i}_{video_id}.jpg")
            
            frame_np = frames_tensor[i].permute(1,2,0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame_np = std * frame_np + mean
            frame_np = np.clip(frame_np, 0, 1) * 255.0
            
            bgr_frame = cv2.cvtColor(np.uint8(frame_np), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_path, bgr_frame)
            raw_bgr_frames.append(bgr_frame)
            
            try:
                res = _ensemble.forward_analyze(f_path)
                frame_scores.append(res["ai_probability"])
            finally:
                if os.path.exists(f_path): os.remove(f_path)
                
        if not frame_scores:
            raise ValueError("No frames could be extracted from video.")

        mean_frame_ai_score = np.mean(frame_scores)
        variance_score = np.var(frame_scores)
        
        # 2. Temporal Model (EfficientNet + LSTM)
        temporal_score = _temporal_video.predict_video(frames_tensor)
        
        # 3. Face Temporal Analysis
        face_temporal_score = _face_temporal.analyze_sequence(raw_bgr_frames)
        
        # 4. Noise Consistency Check
        noise_score = _noise_analyzer.analyze_sequence(raw_bgr_frames)
        
        # 5. Audio Deepfake Detection
        has_audio = _video_processor.extract_audio(temp_video, temp_audio)
        audio_score = 0.0
        if has_audio and os.path.exists(temp_audio):
            features = extract_advanced_audio_features(temp_audio)
            features = features.unsqueeze(0).to(_audio_detector.parameters().__next__().device)
            if torch.cuda.is_available():
                features = features.half()
            
            with torch.no_grad():
                out = _audio_detector(features).cpu().numpy()[0]
                audio_score = float(out[0])
                
        # 6. Final Video Score Calculation
        if has_audio:
            final_score = (
                0.35 * mean_frame_ai_score +
                0.25 * temporal_score +
                0.15 * noise_score +
                0.15 * face_temporal_score +
                0.10 * audio_score
            )
            subscores = [mean_frame_ai_score, temporal_score, noise_score, face_temporal_score, audio_score]
        else:
            # Rebalance weights if no audio
            final_score = (
                0.40 * mean_frame_ai_score +
                0.30 * temporal_score +
                0.15 * noise_score +
                0.15 * face_temporal_score
            )
            subscores = [mean_frame_ai_score, temporal_score, noise_score, face_temporal_score]
            
        # Model agreement (variance among subscores)
        model_agreement = np.var(subscores)
        
        # 7. Add Proper Risk Thresholds
        if model_agreement > 0.08:
            risk = "Uncertain"
        else:
            final_percentage = final_score * 100
            if final_percentage <= 40:
                risk = "Low"
            elif final_percentage <= 65:
                risk = "Medium"
            else:
                risk = "High"

        response_data = {
            "type": "video",
            "is_ai_generated": final_score > 0.5,
            "ai_probability": round(final_score, 4),
            "risk_level": risk,
            "subscores": {
                "mean_frame_score": round(float(mean_frame_ai_score), 4),
                "temporal_model_score": round(float(temporal_score), 4),
                "noise_consistency_score": round(float(noise_score), 4),
                "face_temporal_score": round(float(face_temporal_score), 4),
                "audio_ai_score": round(float(audio_score), 4) if has_audio else None
            },
            "metrics": {
                "variance": round(float(variance_score), 4),
                "model_agreement_variance": round(float(model_agreement), 4)
            }
        }
        
        set_cached_result(file_hash, response_data)
        return response_data

    except Exception as e:
        print(f"[DeepShield Video] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """Standalone Audio Deepfake Detection (CNN+BiLSTM)"""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_hash = compute_hash(temp_path)
        cached = get_cached_result(file_hash)
        if cached: return cached

        # Extract features (148 Mel+MFCC Dim, TimeSteps)
        features = extract_advanced_audio_features(temp_path)
        features = features.unsqueeze(0).to(_audio_detector.parameters().__next__().device)
        
        if torch.cuda.is_available():
            features = features.half()

        with torch.no_grad():
            out = _audio_detector(features).cpu().numpy()[0]
            ai_prob = out[0]
            anomaly_score = out[1]
            
        if ai_prob < 0.3: risk = "Low"
        elif ai_prob < 0.6: risk = "Medium"
        else: risk = "High"
        
        response_data = {
            "is_ai_generated": ai_prob > 0.5,
            "ai_probability_audio": round(float(ai_prob), 4),
            "spectral_anomaly_score": round(float(anomaly_score), 4),
            "risk_level": risk,
        }
        set_cached_result(file_hash, response_data)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.post("/analyze-image-data")
async def analyze_image_data(req: MediaUrlRequest):
    """Extension Context URL Fetcher (Routed through advanced pipeline)"""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    filename = req.url.split("/")[-1].split("?")[0] or "scan.jpg"
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(req.url, follow_redirects=True)
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail="Could not download image from URL")
            with open(temp_path, "wb") as f:
                f.write(r.content)

        file_hash = compute_hash(temp_path)
        cached = get_cached_result(file_hash)
        if cached: return cached

        # Full Image Analysis
        full_res = _ensemble.forward_analyze(temp_path)
        
        # Respect new risk level logic
        ai_probability = full_res["ai_probability"]
        if ai_probability <= 0.40: risk = "Low"
        elif ai_probability <= 0.65: risk = "Medium"
        else: risk = "High"
            
        response_data = {
            "is_ai_generated": ai_probability > 0.5,
            "ai_probability": ai_probability,
            "content_type": full_res.get("content_type", "Photograph"),
            "predicted_class": full_res.get("predicted_class", "Real"),
            "risk_level": risk,
            "confidence_score": full_res["confidence_score"],
            "manipulated_regions_heatmap": full_res["manipulated_regions_heatmap"],
            "patch_manipulated_count": full_res["patch_manipulated_count"],
            "embedding_anomaly_score": full_res.get("embedding_anomaly_score", 0.0)
        }
        
        if full_res["confidence_score"] < 0.4:
            response_data["risk_level"] = "Uncertain"
            
        set_cached_result(file_hash, response_data)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.post("/analyze-image")
async def analyze_image_alias(file: UploadFile = File(...)):
    """Alias for multipart extensions."""
    return await detect_image(file)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Advanced DeepShield Multimodal Target",
        "models_loaded": _ensemble is not None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
