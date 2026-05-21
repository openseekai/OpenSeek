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
from utils.face_detector import get_face_detector

# ── Database Cache Initialization ───────────────────────────────────────────
DB_PATH = "openseek_cache.db"

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

app = FastAPI(title="OpenSeek Ultimate Forensic Service")

_ensemble = None
_face_detector = None

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
    
    # Optimize PyTorch memory usage on CPU (reduces thread-related RAM overhead)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OpenSeek API] Loading Advanced Ensemble Pipeline on {device}…")
    
    _ensemble = AdvancedForensicEnsemble(device)
    _face_detector = get_face_detector()
    
    # FP16 Optimization
    if torch.cuda.is_available():
        print("[OpenSeek API] Optimizing Models for FP16 Inference...")
        _ensemble.half()
        
    # Free up memory allocated during loading
    import gc
    gc.collect()
    
    print("[OpenSeek API] 🟢 Research-Grade Multi-Modal Engine Ready")

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
        # 1. Full Image Analysis (fast mode: skip Grad-CAM + patch scan)
        full_res = _ensemble.forward_analyze(temp_path, fast=True)
        
        # 2. Face-Focused Layer (quick detection only, no second full pass)
        img_cv = cv2.imread(temp_path)
        faces = _face_detector.detect(img_cv)
        
        final_probability = full_res["ai_probability"]
        
        # If faces found, boost probability slightly (no slow second forward pass)
        if faces:
            final_probability = min(1.0, full_res["ai_probability"] * 1.05)
        
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
        print(f"[OpenSeek API] Forensic Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



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

        # Full Image Analysis (fast mode: skip Grad-CAM + patch scan)
        full_res = _ensemble.forward_analyze(temp_path, fast=True)
        
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
        "model": "Advanced OpenSeek Multimodal Target",
        "models_loaded": _ensemble is not None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
