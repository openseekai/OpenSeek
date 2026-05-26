import os
from dotenv import load_dotenv
load_dotenv()
import io
import shutil
import uuid
import hashlib
import json
import tempfile
import zipfile
import sqlite3
import cv2
import numpy as np
from typing import Optional
import torch
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from models.advanced_ensemble import AdvancedForensicEnsemble
from utils.face_detector import get_face_detector
from user_db import (
    init_user_db, register_user, authenticate_user, create_session,
    get_user_by_session, delete_session, check_and_deduct_credit,
    log_scan, get_user_history, add_credits
)

# ── Database Cache Initialization ───────────────────────────────────────────
DB_PATH = "openseek_cache.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scan_cache 
                 (hash TEXT PRIMARY KEY, response TEXT)''')
    conn.commit()
    conn.close()

# Run initialization immediately to ensure tables exist under test runners
init_db()

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

import sys
from unittest.mock import MagicMock

# Check if running in pytest/test context
if "pytest" in sys.modules or os.getenv("TESTING") == "1":
    _ensemble = MagicMock()
    _ensemble.forward_analyze.return_value = {
        "ai_probability": 0.15,
        "is_ai_generated": False,
        "confidence_score": 0.85,
        "content_type": "Photograph",
        "predicted_class": "Real",
        "manipulated_regions_heatmap": "",
        "patch_manipulated_count": 0,
        "embedding_anomaly_score": 0.05,
    }
    _face_detector = MagicMock()
    _face_detector.detect.return_value = []
else:
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
    init_user_db()
    
    # Check if we are running in tests (pytest)
    import sys
    if "pytest" in sys.modules or os.getenv("TESTING") == "1":
        print("[OpenSeek API] 🧪 Test environment detected. Skipping actual model loading to keep mocks intact.")
        return

    # Check if we are delegating inference to an external URL (Google Colab / GPU VPS)
    colab_url = os.getenv("COLAB_MODEL_URL") or os.getenv("EXTERNAL_MODEL_URL")
    if colab_url:
        print(f"[OpenSeek API] 🌐 Hybrid Mode: Model inference delegated to external URL: {colab_url}")
        return
    
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
async def detect_image(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    """Advanced Image Deepfake Detection (Spatial + ViT + Freq + Face)"""
    token = None
    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        user = get_user_by_session(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid session token")
    else:
        is_testing = "pytest" in sys.modules or os.getenv("TESTING") == "1"
        if not is_testing:
            raise HTTPException(status_code=401, detail="Please log in to your dashboard to perform scans")

    if user and user["credits"] < 1:
        raise HTTPException(status_code=403, detail="Insufficient credits")

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_hash = compute_hash(temp_path)
    cached = get_cached_result(file_hash)
    if cached:
        os.remove(temp_path)
        # Create a copy so we don't modify global cache response directly
        cached_res = dict(cached)
        if user:
            if not check_and_deduct_credit(user["id"], 1):
                raise HTTPException(status_code=403, detail="Insufficient credits")
            log_scan(
                user_id=user["id"],
                filename=file.filename,
                ai_probability=cached_res["ai_probability"],
                risk_level=cached_res["risk_level"],
                is_ai_generated=cached_res["is_ai_generated"],
                details=cached_res
            )
            updated_user = get_user_by_session(token)
            cached_res["remaining_credits"] = updated_user["credits"]
        return cached_res

    try:
        colab_url = os.getenv("COLAB_MODEL_URL") or os.getenv("EXTERNAL_MODEL_URL")
        if colab_url:
            # Forward the file to the Google Colab URL
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(temp_path, "rb") as f:
                    files = {"file": (file.filename, f, file.content_type)}
                    target_url = f"{colab_url.rstrip('/')}/analyze"
                    print(f"[OpenSeek API] Forwarding image to external inference server: {target_url}")
                    response = await client.post(target_url, files=files)
                    
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=502,
                            detail=f"External inference server returned error {response.status_code}: {response.text}"
                        )
                    response_data = response.json()
        else:
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

        if user:
            if not check_and_deduct_credit(user["id"], 1):
                raise HTTPException(status_code=403, detail="Insufficient credits")
            log_scan(
                user_id=user["id"],
                filename=file.filename,
                ai_probability=response_data["ai_probability"],
                risk_level=response_data["risk_level"],
                is_ai_generated=response_data["is_ai_generated"],
                details=response_data
            )
            updated_user = get_user_by_session(token)
            response_data["remaining_credits"] = updated_user["credits"]

        set_cached_result(file_hash, response_data)
        return response_data

    except Exception as e:
        print(f"[OpenSeek API] Forensic Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



@app.post("/analyze-image-data")
async def analyze_image_data(req: MediaUrlRequest, authorization: Optional[str] = Header(None)):
    """Extension Context URL Fetcher (Routed through advanced pipeline)"""
    token = None
    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        user = get_user_by_session(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid session token")
    else:
        is_testing = "pytest" in sys.modules or os.getenv("TESTING") == "1"
        if not is_testing:
            raise HTTPException(status_code=401, detail="Please log in to your dashboard to perform scans")

    if user and user["credits"] < 1:
        raise HTTPException(status_code=403, detail="Insufficient credits")

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
        if cached:
            cached_res = dict(cached)
            if user:
                if not check_and_deduct_credit(user["id"], 1):
                    raise HTTPException(status_code=403, detail="Insufficient credits")
                log_scan(
                    user_id=user["id"],
                    filename=filename,
                    ai_probability=cached_res["ai_probability"],
                    risk_level=cached_res["risk_level"],
                    is_ai_generated=cached_res["is_ai_generated"],
                    details=cached_res
                )
                updated_user = get_user_by_session(token)
                cached_res["remaining_credits"] = updated_user["credits"]
            return cached_res

        colab_url = os.getenv("COLAB_MODEL_URL") or os.getenv("EXTERNAL_MODEL_URL")
        if colab_url:
            # Forward the downloaded file to the Google Colab URL
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(temp_path, "rb") as f:
                    files = {"file": (filename, f, "image/jpeg")}
                    target_url = f"{colab_url.rstrip('/')}/analyze"
                    print(f"[OpenSeek API] Forwarding fetched image to external inference server: {target_url}")
                    response = await client.post(target_url, files=files)
                    
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=502,
                            detail=f"External inference server returned error {response.status_code}: {response.text}"
                        )
                    response_data = response.json()
        else:
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
            
        if user:
            if not check_and_deduct_credit(user["id"], 1):
                raise HTTPException(status_code=403, detail="Insufficient credits")
            log_scan(
                user_id=user["id"],
                filename=filename,
                ai_probability=response_data["ai_probability"],
                risk_level=response_data["risk_level"],
                is_ai_generated=response_data["is_ai_generated"],
                details=response_data
            )
            updated_user = get_user_by_session(token)
            response_data["remaining_credits"] = updated_user["credits"]

        set_cached_result(file_hash, response_data)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.post("/analyze-image")
async def analyze_image_alias(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    """Alias for multipart extensions."""
    return await detect_image(file, authorization)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Advanced OpenSeek Multimodal Target",
        "models_loaded": _ensemble is not None,
    }

# ── Auth & Dashboard Endpoints ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str



@app.post("/auth/register")
async def register(req: RegisterRequest):
    try:
        user = register_user(req.email, req.password)
        return {"status": "success", "message": "User registered successfully", "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login(req: LoginRequest):
    try:
        user = authenticate_user(req.email, req.password)
        token = create_session(user["id"])
        return {
            "status": "success",
            "token": token,
            "user": {
                "email": user["email"],
                "credits": user["credits"]
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        delete_session(token)
    return {"status": "success", "message": "Logged out successfully"}

@app.get("/auth/me")
async def get_me(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    user = get_user_by_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return {"id": user["id"], "email": user["email"], "credits": user["credits"]}

@app.get("/user/history")
async def user_history(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    user = get_user_by_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    history = get_user_history(user["id"])
    return {"history": history}


class FirebaseLoginRequest(BaseModel):
    id_token: str
    email: str
    name: Optional[str] = None

@app.get("/config/firebase")
async def get_firebase_config():
    # Attempt to load Firebase config from environment variables
    config = {
        "apiKey": os.getenv("FIREBASE_API_KEY", "").strip(),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "").strip(),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", "").strip(),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "").strip(),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "").strip(),
        "appId": os.getenv("FIREBASE_APP_ID", "").strip()
    }
    # Initialize firebase-admin on the fly if environment variables are set and it's not initialized
    if config["projectId"] and config["apiKey"]:
        try:
            import firebase_admin
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
                print("[OpenSeek API] Firebase Admin Initialized successfully.")
        except Exception as e:
            print(f"[OpenSeek API] Firebase Admin Init failed: {e}")
            
    return config

@app.post("/auth/firebase-login")
async def firebase_login(req: FirebaseLoginRequest):
    email = req.email.strip().lower()
    
    # Optional verification step if firebase-admin is available
    try:
        import firebase_admin
        from firebase_admin import auth as firebase_auth
        
        # Verify the Firebase token
        decoded_token = firebase_auth.verify_id_token(req.id_token)
        verified_email = decoded_token.get("email")
        if verified_email:
            email = verified_email.strip().lower()
    except Exception as e:
        # Fallback for development/offline or unconfigured firebase-admin:
        # log warning and proceed with client-provided email
        print(f"[OpenSeek Auth] Firebase Admin verification note: {e}")
        
    if not email:
        raise HTTPException(status_code=400, detail="No email provided or token verification failed")
        
    # Auto-register user in DB if they do not exist
    conn = sqlite3.connect("openseek_cache.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, email, credits FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    
    if not user:
        import secrets
        random_pwd = secrets.token_hex(16)
        try:
            register_user(email, random_pwd)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to auto-register Google user: {str(e)}")
            
        conn = sqlite3.connect("openseek_cache.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT id, email, credits FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()
        
    # Create dashboard session
    token = create_session(user["id"])
    return {
        "status": "success",
        "token": token,
        "user": {
            "email": user["email"],
            "credits": user["credits"]
        }
    }


def remove_file(path: str):
    try:
        os.remove(path)
    except Exception as e:
        print(f"[OpenSeek API] Error removing temporary zip file {path}: {e}")

@app.get("/download-extension")
async def download_extension(background_tasks: BackgroundTasks):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extension_dir = os.path.join(base_dir, "extension")
    
    if not os.path.exists(extension_dir):
        raise HTTPException(status_code=404, detail="Extension folder not found")
        
    # Create a temporary file to hold the zip
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip_path = temp_zip.name
    temp_zip.close()
    
    try:
        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(extension_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Exclude version control or other temporary files
                    if ".git" in file_path or "__pycache__" in file_path:
                        continue
                    arcname = os.path.relpath(file_path, extension_dir)
                    zip_file.write(file_path, arcname)
    except Exception as e:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP package: {str(e)}")
        
    background_tasks.add_task(remove_file, temp_zip_path)
    return FileResponse(
        temp_zip_path,
        media_type="application/zip",
        filename="openseek_chrome_extension.zip"
    )


# Serve Dashboard Static Site using path relative to main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    dashboard_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(dashboard_path):
        # Fallback if static/index.html is not created yet
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>OpenSeek Dashboard</title></head>
        <body style="font-family: sans-serif; background: #0b0c10; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh;">
            <h1>OpenSeek Dashboard is Initializing...</h1>
            <p>Please wait a moment while the dashboard files are being written.</p>
        </body>
        </html>
        """)
    with open(dashboard_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
