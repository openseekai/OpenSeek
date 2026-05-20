# OpenSeek — Deepfake Detection Backend

> Real-time AI-powered deepfake detection for images, videos, and audio.  
> Built for the **OpenSeek Chrome Extension** hackathon project.

---

## 🏗 Architecture

```
backend/
├── main.py                  # FastAPI app entry-point
├── config.py                # Global settings & constants
├── database.py              # Async SQLite logging
├── requirements.txt
├── Dockerfile               # Multi-stage, non-root user
├── docker-compose.yml
├── .env.example
├── models/
│   ├── image_model.py       # EfficientNet-B0 deepfake detector
│   ├── video_model.py       # Frame + audio pipeline
│   ├── audio_model.py       # MFCC CNN voice anti-spoof
│   └── multimodal.py        # Face-voice consistency
├── routers/
│   ├── image_router.py      # POST /analyze-image
│   ├── video_router.py      # POST /analyze-video
│   ├── audio_router.py      # POST /analyze-audio
│   └── health_router.py     # GET  /health
├── utils/
│   ├── download.py          # Safe async downloader (SSRF-proof)
│   ├── validators.py        # URL validation
│   └── face_detector.py     # MediaPipe face detection
└── tests/
    └── test_endpoints.py    # pytest integration tests
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- `ffmpeg` installed and on `$PATH`

```bash
cd backend

# 1. Create virtualenv
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
uvicorn main:app --reload --port 8000
```

Server available at: **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

---

## 🐳 Docker

```bash
cd backend

# Build + start
docker compose up --build

# Stop
docker compose down
```

The SQLite database is persisted in a Docker volume (`openseek-data`).

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "ok", "models_loaded": true }
```

### `POST /analyze-image`
```json
// Request
{ "url": "https://example.com/photo.jpg" }

// Response
{
  "type": "image",
  "authenticity_score": 82.4,
  "risk_level": "High",
  "face_detected": true,
  "analysis": {
    "facial_inconsistency": 0.61,
    "lighting_mismatch":    0.42,
    "gan_artifacts":        0.37
  }
}
```

### `POST /analyze-video`
```json
// Request
{ "url": "https://example.com/clip.mp4" }

// Response
{
  "type": "video",
  "authenticity_score": 64.0,
  "risk_level": "Medium",
  "video_score": 70.0,
  "audio_score": 52.0,
  "face_voice_match": "Mismatch detected",
  "frame_analysis": {
    "total_frames": 18,
    "suspicious_frames": 6
  }
}
```

### `POST /analyze-audio`
```json
// Request
{ "url": "https://example.com/voice.mp3" }

// Response
{
  "type": "audio",
  "authenticity_score": 59.0,
  "risk_level": "High",
  "analysis": {
    "synthetic_probability": 0.73,
    "pitch_irregularity":    0.41
  }
}
```

---

## 🔐 Security Features

| Feature | Implementation |
|---|---|
| SSRF protection | Private/reserved IP ranges blocked before download |
| MIME type gating | `Content-Type` header validated before body downloaded |
| Size limits | Image 10 MB · Video 100 MB · Audio 20 MB |
| Rate limiting | 30 requests / minute per IP (SlowAPI) |
| URL validation | Pydantic validator — http/https only, no control chars |

---

## 📊 Risk Levels

| Score range | Risk Level |
|---|---|
| 0 – 40 | 🟢 Low |
| 41 – 70 | 🟡 Medium |
| 71 – 100 | 🔴 High |

---

## 🧪 Running Tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

---

## 🔧 Configuration

Copy `.env.example` → `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode |
| `DB_PATH` | `openseek.db` | SQLite database path |

---

## 📝 Detection Logging

Every request is logged to SQLite (`openseek.db`):

```sql
CREATE TABLE detections (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT,
    media_type        TEXT,   -- image | video | audio
    url               TEXT,
    authenticity_score REAL,
    risk_level        TEXT
);
```

---

## 🚢 Production Notes

- For **real deepfake detection accuracy**, replace the EfficientNet-B0 backbone weights with a model fine-tuned on [DFDC](https://ai.facebook.com/datasets/dfdc/) or [FaceForensics++](https://github.com/ondyari/FaceForensics)
- For audio, fine-tune the CNN on [ASVspoof 2019/2021](https://www.asvspoof.org/) data
- Scale horizontally by increasing `--workers` in the uvicorn CMD (stateless design)
- Add Redis + `slowapi` Redis backend for distributed rate limiting at scale
# Deepfake_detection_tool
