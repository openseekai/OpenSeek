# Self-contained image for building with Railway "Root Directory = backend"
# (the repo-root ./Dockerfile is the equivalent for building from the repo root).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/cache/torch
ENV HF_HOME=/app/cache/huggingface

# System libs for OpenCV / MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libxcb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching (CPU-only PyTorch to save space)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-cache model weights (engine-aware: lean by default → just the detector)
RUN python download_models.py

EXPOSE 8080

# Honour Railway's $PORT; lean engine + local models by default
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 75"]
