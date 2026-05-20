#!/bin/bash
echo "🚀 OpenSeek: Initiating Hard Reboot (Stable Forensic V5)..."

# Stopping existing instances
echo "🛑 Stopping existing instances..."
pkill -f "uvicorn main:app" || true

# Moving to backend directory
cd "$(dirname "$0")/backend"

# Starting in background
echo "🔥 Starting Backend (Universal Image Compatibility)..."
source venv/bin/activate
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > openseek_backend.log 2>&1 &

echo "✅ Backend is starting in the background (see openseek_backend.log)"
echo "⚠️ IMPORTANT: Now go to chrome://extensions and RELOAD OpenSeek to sync."
