#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenSeek Backend — One-shot setup script
# Run from the backend/ directory:  bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

VENV_DIR="venv"

echo ""
echo "=== OpenSeek Backend Setup ==="
echo ""

# 1. Check python3 / ffmpeg
if ! command -v python3 &>/dev/null; then
    echo "❌  python3 not found. Please install Python 3.11+."
    exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "⚠  ffmpeg not found — video audio extraction will be disabled."
    echo "   Install with:  sudo apt install ffmpeg"
fi

# 2. Create venv (skip if already exists)
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# 3. Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# 4. Upgrade pip quietly
echo "→ Upgrading pip..."
pip install --upgrade pip --quiet

# 5. Install requirements
echo "→ Installing requirements (this may take a few minutes for PyTorch)..."
pip install -r requirements.txt

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Activate env : source venv/bin/activate"
echo "  Start server : uvicorn main:app --reload --port 8000"
echo "  Run tests    : pytest tests/ -v"
echo "  API docs     : http://localhost:8000/docs"
echo ""
