"""
DeepShield — Dataset Download Guide & Structure Validator
==========================================================
Provides download instructions and folder structure validation
for all supported deepfake detection datasets.

Usage:
    # Validate your data directory structure:
    python training/datasets/download_datasets.py --verify --data_dir ./data

    # Print download instructions:
    python training/datasets/download_datasets.py --guide
"""
from __future__ import annotations

import os
import sys
import argparse

DATASETS = {
    # ── IMAGE DATASETS ────────────────────────────────────────────────────────
    "FaceForensics++": {
        "type": "image+video",
        "url": "https://github.com/ondyari/FaceForensics",
        "kaggle": None,
        "license": "Research only (requires form submission)",
        "notes": [
            "Download from: https://github.com/ondyari/FaceForensics#dataset",
            "Fill out the access form → you'll receive a Python download script",
            "Run: python faceforensics_download_v4.py . -d all -c c40 -t videos",
            "Contains: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter",
            "Extract frames: ffmpeg -i video.mp4 frames/%05d.jpg",
        ],
        "structure": "real/  and  fake/  with images or videos",
    },
    "Celeb-DF v2": {
        "type": "video",
        "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
        "kaggle": None,
        "license": "Research only",
        "notes": [
            "Request access: https://github.com/yuezunli/celeb-deepfakeforensics",
            "Contains: 590 real + 5639 fake celebrity videos",
            "Subject-level split: test set uses different celebrities than training",
        ],
        "structure": "Celeb-real/ → rename to real/,  Celeb-synthesis/ → rename to fake/",
    },
    "DFDC (DeepFake Detection Challenge)": {
        "type": "video",
        "url": "https://ai.facebook.com/datasets/dfdc/",
        "kaggle": "https://www.kaggle.com/competitions/deepfake-detection-challenge/data",
        "license": "Research only",
        "notes": [
            "Download from Kaggle (requires competition acceptance):",
            "  kaggle competitions download -c deepfake-detection-challenge",
            "~470 GB full dataset. Preview set: ~10 GB (dfdc_train_part_0.zip)",
            "metadata.json contains labels: 'label' field is 'FAKE' or 'REAL'",
        ],
        "structure": "Preview: dfdc_train_part_0/ contains .mp4 files + metadata.json",
    },
    "140k Real and Fake Faces (Kaggle)": {
        "type": "image",
        "url": "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces",
        "kaggle": "kaggle datasets download -d xhlulu/140k-real-and-fake-faces",
        "license": "Creative Commons",
        "notes": [
            "Quick start: 70k real + 70k StyleGAN2 faces",
            "Run: kaggle datasets download -d xhlulu/140k-real-and-fake-faces",
            "Unzip → train/real/ and train/fake/ (already structured!)",
        ],
        "structure": "Already in real/ and fake/ format",
    },
    "DeepFake and Real Images (Kaggle)": {
        "type": "image",
        "url": "https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images",
        "kaggle": "kaggle datasets download -d manjilkarki/deepfake-and-real-images",
        "license": "Creative Commons",
        "notes": [
            "Run: kaggle datasets download -d manjilkarki/deepfake-and-real-images",
            "Contains various GAN outputs + diffusion images",
        ],
        "structure": "Dataset/Train/Real and Dataset/Train/Fake",
    },
    # ── AUDIO DATASETS ────────────────────────────────────────────────────────
    "ASVspoof 2021": {
        "type": "audio",
        "url": "https://www.asvspoof.org/index2021.html",
        "kaggle": None,
        "license": "Research only",
        "notes": [
            "Download from: https://www.asvspoof.org/index2021.html",
            "LA (Logical Access) track most relevant for voice cloning detection",
            "Protocol file: ASVspoof2021.LA.cm.eval.trl.txt has labels",
            "Labels: 'genuine' = real, 'spoof' = fake",
        ],
        "structure": "flac/ folder → split into real/ and fake/ using protocol file",
    },
    "WaveFake": {
        "type": "audio",
        "url": "https://github.com/RUB-SysSec/WaveFake",
        "kaggle": "https://www.kaggle.com/datasets/mozillaorg/common-voice",
        "license": "MIT",
        "notes": [
            "Download from: https://zenodo.org/record/5642694",
            "~20 GB — contains real LJSpeech + 6 neural vocoders attacks",
            "Structure: ljspeech_real/ and various fake_* folders",
        ],
        "structure": "Merge all fake folders into fake/ and ljspeech into real/",
    },
    "FakeAVCeleb": {
        "type": "audio+video",
        "url": "https://github.com/DASH-Lab/FakeAVCeleb",
        "kaggle": None,
        "license": "Research only",
        "notes": [
            "Request access: https://github.com/DASH-Lab/FakeAVCeleb",
            "Contains: 500 real + 19,500 fake audio-visual pairs",
            "Both audio and video manipulation covered",
        ],
        "structure": "RealVideo-RealAudio/ → real/,  FakeVideo-FakeAudio/ → fake/",
    },
}


def print_guide():
    print("\n" + "=" * 70)
    print("  DeepShield — Dataset Download Guide")
    print("=" * 70)

    print("""
📂 RECOMMENDED FOLDER STRUCTURE FOR DEEPSHIELD TRAINING:

  data/
  ├── images/
  │   ├── real/    ← .jpg/.png/.webp authentic face images
  │   └── fake/    ← .jpg/.png/.webp deepfake/AI-generated images
  │
  ├── videos/
  │   ├── real/    ← .mp4/.avi authentic face videos
  │   └── fake/    ← .mp4/.avi deepfake videos
  │
  └── audio/
      ├── real/    ← .wav/.flac authentic voices
      └── fake/    ← .wav/.flac synthetic/cloned voices

🚀 QUICK START (smallest datasets first):
  1. Image: kaggle datasets download -d xhlulu/140k-real-and-fake-faces
  2. Audio: Download WaveFake from https://zenodo.org/record/5642694
  3. Video: FaceForensics++ preview (contact authors for script)
""")

    for name, info in DATASETS.items():
        print(f"{'─'*70}")
        print(f"  📦 {name}")
        print(f"     Type    : {info['type'].upper()}")
        print(f"     URL     : {info['url']}")
        if info.get("kaggle"):
            print(f"     Kaggle  : {info['kaggle']}")
        print(f"     License : {info['license']}")
        print(f"     Structure: {info['structure']}")
        print(f"     Notes:")
        for note in info["notes"]:
            print(f"       • {note}")
        print()

    print("=" * 70)
    print("""
⚡ MINIMUM VIABLE DATASET (for hackathon / quick testing):
  Image: 2000 real + 2000 fake faces (Kaggle dataset takes ~2 min to download)
  Video: 100 real + 100 fake clips (FF++ preview)
  Audio: 1000 real + 1000 fake clips (WaveFake subset)

🔑 KAGGLE SETUP (required for Kaggle datasets):
  pip install kaggle
  Create ~/.kaggle/kaggle.json with your API key from:
  https://www.kaggle.com/account → Create New API Token

📊 EXPECTED TRAINING TIME (with GT GPU like RTX 3080):
  Image: ~2 hours (40 epochs, 10k images)
  Video: ~4 hours (30 epochs, 1k videos)
  Audio: ~1 hour  (30 epochs, 5k clips)
  Fusion: ~30 min (20 epochs)
""")


def verify_structure(data_dir: str):
    print(f"\n[Verify] Checking data structure in: {data_dir}\n")
    all_ok = True

    for sub in ["images", "videos", "audio"]:
        sub_dir = os.path.join(data_dir, sub)
        if not os.path.isdir(sub_dir):
            print(f"  ⚠  {sub}/ not found (optional if you're only training one modality)")
            continue

        for label in ["real", "fake"]:
            label_dir = os.path.join(sub_dir, label)
            if not os.path.isdir(label_dir):
                print(f"  ❌ MISSING: {sub}/{label}/")
                all_ok = False
                continue

            files = [f for f in os.listdir(label_dir) if not f.startswith(".")]
            print(f"  ✅ {sub}/{label}/  → {len(files):,} files")
            if len(files) < 10:
                print(f"     ⚠  Very few files in {sub}/{label}/. Consider adding more.")

    print()
    if all_ok:
        print("  ✅ Data structure looks good! Ready to train.")
    else:
        print("  ❌ Some directories are missing. See download guide.")
        print("     Run: python training/datasets/download_datasets.py --guide")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Dataset Guide & Validator")
    parser.add_argument("--guide",    action="store_true", help="Print download guide")
    parser.add_argument("--verify",   action="store_true", help="Verify folder structure")
    parser.add_argument("--data_dir", default="./data",    help="Data root directory to verify")
    args = parser.parse_args()

    if args.guide or not (args.verify):
        print_guide()

    if args.verify:
        verify_structure(args.data_dir)
