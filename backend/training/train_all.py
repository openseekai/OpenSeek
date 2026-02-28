"""
DeepShield — Master Training Orchestrator
==========================================
Runs the full training pipeline in sequence:
  1. Image model   (DualStream EfficientNet-B4 + FFT)
  2. Video model   (EfficientNet-B0 + Temporal Transformer)
  3. Audio model   (2D CNN + Attention)
  4. Fusion module (Cross-Modal Attention)

Usage:
    cd /path/to/backend

    # Full run (all modalities):
    python training/train_all.py \\
        --image_data ./data/images \\
        --video_data ./data/videos \\
        --audio_data ./data/audio \\
        --epochs_image 40 \\
        --epochs_video 30 \\
        --epochs_audio 30 \\
        --epochs_fusion 20

    # Image only:
    python training/train_all.py --image_data ./data/images --skip_video --skip_audio --skip_fusion

    # Smoke test (all modalities, tiny data):
    python training/train_all.py \\
        --image_data ./data_smoke \\
        --video_data ./data_smoke \\
        --audio_data ./data_smoke \\
        --epochs_image 1 --epochs_video 1 --epochs_audio 1 --epochs_fusion 1 \\
        --max_samples 50 --batch_image 4 --batch_video 2 --batch_audio 8
"""
from __future__ import annotations

import os
import sys
import time
import argparse

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def _header(title: str):
    print(f"\n{'#'*60}")
    print(f"  {title}")
    print(f"{'#'*60}\n")


def run_image(args):
    _header("PHASE 1 — Image Model Training")
    from training.image_train import train_image

    class ImageArgs:
        data_dir       = args.image_data
        epochs         = args.epochs_image
        batch_size     = args.batch_image
        lr             = args.lr
        weight_decay   = 1e-4
        warmup_epochs  = args.warmup
        patience       = args.patience
        image_size     = 224
        num_workers    = args.num_workers
        mixup_prob     = 0.4
        cutmix_prob    = 0.3
        max_samples    = args.max_samples
        freeze_backbone = False
        checkpoint_dir = args.checkpoint_dir

    t0 = time.time()
    train_image(ImageArgs())
    print(f"\n  ✅ Image training done in {(time.time()-t0)/60:.1f} min")


def run_video(args):
    _header("PHASE 2 — Video Model Training")
    from training.video_train import train_video

    class VideoArgs:
        data_dir       = args.video_data
        epochs         = args.epochs_video
        batch_size     = args.batch_video
        accum_steps    = args.accum_steps
        lr             = args.lr
        weight_decay   = 1e-4
        warmup_epochs  = args.warmup
        patience       = args.patience
        num_frames     = args.num_frames
        d_model        = 512
        nhead          = 8
        num_layers     = 4
        freeze_blocks  = 4
        image_size     = 224
        num_workers    = args.num_workers
        max_samples    = args.max_samples
        checkpoint_dir = args.checkpoint_dir

    t0 = time.time()
    train_video(VideoArgs())
    print(f"\n  ✅ Video training done in {(time.time()-t0)/60:.1f} min")


def run_audio(args):
    _header("PHASE 3 — Audio Model Training")
    from training.audio_train import train_audio

    class AudioArgs:
        data_dir       = args.audio_data
        epochs         = args.epochs_audio
        batch_size     = args.batch_audio
        lr             = args.lr
        weight_decay   = 1e-4
        warmup_epochs  = args.warmup
        patience       = args.patience
        sample_rate    = 16000
        duration       = 4.0
        num_workers    = args.num_workers
        max_samples    = args.max_samples
        checkpoint_dir = args.checkpoint_dir

    t0 = time.time()
    train_audio(AudioArgs())
    print(f"\n  ✅ Audio training done in {(time.time()-t0)/60:.1f} min")


def run_fusion(args):
    _header("PHASE 4 — Fusion Module Training")
    from training.fusion_train import train_fusion

    class FusionArgs:
        image_data     = args.image_data
        video_data     = args.video_data
        audio_data     = args.audio_data
        image_ckpt     = os.path.join(args.checkpoint_dir, "image", "best_model.pt")
        video_ckpt     = os.path.join(args.checkpoint_dir, "video", "best_model.pt")
        audio_ckpt     = os.path.join(args.checkpoint_dir, "audio", "best_model.pt")
        epochs         = args.epochs_fusion
        batch_size     = args.batch_fusion
        lr             = args.lr * 0.5   # Lower LR for fusion fine-tuning
        weight_decay   = 1e-4
        patience       = args.patience
        num_frames     = args.num_frames
        d_model        = 512
        num_workers    = args.num_workers
        checkpoint_dir = args.checkpoint_dir

    t0 = time.time()
    train_fusion(FusionArgs())
    print(f"\n  ✅ Fusion training done in {(time.time()-t0)/60:.1f} min")


def print_summary(results: dict):
    print(f"\n{'='*60}")
    print("  DeepShield Training Complete — Final Summary")
    print(f"{'='*60}")
    for phase, result in results.items():
        status = "✅" if result else "⏭ Skipped"
        print(f"  {status}  {phase}")
    print()
    print("  Next steps:")
    print("  1. Check training/checkpoints/ for best_model.pt files")
    print("  2. Restart the backend: uvicorn main:app --reload")
    print("  3. The backend will auto-use the new weights if you update")
    print("     main.py to load from training/checkpoints/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="DeepShield — Full Training Pipeline")

    # Data dirs
    parser.add_argument("--image_data",    default=None, help="Image dataset dir (real/ + fake/)")
    parser.add_argument("--video_data",    default=None, help="Video dataset dir (real/ + fake/)")
    parser.add_argument("--audio_data",    default=None, help="Audio dataset dir (real/ + fake/)")

    # Skip flags
    parser.add_argument("--skip_image",   action="store_true")
    parser.add_argument("--skip_video",   action="store_true")
    parser.add_argument("--skip_audio",   action="store_true")
    parser.add_argument("--skip_fusion",  action="store_true")

    # Epochs per phase
    parser.add_argument("--epochs_image",  type=int, default=40)
    parser.add_argument("--epochs_video",  type=int, default=30)
    parser.add_argument("--epochs_audio",  type=int, default=30)
    parser.add_argument("--epochs_fusion", type=int, default=20)

    # Batch sizes per modality
    parser.add_argument("--batch_image",   type=int, default=16)
    parser.add_argument("--batch_video",   type=int, default=4)
    parser.add_argument("--batch_audio",   type=int, default=32)
    parser.add_argument("--batch_fusion",  type=int, default=8)

    # Shared hyperparams
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--warmup",        type=int,   default=5)
    parser.add_argument("--patience",      type=int,   default=7)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--num_frames",    type=int,   default=16)
    parser.add_argument("--accum_steps",   type=int,   default=4)
    parser.add_argument("--max_samples",   type=int,   default=None, help="Cap samples (for smoke tests)")
    parser.add_argument("--checkpoint_dir", default="training/checkpoints")

    args = parser.parse_args()

    results = {}
    t_start = time.time()

    # Phase 1: Image
    if not args.skip_image:
        if args.image_data and os.path.isdir(args.image_data):
            run_image(args)
            results["Image  (EfficientNet-B4 + FFT)"] = True
        else:
            print("  ⚠ --image_data not provided or not found. Skipping image training.")
            results["Image  (EfficientNet-B4 + FFT)"] = False
    else:
        results["Image  (EfficientNet-B4 + FFT)"] = False

    # Phase 2: Video
    if not args.skip_video:
        if args.video_data and os.path.isdir(args.video_data):
            run_video(args)
            results["Video  (Temporal Transformer)"] = True
        else:
            print("  ⚠ --video_data not provided or not found. Skipping video training.")
            results["Video  (Temporal Transformer)"] = False
    else:
        results["Video  (Temporal Transformer)"] = False

    # Phase 3: Audio
    if not args.skip_audio:
        if args.audio_data and os.path.isdir(args.audio_data):
            run_audio(args)
            results["Audio  (2D CNN + Attention)"] = True
        else:
            print("  ⚠ --audio_data not provided or not found. Skipping audio training.")
            results["Audio  (2D CNN + Attention)"] = False
    else:
        results["Audio  (2D CNN + Attention)"] = False

    # Phase 4: Fusion
    if not args.skip_fusion:
        img_ckpt = os.path.join(args.checkpoint_dir, "image", "best_model.pt")
        vid_ckpt = os.path.join(args.checkpoint_dir, "video", "best_model.pt")
        aud_ckpt = os.path.join(args.checkpoint_dir, "audio", "best_model.pt")
        if all(os.path.exists(c) for c in [img_ckpt, vid_ckpt, aud_ckpt]):
            if args.image_data and args.video_data and args.audio_data:
                run_fusion(args)
                results["Fusion (Cross-Modal Attention)"] = True
            else:
                print("  ⚠ All 3 data dirs required for fusion training. Skipping.")
                results["Fusion (Cross-Modal Attention)"] = False
        else:
            print("  ⚠ Missing branch checkpoints. Train all 3 modalities first.")
            results["Fusion (Cross-Modal Attention)"] = False
    else:
        results["Fusion (Cross-Modal Attention)"] = False

    total_min = (time.time() - t_start) / 60
    print(f"\n  Total wall-clock time: {total_min:.1f} min")
    print_summary(results)


if __name__ == "__main__":
    main()
