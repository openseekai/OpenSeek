run """
OpenSeek — Dataset Verifier
==============================
Checks that the data/ folder is valid before training.

Usage:
    python scripts/verify_dataset.py --data_dir ./data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_RECOMMENDED = 500  # per class


def collect_images(folder: Path) -> list[Path]:
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def verify_images(images: list[Path], label: str) -> tuple[int, list[str]]:
    """Open each image and report corrupt files."""
    corrupt = []
    for img_path in tqdm(images, desc=f"  Checking {label}", unit="img"):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            corrupt.append(f"{img_path.name}: {e}")
    return len(images), corrupt


def main():
    parser = argparse.ArgumentParser(description="Verify OpenSeek training dataset")
    parser.add_argument("--data_dir", default="./data", help="Dataset root folder")
    parser.add_argument("--no_integrity", action="store_true",
                        help="Skip per-image integrity check (faster)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"

    print("=" * 60)
    print("  OpenSeek — Dataset Verification")
    print("=" * 60)

    ok = True

    # ── Structure check ─────────────────────────────────────────────────────
    if not data_dir.exists():
        print(f"  ❌  data_dir not found: {data_dir.resolve()}")
        sys.exit(1)

    for d, name in [(real_dir, "real/"), (fake_dir, "fake/")]:
        if not d.exists():
            print(f"  ❌  Missing subfolder: {name}")
            ok = False

    if not ok:
        print("\n  Run: python scripts/prepare_dataset.py --output_dir ./data")
        sys.exit(1)

    # ── Count images ────────────────────────────────────────────────────────
    real_imgs = collect_images(real_dir)
    fake_imgs = collect_images(fake_dir)

    print(f"\n  📁 {real_dir}")
    print(f"     Images: {len(real_imgs):,}")
    print(f"\n  📁 {fake_dir}")
    print(f"     Images: {len(fake_imgs):,}")

    if len(real_imgs) < MIN_RECOMMENDED:
        print(f"\n  ⚠  Only {len(real_imgs)} real images found. "
              f"Recommend ≥{MIN_RECOMMENDED} per class for meaningful training.")
        ok = False
    if len(fake_imgs) < MIN_RECOMMENDED:
        print(f"\n  ⚠  Only {len(fake_imgs)} fake images found. "
              f"Recommend ≥{MIN_RECOMMENDED} per class.")
        ok = False

    # ── Class balance ───────────────────────────────────────────────────────
    total = len(real_imgs) + len(fake_imgs)
    imbalance = abs(len(real_imgs) - len(fake_imgs)) / max(total, 1)
    if imbalance > 0.25:
        minority = "real" if len(real_imgs) < len(fake_imgs) else "fake"
        print(f"\n  ⚠  Class imbalance {imbalance*100:.0f}% — '{minority}' class has fewer images.")
        print(f"     Training will use weighted sampling to compensate.")

    # ── File integrity ──────────────────────────────────────────────────────
    if not args.no_integrity:
        print("\n  Running image integrity check…")
        _, real_corrupt = verify_images(real_imgs, "real")
        _, fake_corrupt = verify_images(fake_imgs, "fake")

        all_corrupt = real_corrupt + fake_corrupt
        if all_corrupt:
            print(f"\n  ⚠  {len(all_corrupt)} corrupt file(s) found:")
            for c in all_corrupt[:10]:
                print(f"     {c}")
            if len(all_corrupt) > 10:
                print(f"     ... and {len(all_corrupt) - 10} more")
            print("  These will be skipped during training automatically.")
        else:
            print("  ✅ All images passed integrity check.")
    else:
        print("\n  (Integrity check skipped with --no_integrity)")

    # ── Extensions breakdown ────────────────────────────────────────────────
    all_imgs = real_imgs + fake_imgs
    ext_counts: dict[str, int] = {}
    for p in all_imgs:
        ext_counts[p.suffix.lower()] = ext_counts.get(p.suffix.lower(), 0) + 1

    print("\n  Extension breakdown:")
    for ext, cnt in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"    {ext:8s}  {cnt:,}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if ok:
        print(f"  ✅ Dataset looks good! Total: {total:,} images")
        print(f"     Real: {len(real_imgs):,}  |  Fake: {len(fake_imgs):,}")
        print()
        print("  Ready to train:")
        print(f"  python train/train_all.py --data_dir {data_dir} --epochs 20")
    else:
        print("  ⚠  Dataset has warnings — training may still work but accuracy could be lower.")
        print()
        print("  Fix issues then re-run: python scripts/prepare_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
