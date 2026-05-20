"""
OpenSeek Training — Subject-Level Split
==========================================
Implements identity-aware data splitting that prevents the same person
from appearing in both training and test sets (no identity leakage).

Key functions:
  split_by_subject()     : Core subject-level splitting function
  load_identity_map()    : Load identity→files mapping from various formats
  cross_dataset_split()  : Split for cross-dataset generalization testing

Why subject-level splits matter:
  Random frame-level splits inflate accuracy, because the same face
  appears in train and test. A model trained this way memorizes identities
  rather than learning generalizable deepfake artifacts.

Usage:
    from training.subject_split import split_by_subject, load_identity_map

    # When you have FF++ style filenames (actor IDs in name):
    identity_map = load_identity_map(data_dir, format="ffpp")
    train_ids, val_ids, test_ids = split_by_subject(identity_map)
    train_files = [f for id in train_ids for f in identity_map[id]]

    # When filenames don't encode identity (fallback):
    from training.utils.datasets import _collect_files, _subject_aware_split
    samples = _collect_files(data_dir, IMAGE_EXTS)
    train_s, val_s, test_s = _subject_aware_split(samples)
"""
from __future__ import annotations

import os
import re
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set


# ── Type aliases ──────────────────────────────────────────────────────────────
IdentityMap  = Dict[str, List[Tuple[str, int]]]  # identity_id → [(path, label), ...]
SplitResult  = Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]


# ── Core Split ────────────────────────────────────────────────────────────────

def split_by_subject(
    identity_map: IdentityMap,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    test_ratio: float  = 0.15,
    seed: int = 42,
    min_samples_per_id: int = 1,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split identity IDs into train/val/test groups.
    
    Args:
        identity_map        : Dict mapping subject_id → list of (path, label) tuples.
        train_ratio         : Fraction of subjects for training.
        val_ratio           : Fraction of subjects for validation.
        test_ratio          : Fraction of subjects for test.
        seed                : Random seed for reproducibility.
        min_samples_per_id  : Exclude identities with fewer than this many samples.
    
    Returns:
        (train_ids, val_ids, test_ids) — lists of subject ID strings.
    
    Example:
        train_ids, val_ids, test_ids = split_by_subject(identity_map)
        train_samples = [s for id in train_ids for s in identity_map[id]]
        val_samples   = [s for id in val_ids   for s in identity_map[id]]
        test_samples  = [s for id in test_ids  for s in identity_map[id]]
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)

    # Filter identities with too few samples
    valid_ids = [
        id_ for id_, samples in identity_map.items()
        if len(samples) >= min_samples_per_id
    ]
    rng.shuffle(valid_ids)

    n = len(valid_ids)
    if n == 0:
        raise ValueError("No valid identities found in identity_map")

    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    # Ensure the split sums correctly
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    train_ids = valid_ids[:n_train]
    val_ids   = valid_ids[n_train:n_train + n_val]
    test_ids  = valid_ids[n_train + n_val:]

    _print_split_stats(identity_map, train_ids, val_ids, test_ids)
    return train_ids, val_ids, test_ids


def get_samples_from_ids(
    identity_map: IdentityMap,
    subject_ids: List[str],
) -> List[Tuple[str, int]]:
    """Flatten identity_map for a given list of subject IDs into a flat sample list."""
    return [sample for id_ in subject_ids for sample in identity_map[id_]]


def _print_split_stats(
    identity_map: IdentityMap,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
):
    def _count(ids):
        samples = get_samples_from_ids(identity_map, ids)
        n_real = sum(1 for _, l in samples if l == 0)
        n_fake = sum(1 for _, l in samples if l == 1)
        return len(ids), n_real, n_fake

    tr_n, tr_r, tr_f = _count(train_ids)
    va_n, va_r, va_f = _count(val_ids)
    te_n, te_r, te_f = _count(test_ids)

    print(f"\n  [SubjectSplit] {'Split':<8} {'Subjects':>8} {'Real':>8} {'Fake':>8} {'Total':>8}")
    print(f"  {'─'*48}")
    print(f"  [SubjectSplit] {'Train':<8} {tr_n:>8,} {tr_r:>8,} {tr_f:>8,} {tr_r+tr_f:>8,}")
    print(f"  [SubjectSplit] {'Val':<8} {va_n:>8,} {va_r:>8,} {va_f:>8,} {va_r+va_f:>8,}")
    print(f"  [SubjectSplit] {'Test':<8} {te_n:>8,} {te_r:>8,} {te_f:>8,} {te_r+te_f:>8,}")

    # Verify no overlap
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)
    assert not (train_set & val_set),   "❌ Train/Val identity overlap!"
    assert not (train_set & test_set),  "❌ Train/Test identity overlap!"
    assert not (val_set   & test_set),  "❌ Val/Test identity overlap!"
    print(f"  [SubjectSplit] ✅ No identity overlap between splits.\n")


# ── Identity Map Loaders ──────────────────────────────────────────────────────

def load_identity_map(
    data_dir: str,
    format: str = "folder",
    extensions: Optional[Set[str]] = None,
) -> IdentityMap:
    """
    Build an identity_map from various dataset structures.
    
    Supported formats:
    
    "folder":   Each subject has a subdirectory.
        data_dir/
          real/
            subject_001/   ← identity folder
              img_001.jpg
            subject_002/
          fake/
            subject_001/
    
    "ffpp":     FaceForensics++ naming: {actor_id}_{clip_id}.jpg
        data_dir/
          real/ 000_001.jpg, 000_002.jpg, 001_001.jpg ...
          fake/ ...
    
    "dfdc":     DFDC JSON metadata: metadata.json with {filename: {label: "FAKE"}}
        data_dir/
          metadata.json
          *.mp4
    
    "flat":     Flat directory — uses filename prefix (2 chars) as identity proxy.
        data_dir/
          real/ *.jpg
          fake/ *.jpg
    
    Args:
        data_dir   : Root directory containing real/ and fake/ subfolders.
        format     : Dataset format ("folder", "ffpp", "dfdc", "flat").
        extensions : File extensions to include. Default: common image+video+audio.
    
    Returns:
        identity_map: {subject_id: [(file_path, label), ...]}
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".wav", ".flac", ".mp3"}

    identity_map: IdentityMap = defaultdict(list)

    if format == "folder":
        _load_folder_format(data_dir, extensions, identity_map)
    elif format == "ffpp":
        _load_ffpp_format(data_dir, extensions, identity_map)
    elif format == "dfdc":
        _load_dfdc_format(data_dir, identity_map)
    elif format == "flat":
        _load_flat_format(data_dir, extensions, identity_map)
    else:
        raise ValueError(f"Unknown format '{format}'. Choose: folder, ffpp, dfdc, flat")

    print(f"  [IdentityMap] Loaded {len(identity_map):,} unique identities "
          f"({sum(len(v) for v in identity_map.values()):,} total samples)")
    return dict(identity_map)


def _load_folder_format(data_dir: str, exts: set, identity_map: IdentityMap):
    for sub, label in [("real", 0), ("fake", 1)]:
        sub_dir = os.path.join(data_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for subject_id in sorted(os.listdir(sub_dir)):
            subject_dir = os.path.join(sub_dir, subject_id)
            if not os.path.isdir(subject_dir):
                continue
            for fname in sorted(os.listdir(subject_dir)):
                if os.path.splitext(fname)[1].lower() in exts:
                    identity_map[subject_id].append(
                        (os.path.join(subject_dir, fname), label)
                    )


def _load_ffpp_format(data_dir: str, exts: set, identity_map: IdentityMap):
    """FaceForensics++ filenames: {actor_id}_{clip_id}.{ext}"""
    for sub, label in [("real", 0), ("fake", 1)]:
        sub_dir = os.path.join(data_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for fname in sorted(os.listdir(sub_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in exts:
                continue
            # Extract actor ID from filename prefix (e.g., "000_001.jpg" → "000")
            match = re.match(r"^(\d+)", fname)
            subject_id = match.group(1) if match else fname[:3]
            identity_map[f"{sub}_{subject_id}"].append(
                (os.path.join(sub_dir, fname), label)
            )


def _load_dfdc_format(data_dir: str, identity_map: IdentityMap):
    """DFDC format: metadata.json with {filename: {label: 'FAKE'/'REAL'}}"""
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  [IdentityMap] Warning: metadata.json not found in {data_dir}")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    for fname, info in metadata.items():
        label_str = info.get("label", "REAL").upper()
        label = 1 if label_str == "FAKE" else 0
        # Use DFDC original video as subject proxy
        original = info.get("original", fname)
        subject_id = os.path.splitext(original)[0]
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            identity_map[subject_id].append((fpath, label))


def _load_flat_format(data_dir: str, exts: set, identity_map: IdentityMap):
    """Flat directory with no subject structure — uses filename prefix as proxy."""
    for sub, label in [("real", 0), ("fake", 1)]:
        sub_dir = os.path.join(data_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for fname in sorted(os.listdir(sub_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in exts:
                continue
            # Use first 3 characters as subject proxy (crude but deterministic)
            subject_id = f"{sub}_{fname[:3].lower()}"
            identity_map[subject_id].append((os.path.join(sub_dir, fname), label))


# ── Cross-Dataset Split ────────────────────────────────────────────────────────

def cross_dataset_split(
    train_dirs: List[str],
    test_dirs: List[str],
    extensions: Optional[Set[str]] = None,
    format: str = "flat",
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Build a cross-dataset evaluation setup:
    - Train on: FaceForensics++ + Celeb-DF
    - Test on:  DFDC (completely separate dataset)
    
    Args:
        train_dirs : Directories to use for training.
        test_dirs  : Directories to use for testing (unseen datasets).
        extensions : File extensions to include.
        format     : Dataset format for identity loading.
    
    Returns:
        (train_samples, test_samples) — lists of (filepath, label) tuples.
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi"}

    train_samples: List[Tuple[str, int]] = []
    test_samples:  List[Tuple[str, int]] = []

    print(f"\n  [CrossDataset] Building cross-dataset evaluation...")
    print(f"  [CrossDataset] Train sources: {[os.path.basename(d) for d in train_dirs]}")
    print(f"  [CrossDataset] Test  sources: {[os.path.basename(d) for d in test_dirs]}")

    for d in train_dirs:
        if os.path.isdir(d):
            id_map = load_identity_map(d, format=format, extensions=extensions)
            for samples in id_map.values():
                train_samples.extend(samples)

    for d in test_dirs:
        if os.path.isdir(d):
            id_map = load_identity_map(d, format=format, extensions=extensions)
            for samples in id_map.values():
                test_samples.extend(samples)

    n_train_r = sum(1 for _, l in train_samples if l == 0)
    n_train_f = sum(1 for _, l in train_samples if l == 1)
    n_test_r  = sum(1 for _, l in test_samples  if l == 0)
    n_test_f  = sum(1 for _, l in test_samples  if l == 1)

    print(f"  [CrossDataset] Train: {n_train_r:,} real + {n_train_f:,} fake = {len(train_samples):,}")
    print(f"  [CrossDataset] Test : {n_test_r:,}  real + {n_test_f:,}  fake = {len(test_samples):,}")

    return train_samples, test_samples


# ── Duplicate Removal ─────────────────────────────────────────────────────────

def remove_duplicate_identities(
    identity_map: IdentityMap,
    target_map: IdentityMap,
) -> IdentityMap:
    """
    Remove from `identity_map` any subjects whose IDs appear in `target_map`.
    Useful when combining multiple datasets to avoid the same person
    leaking across the train/test boundary.
    
    Args:
        identity_map : Source identity map to filter.
        target_map   : Reference identity map (subjects to exclude).
    
    Returns:
        Filtered identity map.
    """
    exclude_ids = set(target_map.keys())
    filtered = {id_: v for id_, v in identity_map.items() if id_ not in exclude_ids}
    removed = len(identity_map) - len(filtered)
    if removed > 0:
        print(f"  [Dedup] Removed {removed} duplicate identities from combined dataset.")
    return filtered


# ── Convenience Wrapper ────────────────────────────────────────────────────────

def build_subject_split(
    data_dir: str,
    format: str = "flat",
    extensions: Optional[Set[str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    One-call convenience function: build identity map and split in one step.
    
    Returns:
        (train_samples, val_samples, test_samples) flat lists of (path, label).
    
    Example:
        train_s, val_s, test_s = build_subject_split("./data/images", format="flat")
    """
    identity_map = load_identity_map(data_dir, format=format, extensions=extensions)
    train_ids, val_ids, test_ids = split_by_subject(
        identity_map, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    return (
        get_samples_from_ids(identity_map, train_ids),
        get_samples_from_ids(identity_map, val_ids),
        get_samples_from_ids(identity_map, test_ids),
    )
