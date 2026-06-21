#!/usr/bin/env python3
"""
OpenSeek benchmark & label-verification tool.

Two jobs:

1. Verify the HuggingFace label mapping (cheap, no labelled data needed):

       python scripts/benchmark.py --verify-labels path/to/any_image.jpg

   Prints the RAW {label: score} output of each HF model so you can confirm
   which label means "AI / fake". The engine assumes the deepfake/fake label
   is the AI class — this proves it.

2. Measure real accuracy on a labelled folder:

       python scripts/benchmark.py --data_dir path/to/dataset

   The folder must contain two sub-folders:
       dataset/real/   <- genuine photos
       dataset/fake/   <- AI-generated / deepfake images

   Prints accuracy, precision, recall, F1 (and AUC if scikit-learn is present).
   An image is predicted AI when ai_probability > THRESHOLD (default 0.5).

Run it from the `backend/` directory so the model imports resolve.
"""
import argparse
import os
import sys

# Allow running as `python scripts/benchmark.py` from the backend/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def verify_labels(image_path: str):
    """Dump raw HF classifier outputs so the label->AI mapping is auditable."""
    from transformers import pipeline

    models = [
        ("primary", os.environ.get("OPENSEEK_DETECTOR_MODEL", "haywoodsloan/ai-image-detector-deploy")),
        ("face", "dima806/deepfake_vs_real_image_detection"),
    ]
    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    from models.advanced_ensemble import AdvancedForensicEnsemble

    for name, model_id in models:
        print(f"\n=== {name}: {model_id} ===")
        try:
            clf = pipeline("image-classification", model=model_id, top_k=None, device=-1)
            out = clf(img)
            for r in out:
                print(f"   label={r['label']!r:<28} score={r['score']:.4f}")
            ai_prob = AdvancedForensicEnsemble._ai_prob_from_classifier(out)
            print(f"   --> engine reads P(AI) = {ai_prob}")
        except Exception as e:
            print(f"   FAILED to load/run: {e}")


def _list_images(folder: str):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(IMAGE_EXTS)
    ]


def run_benchmark(data_dir: str, threshold: float):
    import torch
    from models.advanced_ensemble import AdvancedForensicEnsemble

    real_imgs = _list_images(os.path.join(data_dir, "real"))
    fake_imgs = _list_images(os.path.join(data_dir, "fake"))
    if not real_imgs and not fake_imgs:
        print(f"No images found under {data_dir}/real or {data_dir}/fake")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ensemble on {device} ... (first run downloads HF models)")
    engine = AdvancedForensicEnsemble(device)

    y_true, y_prob = [], []
    samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
    print(f"Scoring {len(samples)} images "
          f"({len(real_imgs)} real, {len(fake_imgs)} fake)...\n")

    for path, label in samples:
        try:
            res = engine.forward_analyze(path, fast=True)
            prob = float(res.get("ai_probability", 0.0))
        except Exception as e:
            print(f"   skip {os.path.basename(path)}: {e}")
            continue
        y_true.append(label)
        y_prob.append(prob)
        pred = "AI " if prob > threshold else "REAL"
        truth = "AI " if label else "REAL"
        mark = "ok " if (prob > threshold) == bool(label) else "XX "
        print(f"   {mark} truth={truth} pred={pred} p={prob:.3f}  {os.path.basename(path)}")

    _report_metrics(y_true, y_prob, threshold)


def _report_metrics(y_true, y_prob, threshold):
    """Print accuracy / precision / recall / F1 / AUC for scored samples."""
    if not y_true:
        print("\nNo images scored successfully.")
        return
    y_pred = [1 if p > threshold else 0 for p in y_prob]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print("\n" + "=" * 40)
    print(f"  Samples     : {n}")
    print(f"  Threshold   : {threshold}")
    print(f"  Accuracy    : {acc:.3f}")
    print(f"  Precision   : {prec:.3f}  (of flagged-AI, how many were AI)")
    print(f"  Recall      : {rec:.3f}  (of real AI, how many we caught)")
    print(f"  F1          : {f1:.3f}")
    print(f"  Confusion   : TP={tp} TN={tn} FP={fp} FN={fn}")
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) == 2:
            print(f"  ROC-AUC     : {roc_auc_score(y_true, y_prob):.3f}")
    except Exception:
        print("  ROC-AUC     : (install scikit-learn for AUC)")
    print("=" * 40)


def _login(base_url, email, password):
    """Exchange email/password for a session token via /auth/login."""
    import httpx
    r = httpx.post(f"{base_url.rstrip('/')}/auth/login",
                   json={"email": email, "password": password}, timeout=30)
    r.raise_for_status()
    return r.json()["token"]


def run_live_benchmark(base_url, data_dir, threshold, token=None, email=None, password=None):
    """Score a labelled folder against a DEPLOYED OpenSeek endpoint over HTTP.

    Hits {base_url}/detect-image with each image so the numbers reflect exactly
    what real extension users get from the live server (model + threshold + infra).
    Auth: pass --token, or --email/--password to log in automatically.
    """
    import httpx

    base_url = base_url.rstrip("/")
    if not token and email and password:
        print(f"Logging in to {base_url} as {email} ...")
        token = _login(base_url, email, password)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        h = httpx.get(f"{base_url}/health", timeout=20).json()
        print(f"Live /health: models_loaded={h.get('models_loaded')} "
              f"hybrid_mode={h.get('hybrid_mode_active')} db={h.get('database')}")
    except Exception as e:
        print(f"(could not read /health: {e})")

    real_imgs = _list_images(os.path.join(data_dir, "real"))
    fake_imgs = _list_images(os.path.join(data_dir, "fake"))
    samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
    if not samples:
        print(f"No images under {data_dir}/real or {data_dir}/fake")
        return
    print(f"Scoring {len(samples)} images via {base_url}/detect-image "
          f"({len(real_imgs)} real, {len(fake_imgs)} fake)...\n")

    y_true, y_prob = [], []
    with httpx.Client(timeout=120) as client:
        for path, label in samples:
            try:
                with open(path, "rb") as fh:
                    files = {"file": (os.path.basename(path), fh, "image/jpeg")}
                    resp = client.post(f"{base_url}/detect-image", files=files, headers=headers)
                if resp.status_code != 200:
                    print(f"   skip {os.path.basename(path)}: HTTP {resp.status_code} {resp.text[:80]}")
                    continue
                prob = float(resp.json().get("ai_probability", 0.0))
            except Exception as e:
                print(f"   skip {os.path.basename(path)}: {e}")
                continue
            y_true.append(label)
            y_prob.append(prob)
            mark = "ok " if (prob > threshold) == bool(label) else "XX "
            print(f"   {mark} truth={'AI ' if label else 'REAL'} p={prob:.3f}  {os.path.basename(path)}")

    _report_metrics(y_true, y_prob, threshold)


def main():
    ap = argparse.ArgumentParser(description="OpenSeek benchmark / label verifier")
    ap.add_argument("--verify-labels", metavar="IMAGE",
                    help="Print raw HF label scores for one image and exit.")
    ap.add_argument("--data_dir", metavar="DIR",
                    help="Dataset folder with real/ and fake/ subfolders.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="ai_probability cutoff for the AI decision (default 0.5).")
    ap.add_argument("--live-url", metavar="URL",
                    help="Benchmark a DEPLOYED endpoint over HTTP instead of loading "
                         "models locally (needs --data_dir + auth). e.g. "
                         "https://openseek-production.up.railway.app")
    ap.add_argument("--token", help="Session bearer token for --live-url.")
    ap.add_argument("--email", help="Email to auto-login for --live-url.")
    ap.add_argument("--password", help="Password to auto-login for --live-url.")
    args = ap.parse_args()

    if args.verify_labels:
        verify_labels(args.verify_labels)
    elif args.live_url:
        if not args.data_dir:
            ap.error("--live-url requires --data_dir (folder with real/ and fake/)")
        run_live_benchmark(args.live_url, args.data_dir, args.threshold,
                           token=args.token, email=args.email, password=args.password)
    elif args.data_dir:
        run_benchmark(args.data_dir, args.threshold)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
