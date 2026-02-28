"""
DeepShield Training — Standalone Metrics Module
=================================================
Top-level alias for training.utils.metrics with additional
TensorBoard integration helpers.

Tracks: AUC-ROC, F1, Precision, Recall, Confusion Matrix,
        False Positive Rate, False Negative Rate.

Usage:
    from training.metrics import MetricsTracker, log_to_tensorboard

    tracker = MetricsTracker()
    tracker.update(probs, labels)
    m = tracker.compute(epoch=1, split="val", loss=0.35)
    print(m.summary())

    # TensorBoard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter("runs/image_exp")
    log_to_tensorboard(writer, m, tag="Image")
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional, List

# Re-export everything from utils.metrics
from training.utils.metrics import (
    EpochMetrics,
    MetricsTracker,
    save_metrics_log,
    plot_roc_curve,
    print_confusion_matrix,
)

__all__ = [
    "EpochMetrics",
    "MetricsTracker",
    "save_metrics_log",
    "plot_roc_curve",
    "print_confusion_matrix",
    "log_to_tensorboard",
    "compute_cross_dataset_metrics",
    "MetricAggregator",
]


# ── TensorBoard Logging ───────────────────────────────────────────────────────

def log_to_tensorboard(
    writer,
    metrics: EpochMetrics,
    tag: str = "Model",
    log_images: bool = False,
):
    """
    Log all metrics to TensorBoard SummaryWriter.
    
    Args:
        writer    : torch.utils.tensorboard.SummaryWriter instance.
        metrics   : EpochMetrics from MetricsTracker.compute().
        tag       : Prefix for metric names (e.g. "Image", "Video").
        log_images: Whether to log ROC curve as image (requires matplotlib).
    
    Usage:
        writer = SummaryWriter("runs/image_training")
        log_to_tensorboard(writer, val_metrics, tag="Image")
    """
    if writer is None:
        return

    step = metrics.epoch
    prefix = f"{tag}/{metrics.split}"

    writer.add_scalar(f"{prefix}/loss",      metrics.loss,      step)
    writer.add_scalar(f"{prefix}/accuracy",  metrics.accuracy,  step)
    writer.add_scalar(f"{prefix}/auc",       metrics.auc,       step)
    writer.add_scalar(f"{prefix}/f1",        metrics.f1,        step)
    writer.add_scalar(f"{prefix}/precision", metrics.precision, step)
    writer.add_scalar(f"{prefix}/recall",    metrics.recall,    step)

    # Derived metrics
    tp, tn, fp, fn = metrics.tp, metrics.tn, metrics.fp, metrics.fn
    total = max(tp + tn + fp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    writer.add_scalar(f"{prefix}/fpr", fpr, step)
    writer.add_scalar(f"{prefix}/fnr", fnr, step)

    if log_images and len(metrics.all_probs) > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve as sk_roc_curve

            fpr_arr, tpr_arr, _ = sk_roc_curve(metrics.all_labels, metrics.all_probs)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr_arr, tpr_arr, lw=2, label=f"AUC={metrics.auc:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"{tag} ROC (Epoch {step})")
            ax.legend()
            writer.add_figure(f"{prefix}/roc_curve", fig, step)
            plt.close(fig)
        except Exception as e:
            pass  # Don't fail training on plot errors

    # Confusion matrix as text table
    writer.add_text(
        f"{prefix}/confusion_matrix",
        f"TP={tp} TF={tn} FP={fp} FN={fn} (FPR={fpr:.4f} FNR={fnr:.4f})",
        step,
    )


def get_writer(log_dir: str, tag: str) -> Optional[object]:
    """
    Create a TensorBoard SummaryWriter. Returns None if TensorBoard unavailable.
    
    Usage:
        writer = get_writer("./runs", "image_training")
        # → writes to ./runs/image_training/
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        run_dir = os.path.join(log_dir, tag)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(run_dir)
        print(f"  [TensorBoard] Logging to {run_dir}")
        print(f"  [TensorBoard] View with: tensorboard --logdir {log_dir}")
        return writer
    except ImportError:
        print("  [TensorBoard] Not installed. Install with: pip install tensorboard")
        print("  [TensorBoard] Continuing without TensorBoard logging.")
        return None


# ── Cross-Dataset Evaluation ─────────────────────────────────────────────────

def compute_cross_dataset_metrics(
    model,
    test_loaders: dict,
    device,
    criterion=None,
) -> dict:
    """
    Evaluate model on multiple test datasets and return per-dataset AUC.
    
    Args:
        model       : Trained PyTorch model.
        test_loaders: Dict of {dataset_name: DataLoader}
                      e.g. {"dfdc": dfdc_loader, "celebdf": celebdf_loader}
        device      : torch.device
        criterion   : Loss function (optional, for loss tracking).
    
    Returns:
        Dict of {dataset_name: EpochMetrics}
    
    Example:
        results = compute_cross_dataset_metrics(model, {
            "DFDC": dfdc_test_loader,
            "Celeb-DF": celebdf_test_loader,
        }, device=device)
        for name, m in results.items():
            print(f"{name}: AUC={m.auc:.4f}")
    """
    import torch
    model.eval()
    results = {}

    for ds_name, loader in test_loaders.items():
        tracker = MetricsTracker()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                logits = model(inputs)
                if criterion:
                    loss = criterion(logits.view(-1, 1), labels.view(-1, 1))
                    total_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()
                lbl   = labels.cpu().numpy().squeeze()
                tracker.update(
                    np.atleast_1d(probs) if probs.ndim == 0 else probs,
                    np.atleast_1d(lbl)   if lbl.ndim == 0   else lbl,
                )

        avg_loss = total_loss / max(len(loader.dataset), 1)
        m = tracker.compute(epoch=0, split=ds_name, loss=avg_loss)
        results[ds_name] = m
        print(f"  [CrossDataset] {ds_name}: AUC={m.auc:.4f} F1={m.f1:.4f} "
              f"Prec={m.precision:.4f} Rec={m.recall:.4f}")

    return results


# ── K-Fold Metric Aggregator ──────────────────────────────────────────────────

class MetricAggregator:
    """
    Aggregates metrics across multiple folds for cross-validation reporting.
    
    Usage:
        agg = MetricAggregator()
        for fold, ... in enumerate(folds):
            m = tracker.compute(...)
            agg.add(m)
        agg.report()
    """
    def __init__(self):
        self._metrics: List[EpochMetrics] = []

    def add(self, m: EpochMetrics):
        self._metrics.append(m)

    def report(self) -> dict:
        if not self._metrics:
            print("  [MetricAgg] No metrics to aggregate.")
            return {}

        aucs = [m.auc for m in self._metrics]
        f1s  = [m.f1  for m in self._metrics]
        accs = [m.accuracy for m in self._metrics]

        result = {
            "auc_mean":  np.mean(aucs),
            "auc_std":   np.std(aucs),
            "f1_mean":   np.mean(f1s),
            "f1_std":    np.std(f1s),
            "acc_mean":  np.mean(accs),
            "acc_std":   np.std(accs),
        }

        print(f"\n  ┌{'─'*50}┐")
        print(f"  │  Cross-Validation Results ({len(self._metrics)} folds)     │")
        print(f"  ├{'─'*50}┤")
        print(f"  │  AUC : {result['auc_mean']:.4f} ± {result['auc_std']:.4f}              │")
        print(f"  │  F1  : {result['f1_mean']:.4f} ± {result['f1_std']:.4f}              │")
        print(f"  │  Acc : {result['acc_mean']:.4f} ± {result['acc_std']:.4f}              │")
        print(f"  └{'─'*50}┘\n")

        return result
