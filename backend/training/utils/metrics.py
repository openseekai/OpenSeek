"""
DeepShield Training — Evaluation Metrics
==========================================
Tracks: AUC, F1, Precision, Recall, Confusion Matrix, ROC Curve.
All metrics are computed from accumulated predictions across an epoch.
"""
from __future__ import annotations

import os
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        roc_curve,
        average_precision_score,
    )
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[Metrics] Warning: scikit-learn not installed. AUC/F1 unavailable.")


@dataclass
class EpochMetrics:
    """Container for all metrics computed over one validation epoch."""
    epoch: int
    split: str               # "train" or "val"
    loss: float = 0.0
    accuracy: float = 0.0
    auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    all_probs: List[float] = field(default_factory=list, repr=False)
    all_labels: List[int] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        return (
            f"[{self.split.upper()} Ep {self.epoch:02d}] "
            f"Loss: {self.loss:.4f} | Acc: {self.accuracy*100:.2f}% | "
            f"AUC: {self.auc:.4f} | F1: {self.f1:.4f} | "
            f"Prec: {self.precision:.4f} | Rec: {self.recall:.4f}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("all_probs", None)
        d.pop("all_labels", None)
        return d


class MetricsTracker:
    """
    Accumulates per-batch predictions and computes full epoch metrics.
    
    Usage:
        tracker = MetricsTracker()
        # Inside batch loop:
        tracker.update(probs, labels)
        # At epoch end:
        metrics = tracker.compute(epoch=1, split="val", loss=0.32)
        print(metrics.summary())
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._probs: List[float] = []
        self._labels: List[int] = []

    def update(self, probs: np.ndarray, labels: np.ndarray):
        """
        probs  : numpy array of predicted probabilities, shape (N,)
        labels : numpy array of ground truth 0/1, shape (N,)
        """
        self._probs.extend(probs.tolist() if hasattr(probs, "tolist") else list(probs))
        self._labels.extend([int(l) for l in labels])

    def compute(self, epoch: int, split: str, loss: float) -> EpochMetrics:
        probs = np.array(self._probs)
        labels = np.array(self._labels, dtype=int)
        preds = (probs >= 0.5).astype(int)

        correct = (preds == labels).sum()
        accuracy = correct / max(len(labels), 1)

        m = EpochMetrics(
            epoch=epoch,
            split=split,
            loss=loss,
            accuracy=accuracy,
            all_probs=self._probs,
            all_labels=list(self._labels),
        )

        if _SKLEARN_OK and len(np.unique(labels)) > 1:
            try:
                m.auc       = float(roc_auc_score(labels, probs))
                m.f1        = float(f1_score(labels, preds, zero_division=0))
                m.precision = float(precision_score(labels, preds, zero_division=0))
                m.recall    = float(recall_score(labels, preds, zero_division=0))
            except Exception as e:
                print(f"[Metrics] sklearn error: {e}")

        cm = confusion_matrix(labels, preds, labels=[0, 1]) if _SKLEARN_OK else np.zeros((2, 2))
        if cm.shape == (2, 2):
            m.tn, m.fp, m.fn, m.tp = cm.ravel().tolist()

        self.reset()
        return m


def save_metrics_log(metrics_list: List[EpochMetrics], save_dir: str, filename: str = "metrics_log.json"):
    """Save all epoch metrics to a JSON file for later analysis."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    data = [m.to_dict() for m in metrics_list]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Metrics] Log saved → {path}")


def plot_roc_curve(
    metrics: EpochMetrics,
    save_dir: str,
    tag: str = "model",
):
    """Save ROC curve plot as PNG (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not _SKLEARN_OK or len(np.unique(metrics.all_labels)) < 2:
            return

        fpr, tpr, _ = roc_curve(metrics.all_labels, metrics.all_probs)
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {metrics.auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {tag} (Epoch {metrics.epoch})")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(save_dir, f"roc_{tag}_ep{metrics.epoch:03d}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"[Metrics] ROC curve saved → {path}")
    except Exception as e:
        print(f"[Metrics] ROC plot failed: {e}")


def print_confusion_matrix(metrics: EpochMetrics):
    print(f"\n  Confusion Matrix (Ep {metrics.epoch}):")
    print(f"  {'':10s}  Pred Real  Pred Fake")
    print(f"  {'True Real':10s}     {metrics.tn:5d}     {metrics.fp:5d}")
    print(f"  {'True Fake':10s}     {metrics.fn:5d}     {metrics.tp:5d}")
    print()
