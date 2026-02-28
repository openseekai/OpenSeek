"""
DeepShield Training — Training Utilities
==========================================
Provides:
- WarmupCosineScheduler   : 5-epoch linear warmup + cosine annealing
- CheckpointManager       : Saves best + last checkpoints, logs history
- EarlyStopping           : Patience-based stopping on validation AUC
- AMP context manager     : Mixed precision with safe CPU fallback
- gradient_clipping       : Applied after loss.backward()
"""
from __future__ import annotations

import os
import json
import time
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


# ── Warmup + Cosine Annealing ─────────────────────────────────────────────────

class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup for `warmup_epochs` then cosine annealing to `eta_min`.
    Drop-in replacement for CosineAnnealingLR with proper warmup.
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]


# ── Checkpoint Manager ────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages best-model and last-model checkpoints plus a JSON training log.
    
    Saves:
      checkpoints/{tag}/best_model.pt
      checkpoints/{tag}/last_model.pt
      checkpoints/{tag}/training_log.json
    
    Args:
        save_dir : Root directory for checkpoints.
        tag      : Subdirectory name (e.g. "image", "video", "audio", "fusion").
        monitor  : Metric to maximize for "best" checkpoint ("auc" recommended).
    """
    def __init__(self, save_dir: str, tag: str, monitor: str = "auc"):
        self.dir = os.path.join(save_dir, tag)
        os.makedirs(self.dir, exist_ok=True)
        self.best_path = os.path.join(self.dir, "best_model.pt")
        self.last_path = os.path.join(self.dir, "last_model.pt")
        self.log_path  = os.path.join(self.dir, "training_log.json")
        self.monitor = monitor
        self.best_score = -float("inf")
        self.history: list[dict] = []

        # Load existing log if resuming
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path) as f:
                    self.history = json.load(f)
                if self.history:
                    scores = [h.get(self.monitor, 0) for h in self.history]
                    self.best_score = max(scores)
                print(f"[Checkpoint] Resumed log ({len(self.history)} epochs). "
                      f"Best {self.monitor}: {self.best_score:.4f}")
            except Exception as e:
                print(f"[Checkpoint] Could not load existing log: {e}")

    def save(self, model: nn.Module, metrics_dict: dict) -> bool:
        """
        Save last checkpoint. If current score > best, also save best checkpoint.
        Returns True if this is a new best.
        """
        score = metrics_dict.get(self.monitor, 0.0)
        self.history.append({**metrics_dict, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

        # Always save last
        torch.save(model.state_dict(), self.last_path)

        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            torch.save(model.state_dict(), self.best_path)
            print(f"  ✅ New best {self.monitor}: {score:.4f} → {self.best_path}")

        # Save JSON log
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return is_best

    def load_best(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Load best checkpoint weights into model."""
        if os.path.exists(self.best_path):
            model.load_state_dict(torch.load(self.best_path, map_location=device))
            print(f"[Checkpoint] Loaded best weights from {self.best_path}")
        else:
            print(f"[Checkpoint] No best checkpoint found at {self.best_path}")
        return model


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when monitored metric doesn't improve for `patience` epochs.
    
    Args:
        patience  : Number of epochs to wait without improvement.
        min_delta : Minimum improvement to be considered significant.
        monitor   : Metric to track ("auc" recommended).
    """
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, monitor: str = "auc"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metrics_dict: dict) -> bool:
        """
        Call at end of each validation epoch.
        Returns True if training should stop.
        """
        score = metrics_dict.get(self.monitor, 0.0)
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                  f"(best {self.monitor}={self.best:.4f})")
            if self.counter >= self.patience:
                print(f"  [EarlyStopping] 🛑 Stopping training.")
                self.should_stop = True
        return self.should_stop


# ── AMP Context ───────────────────────────────────────────────────────────────

def get_amp_scaler(device: torch.device) -> Optional[torch.cuda.amp.GradScaler]:
    """Returns a GradScaler for GPU, None for CPU (AMP not supported on CPU)."""
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


def amp_context(device: torch.device):
    """Returns the correct autocast context for device type."""
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return torch.no_grad.__class__()  # Identity context for CPU


# ── Gradient Clipping Helper ──────────────────────────────────────────────────

def clip_and_step(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler],
    max_norm: float = 1.0,
):
    """
    Unscales gradients (if AMP), clips them, then steps optimizer.
    
    Args:
        optimizer  : The optimizer to step.
        model      : The model (for grad norm clipping).
        scaler     : GradScaler from get_amp_scaler(). None → plain step.
        max_norm   : Gradient clipping max norm.
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()


# ── Epoch Timer ───────────────────────────────────────────────────────────────

class EpochTimer:
    """Simple context manager for timing epochs."""
    def __init__(self, epoch: int, total: int, tag: str = ""):
        self.epoch = epoch
        self.total = total
        self.tag = tag

    def __enter__(self):
        self._t = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._t
        print(f"\n  [{self.tag}] Epoch {self.epoch}/{self.total} completed in {self.elapsed:.1f}s")
