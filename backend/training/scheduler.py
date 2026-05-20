"""
OpenSeek Training — Scheduler Module
========================================
Provides multiple LR scheduler options for production training.

Available:
  WarmupCosineScheduler  : Linear warmup (5ep) → CosineAnnealing (recommended)
  get_onecycle_scheduler : OneCycleLR for aggressive 1-phase training
  get_plateau_scheduler  : ReduceLROnPlateau (metric-driven)
  get_scheduler          : Factory function — pick one by name

Usage:
    from training.scheduler import get_scheduler, WarmupCosineScheduler

    # As factory (recommended — works with train_all.py config):
    sched = get_scheduler("warmup_cosine", optimizer, epochs=40, warmup=5)

    # Or direct:
    sched = WarmupCosineScheduler(optimizer, warmup_epochs=5, max_epochs=40)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    SequentialLR,
    LinearLR,
)


# ── Warmup + Cosine Annealing ─────────────────────────────────────────────────

class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup for `warmup_epochs` then cosine annealing to `eta_min`.
    Prevents training instability from large LR at initialization.

    Recommended settings:
        warmup_epochs = 5 (or 10% of total epochs)
        eta_min       = base_lr * 0.01

    Args:
        optimizer     : The optimizer.
        warmup_epochs : Number of linear warmup epochs.
        max_epochs    : Total training epochs.
        eta_min       : Final minimum LR (default: 1e-6).
        last_epoch    : Resume epoch (-1 = start fresh).
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = max(1, warmup_epochs)
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
            return [base_lr * scale for base_lr in self.base_lrs]
        progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]


# ── Cosine Annealing with Warm Restarts ───────────────────────────────────────

class WarmupCosineRestartScheduler(_LRScheduler):
    """
    Warmup followed by cosine annealing with warm restarts (SGDR).
    Useful for longer training runs where cyclical LR helps escape local minima.

    Args:
        T_0          : Epochs per restart cycle (first cycle).
        T_mult       : Cycle length multiplier after each restart.
        eta_min      : Minimum LR.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = max(1, warmup_epochs)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._cycle_epoch = 0
        self._T_cur = T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
            return [base_lr * scale for base_lr in self.base_lrs]

        cycle_pos = (epoch - self.warmup_epochs) % self._T_cur
        progress = cycle_pos / max(self._T_cur, 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Advance cycle
        if epoch > self.warmup_epochs and cycle_pos == 0:
            self._T_cur = max(self._T_cur * self.T_mult, 1)

        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]


# ── Metric-Driven Scheduler ───────────────────────────────────────────────────

def get_plateau_scheduler(
    optimizer: Optimizer,
    patience: int = 3,
    factor: float = 0.5,
    min_lr: float = 1e-7,
    mode: str = "max",   # "max" for AUC, "min" for loss
) -> ReduceLROnPlateau:
    """
    Reduces LR when a monitored metric has stopped improving.
    Call: scheduler.step(val_auc) — updates LR based on AUC plateau.
    """
    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        verbose=True,
    )


# ── OneCycle LR ───────────────────────────────────────────────────────────────

def get_onecycle_scheduler(
    optimizer: Optimizer,
    max_lr: float,
    steps_per_epoch: int,
    epochs: int,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> OneCycleLR:
    """
    OneCycleLR: ramps LR up then anneals. Often converges faster than cosine.
    Good for: shorter training runs (20-30 epochs).
    Call scheduler.step() after EACH BATCH (not each epoch).

    Args:
        max_lr           : Peak learning rate (usually 3-10× base LR).
        steps_per_epoch  : Number of batches per epoch (len(train_loader)).
        pct_start        : Fraction of training for warmup phase (0.3 recommended).
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        anneal_strategy="cos",
    )


# ── Factory Function ──────────────────────────────────────────────────────────

SCHEDULER_NAMES = ["warmup_cosine", "warmup_cosine_restart", "plateau", "onecycle", "cosine"]

def get_scheduler(
    name: str,
    optimizer: Optimizer,
    epochs: int = 40,
    warmup: int = 5,
    eta_min: float = 1e-6,
    # OneCycle specific
    max_lr: Optional[float] = None,
    steps_per_epoch: Optional[int] = None,
    # Plateau specific
    patience: int = 3,
    **kwargs,
):
    """
    Factory: create a scheduler by name.

    Args:
        name            : One of: warmup_cosine, warmup_cosine_restart, plateau, onecycle, cosine
        optimizer       : Model optimizer.
        epochs          : Total training epochs.
        warmup          : Warmup epochs (warmup_cosine* only).
        eta_min         : Minimum LR (warmup_cosine* only).
        max_lr          : Peak LR (onecycle only).
        steps_per_epoch : Batches per epoch (onecycle only).
        patience        : Plateau patience (plateau only).

    Example:
        sched = get_scheduler("warmup_cosine", optimizer, epochs=40, warmup=5)
        # In training loop:
        sched.step()  # after each epoch
    """
    name = name.lower().strip()
    if name not in SCHEDULER_NAMES:
        raise ValueError(f"Unknown scheduler '{name}'. Choose from: {SCHEDULER_NAMES}")

    base_lr = optimizer.param_groups[0]["lr"]

    if name == "warmup_cosine":
        return WarmupCosineScheduler(optimizer, warmup_epochs=warmup, max_epochs=epochs, eta_min=eta_min)
    elif name == "warmup_cosine_restart":
        return WarmupCosineRestartScheduler(optimizer, warmup_epochs=warmup, T_0=10, T_mult=2, eta_min=eta_min)
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    elif name == "plateau":
        return get_plateau_scheduler(optimizer, patience=patience)
    elif name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for OneCycleLR")
        peak = max_lr or base_lr * 10
        return get_onecycle_scheduler(optimizer, max_lr=peak, steps_per_epoch=steps_per_epoch, epochs=epochs)
    else:
        raise ValueError(f"Scheduler '{name}' not implemented")


def is_epoch_scheduler(scheduler) -> bool:
    """Returns True if scheduler.step() should be called per-epoch (not per-batch)."""
    return not isinstance(scheduler, OneCycleLR)
