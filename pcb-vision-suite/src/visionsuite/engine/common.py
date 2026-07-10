"""Shared training machinery used by all three task engines: device
selection, AMP, warmup+cosine LR schedule, checkpointing, early stopping,
and reproducibility seeding."""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warmup_cosine(optimizer, total_epochs: int, steps_per_epoch: int,
                  warmup_epochs: float = 1.0):
    total = max(1, total_epochs * steps_per_epoch)
    warm = max(1, int(warmup_epochs * steps_per_epoch))

    def fn(step):
        if step < warm:
            return step / warm
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class Checkpointer:
    """Saves ``last.pt`` every epoch and ``best.pt`` when the monitored
    metric improves; stores enough state to resume."""

    def __init__(self, ckpt_dir: str | Path, monitor: str, mode: str = "max"):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.monitor, self.mode = monitor, mode
        self.best = -float("inf") if mode == "max" else float("inf")

    def is_better(self, value: float) -> bool:
        return value > self.best if self.mode == "max" else value < self.best

    def step(self, value: float, payload: dict) -> bool:
        torch.save(payload, self.dir / "last.pt")
        if value is not None and self.is_better(value):
            self.best = value
            torch.save(payload, self.dir / "best.pt")
            return True
        return False


class EarlyStopper:
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.bad = 0

    def step(self, value: float) -> bool:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        return self.bad > self.patience
