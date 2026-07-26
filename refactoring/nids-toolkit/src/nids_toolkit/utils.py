"""Reproducibility, device, and logging helpers."""

from __future__ import annotations

import datetime
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed every RNG that affects a run.

    Re-invoke before each experiment config so every (arch, source) run starts
    from an identical RNG state instead of inheriting whatever the previous run
    consumed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int = 42) -> torch.Generator:
    """A dedicated seeded ``torch.Generator`` for shuffling DataLoaders.

    DataLoader shuffles are otherwise driven by the global RNG; giving each
    shuffling loader its own seeded generator makes shuffling reproducible
    independent of what ran before.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device() -> str:
    """Return ``'cuda'`` when available, else ``'cpu'``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def log_step(step_num: int, step_name: str) -> None:
    """Timestamped, colourised step banner (matches the notebook logging)."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\033[36m[{timestamp}]\033[96m [Step {step_num}]\033[0m {step_name}")
