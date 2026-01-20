"""Utility helpers for training and inference."""
from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def summarize_loss(history: Dict[str, float]) -> str:
    """Format loss metrics for printing."""
    parts = [f"{name}: {value:.4f}" for name, value in history.items()]
    return " | ".join(parts)
