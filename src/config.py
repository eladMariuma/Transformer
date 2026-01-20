"""Configuration for the toy Transformer project."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters and settings for training and inference."""

    # Data
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    min_len: int = 3
    max_len: int = 12
    train_size: int = 512
    batch_size: int = 32

    # Model
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1

    # Training
    epochs: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42

    # Inference
    max_decode_len: int = 20

    # Paths
    artifacts_dir: str = "artifacts"
    tokenizer_path: str = "artifacts/tokenizer.json"
    checkpoint_path: str = "artifacts/transformer.pt"
