"""Shape checks for the Transformer components."""
from __future__ import annotations

import torch

from src.transformer import Transformer


def test_transformer_shapes() -> None:
    """Ensure model outputs match expected shapes."""
    batch_size = 2
    src_len = 7
    tgt_len = 5
    vocab_size = 20
    pad_id = 0

    model = Transformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        dropout=0.1,
        pad_id=pad_id,
    )
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
    logits = model(src, tgt)

    assert logits.shape == (batch_size, tgt_len, vocab_size)
