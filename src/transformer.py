"""Full Transformer encoder-decoder model."""
from __future__ import annotations

import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder
from .layers import create_causal_mask, create_padding_mask
from .positional_encoding import SinusoidalPositionalEncoding


class Transformer(nn.Module):
    """Transformer encoder-decoder for sequence-to-sequence tasks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.output_head = nn.Linear(d_model, vocab_size)

    def encode(self, src_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source tokens and return memory and source mask."""
        src_mask = create_padding_mask(src_tokens, self.pad_id)
        x = self.embedding(src_tokens) * (self.d_model**0.5)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        memory = self.encoder(x, src_mask)
        return memory, src_mask

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode target tokens using encoder memory."""
        tgt_mask = create_padding_mask(tgt_tokens, self.pad_id)
        causal_mask = create_causal_mask(tgt_tokens.size(1), device=tgt_tokens.device)
        combined_mask = tgt_mask | causal_mask
        x = self.embedding(tgt_tokens) * (self.d_model**0.5)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        return self.decoder(x, memory, combined_mask, src_mask)

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for the target tokens."""
        memory, src_mask = self.encode(src_tokens)
        decoder_output = self.decode(tgt_tokens, memory, src_mask)
        return self.output_head(decoder_output)
