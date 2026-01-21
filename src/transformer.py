import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int
    embedding_dim: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1
    max_length: int = 64
    pad_token_id: int = 0


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float, max_length: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(1, max_length, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        length = token_embeddings.size(1)
        token_embeddings = token_embeddings + self.pe[:, :length]
        return self.dropout(token_embeddings)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id
        )
        self.positional_encoding = PositionalEncoding(
            config.embedding_dim, config.dropout, config.max_length
        )
        self.transformer = nn.Transformer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src_embeddings = self.positional_encoding(self.token_embedding(src_tokens))
        tgt_embeddings = self.positional_encoding(self.token_embedding(tgt_tokens))

        transformer_output = self.transformer(
            src=src_embeddings,
            tgt=tgt_embeddings,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        return self.output_layer(transformer_output)


def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, device=device) == 1, diagonal=1)
