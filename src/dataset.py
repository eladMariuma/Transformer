from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class Vocab:
    pad: int = 0
    bos: int = 1
    eos: int = 2
    offset: int = 3
    max_token: int = 12

    @property
    def size(self) -> int:
        return self.max_token + 1

    def encode_numbers(self, numbers: Iterable[int]) -> list[int]:
        return [n + self.offset for n in numbers]

    def decode_numbers(self, tokens: Iterable[int]) -> list[int]:
        result = []
        for token in tokens:
            if token in (self.pad, self.bos, self.eos):
                continue
            result.append(token - self.offset)
        return result


@dataclass
class Batch:
    src: torch.Tensor
    tgt_input: torch.Tensor
    tgt_output: torch.Tensor
    src_padding_mask: torch.Tensor
    tgt_padding_mask: torch.Tensor


def generate_pair(rng: np.random.Generator, vocab: Vocab, min_len: int, max_len: int) -> tuple[list[int], list[int]]:
    length = rng.integers(min_len, max_len + 1)
    numbers = rng.integers(0, vocab.max_token - vocab.offset + 1, size=length)
    src = vocab.encode_numbers(numbers.tolist())
    tgt = vocab.encode_numbers(numbers[::-1].tolist())
    return src, tgt


def build_batch(
    pairs: list[tuple[list[int], list[int]]],
    vocab: Vocab,
) -> Batch:
    src_sequences = []
    tgt_input_sequences = []
    tgt_output_sequences = []

    for src, tgt in pairs:
        src_sequences.append(src)
        tgt_input_sequences.append([vocab.bos] + tgt)
        tgt_output_sequences.append(tgt + [vocab.eos])

    src_length = max(len(seq) for seq in src_sequences)
    tgt_length = max(len(seq) for seq in tgt_input_sequences)

    def pad_sequence(seq: list[int], length: int) -> list[int]:
        return seq + [vocab.pad] * (length - len(seq))

    src_tensor = torch.tensor([pad_sequence(seq, src_length) for seq in src_sequences])
    tgt_input_tensor = torch.tensor(
        [pad_sequence(seq, tgt_length) for seq in tgt_input_sequences]
    )
    tgt_output_tensor = torch.tensor(
        [pad_sequence(seq, tgt_length) for seq in tgt_output_sequences]
    )

    src_padding_mask = src_tensor == vocab.pad
    tgt_padding_mask = tgt_input_tensor == vocab.pad

    return Batch(
        src=src_tensor,
        tgt_input=tgt_input_tensor,
        tgt_output=tgt_output_tensor,
        src_padding_mask=src_padding_mask,
        tgt_padding_mask=tgt_padding_mask,
    )


def build_batches(
    rng: np.random.Generator,
    vocab: Vocab,
    batch_size: int,
    steps: int,
    min_len: int,
    max_len: int,
) -> list[Batch]:
    batches = []
    for _ in range(steps):
        pairs = [generate_pair(rng, vocab, min_len, max_len) for _ in range(batch_size)]
        batches.append(build_batch(pairs, vocab))
    return batches
