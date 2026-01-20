"""Synthetic dataset utilities for the toy seq2seq task."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


@dataclass
class Seq2SeqExample:
    """Single input-output pair."""

    source: str
    target: str


def generate_pairs(
    size: int,
    alphabet: str,
    min_len: int,
    max_len: int,
    seed: int = 0,
) -> List[Seq2SeqExample]:
    """Generate random strings and their reversed targets."""
    rng = random.Random(seed)
    examples: List[Seq2SeqExample] = []
    for _ in range(size):
        length = rng.randint(min_len, max_len)
        source = "".join(rng.choice(alphabet) for _ in range(length))
        target = source[::-1]
        examples.append(Seq2SeqExample(source=source, target=target))
    return examples


class Seq2SeqDataset(Dataset):
    """Dataset that stores tokenized sequence pairs."""

    def __init__(self, examples: Sequence[Seq2SeqExample], tokenizer: CharTokenizer) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.encoded_pairs = [
            (
                tokenizer.encode(example.source, add_bos=True, add_eos=True),
                tokenizer.encode(example.target, add_bos=True, add_eos=True),
            )
            for example in examples
        ]

    def __len__(self) -> int:
        return len(self.encoded_pairs)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.encoded_pairs[idx]


def collate_batch(
    batch: Iterable[Tuple[List[int], List[int]]], pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad and stack a batch of tokenized examples."""
    src_batch, tgt_batch = zip(*batch)
    src_max = max(len(seq) for seq in src_batch)
    tgt_max = max(len(seq) for seq in tgt_batch)

    def pad_sequence(seq: List[int], max_len: int) -> List[int]:
        return seq + [pad_id] * (max_len - len(seq))

    src_tensor = torch.tensor([pad_sequence(seq, src_max) for seq in src_batch])
    tgt_tensor = torch.tensor([pad_sequence(seq, tgt_max) for seq in tgt_batch])
    return src_tensor, tgt_tensor


def collect_texts(examples: Sequence[Seq2SeqExample]) -> List[str]:
    """Collect all texts for tokenizer fitting."""
    texts: List[str] = []
    for example in examples:
        texts.append(example.source)
        texts.append(example.target)
    return texts
