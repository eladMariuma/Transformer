"""Character-level tokenizer with special tokens."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class TokenizerConfig:
    """Configuration for special tokens."""

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"


class CharTokenizer:
    """Simple character-level tokenizer.

    This tokenizer builds a vocabulary from an iterable of strings and provides
    encode/decode utilities with special tokens.
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    @property
    def pad_id(self) -> int:
        """Return the pad token id."""
        return self.stoi[self.config.pad_token]

    @property
    def bos_id(self) -> int:
        """Return the beginning-of-sequence token id."""
        return self.stoi[self.config.bos_token]

    @property
    def eos_id(self) -> int:
        """Return the end-of-sequence token id."""
        return self.stoi[self.config.eos_token]

    def fit(self, texts: Iterable[str]) -> None:
        """Build vocabulary from the provided texts."""
        unique_chars = sorted(set("".join(texts)))
        vocab = [
            self.config.pad_token,
            self.config.bos_token,
            self.config.eos_token,
            *unique_chars,
        ]
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert text to a list of token ids."""
        if not self.stoi:
            raise ValueError("Tokenizer vocabulary is empty. Call fit() first.")
        tokens = [self.stoi[char] for char in text]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Convert token ids back to a string."""
        tokens = []
        for idx in ids:
            token = self.itos[int(idx)]
            if skip_special and token in {
                self.config.pad_token,
                self.config.bos_token,
                self.config.eos_token,
            }:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path: str) -> None:
        """Save tokenizer vocabulary to disk."""
        payload = {
            "config": self.config.__dict__,
            "stoi": self.stoi,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load tokenizer vocabulary from disk."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = TokenizerConfig(**payload["config"])
        tokenizer = cls(config)
        tokenizer.stoi = {k: int(v) for k, v in payload["stoi"].items()}
        tokenizer.itos = {v: k for k, v in tokenizer.stoi.items()}
        return tokenizer
