"""Run greedy decoding on a few samples."""
from __future__ import annotations

from typing import List

import torch

from .config import Config
from .data import Seq2SeqExample, generate_pairs
from .tokenizer import CharTokenizer
from .transformer import Transformer
from .utils import get_device, set_seed


def greedy_decode(
    model: Transformer,
    src_tokens: torch.Tensor,
    tokenizer: CharTokenizer,
    max_len: int,
) -> List[int]:
    """Greedy decoding loop for a single example."""
    model.eval()
    device = src_tokens.device
    memory, src_mask = model.encode(src_tokens)

    decoded = [tokenizer.bos_id]
    for _ in range(max_len):
        tgt_tokens = torch.tensor([decoded], device=device)
        logits = model.decode(tgt_tokens, memory, src_mask)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        decoded.append(next_token)
        if next_token == tokenizer.eos_id:
            break
    return decoded


def main() -> None:
    """Load model artifacts and run inference."""
    config = Config()
    set_seed(config.seed)
    device = get_device()

    tokenizer = CharTokenizer.load(config.tokenizer_path)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)

    model = Transformer(
        vocab_size=len(tokenizer.stoi),
        d_model=config.d_model,
        num_heads=config.n_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        pad_id=tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    samples = generate_pairs(5, config.alphabet, config.min_len, config.max_len, seed=123)
    print("\nGreedy decoding samples:")
    for example in samples:
        src_tokens = torch.tensor(
            [tokenizer.encode(example.source, add_bos=True, add_eos=True)], device=device
        )
        decoded_ids = greedy_decode(model, src_tokens, tokenizer, config.max_decode_len)
        prediction = tokenizer.decode(decoded_ids)
        print(f"Input: {example.source} | Target: {example.target} | Pred: {prediction}")


if __name__ == "__main__":
    main()
