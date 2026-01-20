"""Train the toy Transformer on the reverse-string task."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .data import Seq2SeqDataset, collect_texts, collate_batch, generate_pairs
from .tokenizer import CharTokenizer
from .transformer import Transformer
from .utils import ensure_dir, get_device, set_seed, summarize_loss


def build_dataloader(config: Config, tokenizer: CharTokenizer) -> DataLoader:
    """Create a DataLoader for training examples."""
    examples = generate_pairs(
        size=config.train_size,
        alphabet=config.alphabet,
        min_len=config.min_len,
        max_len=config.max_len,
        seed=config.seed,
    )
    dataset = Seq2SeqDataset(examples, tokenizer)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_id),
    )


def build_tokenizer(config: Config) -> CharTokenizer:
    """Fit a tokenizer on synthetic data."""
    examples = generate_pairs(
        size=config.train_size,
        alphabet=config.alphabet,
        min_len=config.min_len,
        max_len=config.max_len,
        seed=config.seed,
    )
    tokenizer = CharTokenizer()
    tokenizer.fit(collect_texts(examples))
    return tokenizer


def train_epoch(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        decoder_input = tgt[:, :-1]
        decoder_target = tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(src, decoder_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * decoder_target.numel()
        total_tokens += decoder_target.numel()

    return {"loss": total_loss / max(total_tokens, 1)}


def main() -> None:
    """Entry point for training."""
    config = Config()
    set_seed(config.seed)
    device = get_device()

    tokenizer = build_tokenizer(config)
    dataloader = build_dataloader(config, tokenizer)

    model = Transformer(
        vocab_size=len(tokenizer.stoi),
        d_model=config.d_model,
        num_heads=config.n_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        pad_id=tokenizer.pad_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    print(f"Training on {device} with {sum(p.numel() for p in model.parameters()):,} params")

    for epoch in range(1, config.epochs + 1):
        metrics = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch:02d}: {summarize_loss(metrics)}")

    ensure_dir(config.artifacts_dir)
    tokenizer.save(config.tokenizer_path)
    torch.save({"model_state": model.state_dict(), "config": config.__dict__}, config.checkpoint_path)
    print(f"Saved artifacts to {config.artifacts_dir}")

    # Simple sanity check: overfit on a small batch
    model.train()
    batch = next(iter(dataloader))
    src, tgt = (tensor.to(device) for tensor in batch)
    decoder_input = tgt[:, :-1]
    decoder_target = tgt[:, 1:]
    for step in range(30):
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, decoder_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            print(f"Sanity step {step + 1:02d}: loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
