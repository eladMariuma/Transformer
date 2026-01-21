import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from dataset import Batch, Vocab, build_batches, generate_pair
from transformer import Seq2SeqTransformer, TransformerConfig, generate_square_subsequent_mask


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(
    model: Seq2SeqTransformer,
    batches: list[Batch],
    optimizer: Adam,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in batches:
        src = batch.src.to(device)
        tgt_input = batch.tgt_input.to(device)
        tgt_output = batch.tgt_output.to(device)
        src_padding_mask = batch.src_padding_mask.to(device)
        tgt_padding_mask = batch.tgt_padding_mask.to(device)

        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)
        logits = model(
            src,
            tgt_input,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask,
        )

        loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(batches)


def greedy_decode(
    model: Seq2SeqTransformer,
    src_tokens: torch.Tensor,
    vocab: Vocab,
    max_len: int,
    device: torch.device,
) -> list[int]:
    model.eval()
    src_tokens = src_tokens.to(device)
    src_padding_mask = src_tokens == vocab.pad
    memory = model.transformer.encoder(
        model.positional_encoding(model.token_embedding(src_tokens)),
        src_key_padding_mask=src_padding_mask,
    )

    ys = torch.tensor([[vocab.bos]], device=device)
    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device)
        output = model.transformer.decoder(
            model.positional_encoding(model.token_embedding(ys)),
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        logits = model.output_layer(output[:, -1])
        next_token = int(torch.argmax(logits, dim=-1))
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == vocab.eos:
            break
    return ys.squeeze(0).tolist()


def save_sample_dataset(output_dir: Path, vocab: Vocab, seed: int) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for _ in range(5):
        src, tgt = generate_pair(rng, vocab, min_len=3, max_len=6)
        lines.append(f"SRC: {src} -> TGT: {tgt}")
    (output_dir / "sample_pairs.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toy encoder-decoder Transformer.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    vocab = Vocab()
    config = TransformerConfig(vocab_size=vocab.size)
    model = Seq2SeqTransformer(config).to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad)

    rng = np.random.default_rng(args.seed)
    batches = build_batches(
        rng,
        vocab,
        batch_size=args.batch_size,
        steps=args.steps,
        min_len=3,
        max_len=8,
    )

    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, batches, optimizer, loss_fn, device)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f}")

    sample_src, _ = generate_pair(rng, vocab, min_len=4, max_len=6)
    src_tensor = torch.tensor([sample_src])
    decoded_tokens = greedy_decode(model, src_tensor, vocab, max_len=10, device=device)
    decoded_numbers = vocab.decode_numbers(decoded_tokens)
    print(f"Input tokens: {sample_src}")
    print(f"Predicted reversed: {decoded_numbers}")

    save_sample_dataset(Path("data"), vocab, args.seed)


if __name__ == "__main__":
    main()
