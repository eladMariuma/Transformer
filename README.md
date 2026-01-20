# Transformer Toy Project (Encoder–Decoder)

A compact, educational Transformer (Encoder–Decoder) implementation in PyTorch. It includes a
character-level tokenizer, sinusoidal positional encodings, multi-head attention, training on a
synthetic reverse-string task, and greedy decoding inference.

## Requirements

- Python 3.10+ (recommended 3.11)
- PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Train the model:

```bash
python -m src.train
```

Run inference:

```bash
python -m src.infer
```

## Project Structure

```
requirements.txt
README.md
src/
  config.py
  tokenizer.py
  positional_encoding.py
  attention.py
  layers.py
  encoder.py
  decoder.py
  transformer.py
  data.py
  train.py
  infer.py
  utils.py
docs/
  transformer_explained.md
tests/
  test_shapes.py
```

## What You Get

- Character-level tokenizer with `<pad>`, `<bos>`, `<eos>` tokens.
- Encoder–Decoder Transformer with multi-head attention and masking.
- End-to-end demo: synthetic data, short training, greedy decoding.
- Hebrew documentation explaining each block and the data flow.

## Example Inference Output

```text
Greedy decoding samples:
Input: abcd | Target: dcba | Pred: dcba
```

(Your output will vary slightly because of random initialization.)
