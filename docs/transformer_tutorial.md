# דאטהספר: Transformer Encoder–Decoder מאפס עד דוגמה עובדת

המסמך הזה בנוי כמו דאטהספר: **בלוקים של הסברים**, **בלוקים של הרצה**, ו**דאטה לדוגמה**. המטרה היא להדגים Transformer מלא עם Encoder + Decoder, כולל כל השלבים ביניהם.

---

## 1) מבנה הפרויקט

**הסבר:**
הפרויקט מחולק לרכיבים ברורים: מודל, דאטה, ואימון.

```
Transformer/
├── src/
│   ├── transformer.py  # המודל
│   ├── dataset.py      # הדאטה
│   └── train.py        # אימון והרצה
├── docs/
│   └── transformer_tutorial.md
├── data/
└── requirements.txt
```

---

## 2) הדאטה – רצפים והיפוך (Reverse)

**הסבר:**
ניצור דאטה מלאכותי של רצפים מספריים. ה־Encoder יקבל רצף, ה־Decoder ילמד להפיק את אותו הרצף במהופך. זה מאפשר לנו לבדוק שהמערכת עובדת בלי תלות בדאטה חיצוני.

**בלוק דאטה לדוגמה (קונספטואלי):**
```
SRC: [5, 9, 3, 4]  ->  TGT: [4, 3, 9, 5]
SRC: [2, 1, 6]     ->  TGT: [6, 1, 2]
```

**בלוק קוד (מתוך `src/dataset.py`):**
```python
src, tgt = generate_pair(rng, vocab, min_len=3, max_len=6)
# src: רצף מספרים
# tgt: אותו רצף במהופך
```

---

## 3) ה־Encoder

**הסבר:**
ה־Encoder מקבל את הטוקנים המוטמעים (Embedding + Positional Encoding) ומייצר ייצוג סמנטי של הרצף. הרעיון המרכזי: self-attention שמאפשר לכל טוקן “להסתכל” על שאר הרצף.

**בלוק קוד (מתוך `src/transformer.py`):**
```python
self.transformer = nn.Transformer(
    d_model=config.embedding_dim,
    nhead=config.num_heads,
    num_encoder_layers=config.num_encoder_layers,
    num_decoder_layers=config.num_decoder_layers,
    dim_feedforward=config.feedforward_dim,
    dropout=config.dropout,
    batch_first=True,
)
```

---

## 4) ה־Decoder + Mask סיבתי

**הסבר:**
ה־Decoder מקבל את הפלטים הקודמים, ומשתמש ב־mask כדי לא “להציץ” קדימה. זה מונע leakage של מידע מהעתיד בזמן האימון.

**בלוק קוד (מתוך `src/transformer.py`):**
```python
def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, device=device) == 1, diagonal=1)
```

---

## 5) שלבי האימון

**הסבר:**
השלבים הקלאסיים:
1. יצירת batchים עם Padding.
2. חישוב maskים (padding + causal).
3. Forward pass על המודל.
4. Loss (CrossEntropy) + Backprop.

**בלוק קוד (מתוך `src/train.py`):**
```python
logits = model(
    src,
    tgt_input,
    src_key_padding_mask=src_padding_mask,
    tgt_key_padding_mask=tgt_padding_mask,
    memory_key_padding_mask=src_padding_mask,
    tgt_mask=tgt_mask,
)
loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
```

---

## 6) הרצה בפועל

**הסבר:**
הפקודה הבאה מאמנת מודל קטן ומדפיסה דוגמה של תוצאה (reverse):

**בלוק הרצה:**
```bash
python src/train.py --epochs 10 --batch-size 64 --steps 100
```

**חלופה מהירה עם Makefile:**
```bash
make train
```

**דוגמת פלט צפויה (דוגמה):**
```
Epoch 01 | Loss: 2.34
...
Input tokens: [10, 6, 7, 5]
Predicted reversed: [5, 7, 6, 10]
```

---

## 7) יצירת דאטה לדוגמה לקבצים

**הסבר:**
בסיום ההרצה ייווצר קובץ `data/sample_pairs.txt` עם דוגמאות דאטה:

**בלוק הרצה:**
```bash
cat data/sample_pairs.txt
```

**חלופה מהירה עם Makefile:**
```bash
make sample
```

**דוגמה:**
```
SRC: [5, 8, 6] -> TGT: [6, 8, 5]
SRC: [2, 1, 9, 3] -> TGT: [3, 9, 1, 2]
```

---

## 8) מודל בקרה לבלוק של 2000 טוקנים

**הסבר:**
לאחר אימון, אנחנו מייצרים בלוק של **2000 טוקנים בבת אחת** מתוך המודל. מיד אחר כך פועל **מודל בקרה** שמקבל את רצף הטוקנים, מאמת שהוא חוקי (בטווח אוצר המילים ובאורך 2000), ואם לא – מתקן לפי הצורך ומחזיר את הבלוק המתוקן.

**בלוק קוד (מתוך `src/transformer.py`):**
```python
class ControlModel(nn.Module):
    def validate_and_correct(self, token_ids: torch.Tensor) -> torch.Tensor:
        corrected = token_ids.clone()
        corrected = torch.where(
            (corrected >= 0) & (corrected < self.vocab_size),
            corrected,
            torch.full_like(corrected, self.pad_token_id),
        )
        # תיקון אורך ל-2000
```

**בלוק קוד (מתוך `src/train.py`):**
```python
corrected_block = generate_block_with_control(
    model,
    control_model,
    vocab,
    rng,
    device,
    block_length=2000,
)
```

**הקובץ שנוצר:**
```
data/generated_block.txt
```

---

## 9) סיכום

קיבלתם Transformer מלא עם Encoder–Decoder, כולל דאטה, אימון, ו־inference. המודל קטן במכוון, כך שאפשר להריץ מהר ובקלות.
