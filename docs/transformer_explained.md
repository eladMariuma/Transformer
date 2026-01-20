# Transformer מוסבר בקצרה (Encoder–Decoder)

המסמך הזה מסביר את אבני-הבניין של Transformer בצורה לימודית ופשוטה, על בסיס הפרויקט שב־`src/`.

## Tokenization (Character-level)
במקום tokenizer מורכב, אנחנו מפרקים כל מחרוזת לאותיות בודדות. לכל אות יש מזהה מספרי (ID),
ומוסיפים גם שלושה טוקנים מיוחדים:
- `<pad>` לריפוד רצפים (padding)
- `<bos>` לתחילת רצף
- `<eos>` לסוף רצף

## Embeddings + Positional Encoding
Transformer לא מקבל טוקנים ישירות; הוא מקבל וקטורים (embeddings). לכן:
1. `Embedding` ממפה כל ID לוקטור בגודל `d_model`.
2. Positional Encoding (סינוס/קוסינוס) מוסיף מידע על מיקום הטוקן ברצף.

הנוסחה (כמו במאמר) משתמשת בסינוס וקוסינוס בתדרים שונים כדי לאפשר למודל להבין סדר.

## Q / K / V (Query, Key, Value)
Self-Attention לומד קשרים בין טוקנים. לכל טוקן יוצרים:
- **Query (Q)**: "מה אני מחפש?"
- **Key (K)**: "מה יש לי להציע?"
- **Value (V)**: "איזה מידע אני מספק?"

אנו מחשבים דמיון בין Q ל-K, ואז משקללים את ה־V.

## Scaled Dot-Product Attention
הציון בין טוקנים מחושב כך:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

החלוקה ב־`sqrt(d_k)` מאזנת גדלים ומייצבת אימון.

## Multi-Head Attention
במקום ראש יחיד, מפצלים למספר ראשים. כל ראש לומד יחסים שונים בין טוקנים.
בסוף מאחדים חזרה לוקטור אחד בגודל `d_model`.

## Masked Self-Attention (Decoder)
ב־Decoder אסור "לראות" טוקנים עתידיים. לכן מוסיפים **causal mask**
שממסך את החלק העליון של מטריצת הקשב.

## Cross-Attention (Encoder–Decoder)
אחרי self-attention ב־Decoder, מוסיפים שכבת קשב שמסתכלת על הפלט של ה־Encoder.
כך ה־Decoder יכול להתבסס על מידע מהקלט.

## Residual + LayerNorm (Add & Norm)
בכל תת-שכבה מוסיפים חיבור שארית (Residual) ולאחר מכן LayerNorm.
זה משפר יציבות ומקל על אימון רשת עמוקה.

## Feed-Forward Network (FFN)
רשת קטנה לכל טוקן באופן עצמאי:

```
Linear -> ReLU -> Dropout -> Linear
```

## Softmax / Logits
במהלך האימון נשתמש בלוגיטים + `CrossEntropyLoss`. באינפרנס נעשה `softmax`
כדי לבחור את הטוקן הבא (greedy decoding).

## Encoder–Decoder Flow (ASCII Diagram)

```
Input Tokens
     │
[Embedding + PosEnc]
     │
  Encoder Stack
     │
  Encoder Memory
     │
     ├─────────────────────────────┐
     │                             │
[Embedding + PosEnc]         (Cross-Attention)
     │                             │
Masked Self-Attention             │
     │                             │
  Decoder Stack ◀──────────────────┘
     │
  Linear Head (Logits)
     │
Output Tokens
```

## English Summary
This toy project demonstrates a full Transformer encoder–decoder with character-level
tokenization, sinusoidal positional encoding, multi-head attention, masking, and greedy decoding
on a reverse-string task. The code is intentionally small and heavily commented to support learning.
