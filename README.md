# Transformer לדאטהספר (Encoder–Decoder)

הרפוזיטורי הזה מכיל דוגמה מלאה ל־Transformer עם **Encoder + Decoder**, כולל הסבר תיאורטי, דאטה מלאכותי, אימון, והרצה מקצה לקצה. הכל כתוב כך שאפשר לשכפל ולהריץ מיד.

## מה יש כאן

- `src/transformer.py` – מודל Transformer עם positional encoding, encoder, decoder, ושכבת פלט.
- `src/dataset.py` – יצירת דאטה מלאכותי (רצפים של מספרים) והכנת batchים.
- `src/train.py` – אימון, הערכת תוצאה, והדפסה של דוגמה.
- `docs/transformer_tutorial.md` – “דאטהספר”: בלוקים של הסברים + בלוקים של הרצה עם דאטה.

## התקנה והרצה

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

הרצה:

```bash
python src/train.py --epochs 10 --batch-size 64 --steps 100
```

בסיום, התסריט גם ייצור קבצים:

- `data/sample_pairs.txt` עם דוגמאות דאטה.
- `data/generated_block.txt` עם בלוק של 2000 טוקנים שעבר אימות ותיקון על ידי מודל הבקרה.

## הסבר קצר על הדאטה

המודל מקבל רצף מספרים ומחזיר את אותו הרצף במהופך (reverse). זו משימה קלה שמדגימה את כל החלקים של Encoder–Decoder בלי להיות תלויים בדאטה חיצוני.

למידע מלא ומודרך – פתחו את `docs/transformer_tutorial.md`.
