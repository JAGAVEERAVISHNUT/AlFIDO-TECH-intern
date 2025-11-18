# Task 3 — Sentiment Analysis

This folder includes `task 3.py`, a script to perform sentiment analysis using either NLTK (VADER) or spaCy + SpacyTextBlob.

Setup (PowerShell):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
# Download the spaCy English model if you plan to use spaCy method
python -m spacy download en_core_web_sm
```

Quick examples:

Analyze sample sentences:

```powershell
python "task 3.py" --sample --method nltk
python "task 3.py" --sample --method spacy --model en_core_web_sm
```

Analyze a CSV file (assumes a `text` column by default):

```powershell
python "task 3.py" --input dataset1.csv --text-column text --method nltk --output results.csv
```

Outputs
- If `--output` provided, results are written there.
- Otherwise results are saved to `sentiment_output.csv`.

Notes
- NLTK VADER works best on short, social-media-style English text.
- spaCy + SpacyTextBlob provides polarity/subjectivity scores.
