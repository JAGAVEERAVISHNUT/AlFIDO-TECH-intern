"""
Task 3 - Sentiment Analysis (NLTK or spaCy)

This script performs sentiment analysis on text data using either NLTK's VADER
or spaCy combined with SpacyTextBlob. It can process a CSV file with a text
column or analyze ad-hoc sample texts.

Usage examples (PowerShell):
	python "task 3.py" --sample
	python "task 3.py" --input dataset1.csv --text-column text --method nltk --output results.csv
	python "task 3.py" --input dataset1.csv --text-column review --method spacy --model en_core_web_sm

Requirements:
	- See `requirements.txt` in the repo. After installing packages, run:
		python -m spacy download en_core_web_sm

The script will automatically download the VADER lexicon for NLTK if needed.
"""

from typing import List, Dict
import argparse
import sys
import pandas as pd


def analyze_nltk(texts: List[str]) -> List[Dict]:
	"""Analyze texts using NLTK VADER. Returns list of dicts with scores."""
	try:
		import nltk
		from nltk.sentiment.vader import SentimentIntensityAnalyzer
	except Exception as e:
		raise RuntimeError("NLTK is not installed. Please install dependencies.") from e

	# Ensure the VADER lexicon is available
	try:
		nltk.data.find('sentiment/vader_lexicon.zip')
	except LookupError:
		nltk.download('vader_lexicon', quiet=True)

	sia = SentimentIntensityAnalyzer()
	results = []
	for t in texts:
		t_str = '' if t is None else str(t)
		s = sia.polarity_scores(t_str)
		comp = s.get('compound', 0.0)
		if comp >= 0.05:
			label = 'positive'
		elif comp <= -0.05:
			label = 'negative'
		else:
			label = 'neutral'
		results.append({
			'text': t_str,
			'method': 'nltk_vader',
			'neg': s.get('neg'),
			'neu': s.get('neu'),
			'pos': s.get('pos'),
			'compound': comp,
			'label': label,
		})
	return results


def analyze_spacy(texts: List[str], model_name: str = 'en_core_web_sm') -> List[Dict]:
	"""Analyze texts using spaCy + SpacyTextBlob. Returns list of dicts with polarity and subjectivity."""
	try:
		import spacy
		from spacytextblob.spacytextblob import SpacyTextBlob
	except Exception as e:
		raise RuntimeError("spaCy or spacytextblob is not installed. Please install dependencies.") from e

	try:
		nlp = spacy.load(model_name)
	except Exception as e:
		raise RuntimeError(
			f"Could not load spaCy model '{model_name}'. Run: python -m spacy download {model_name}"
		) from e

	if 'spacytextblob' not in nlp.pipe_names:
		nlp.add_pipe('spacytextblob')

	results = []
	for t in texts:
		t_str = '' if t is None else str(t)
		doc = nlp(t_str)
		polarity = getattr(doc._, 'polarity', 0.0)
		subjectivity = getattr(doc._, 'subjectivity', 0.0)
		if polarity > 0.05:
			label = 'positive'
		elif polarity < -0.05:
			label = 'negative'
		else:
			label = 'neutral'

		results.append({
			'text': t_str,
			'method': 'spacy_textblob',
			'polarity': polarity,
			'subjectivity': subjectivity,
			'label': label,
		})
	return results


def main(argv=None):
	parser = argparse.ArgumentParser(description='Sentiment analysis with NLTK or spaCy')
	parser.add_argument('--input', '-i', help='Path to input CSV file (optional)')
	parser.add_argument('--text-column', '-c', help='Name of text column in CSV (default: text)')
	parser.add_argument('--method', '-m', choices=['nltk', 'spacy'], default='nltk', help='Analysis method')
	parser.add_argument('--model', default='en_core_web_sm', help='spaCy model name (for method=spacy)')
	parser.add_argument('--output', '-o', help='Path to output CSV file (optional)')
	parser.add_argument('--sample', action='store_true', help='Run sample analysis on example sentences')
	args = parser.parse_args(argv)

	if args.sample:
		sample_texts = [
			"I love this product! It's absolutely wonderful.",
			"This is the worst experience I've ever had.",
			"It's okay, nothing special.",
		]
		method = args.method
		if method == 'nltk':
			out = analyze_nltk(sample_texts)
		else:
			out = analyze_spacy(sample_texts, args.model)
		df = pd.DataFrame(out)
		print(df.to_string(index=False))
		return

	if not args.input:
		print('No input provided. Use --sample or --input <file.csv>.', file=sys.stderr)
		parser.print_help()
		return

	# Read CSV
	try:
		df_in = pd.read_csv(args.input)
	except Exception as e:
		raise RuntimeError(f"Failed to read input CSV: {e}") from e

	text_col = args.text_column or 'text'
	if text_col not in df_in.columns:
		raise RuntimeError(f"Text column '{text_col}' not found in input CSV. Available columns: {list(df_in.columns)}")

	texts = df_in[text_col].astype(str).tolist()

	if args.method == 'nltk':
		results = analyze_nltk(texts)
	else:
		results = analyze_spacy(texts, args.model)

	df_out = pd.DataFrame(results)
	if args.output:
		df_out.to_csv(args.output, index=False)
		print(f"Wrote results to {args.output}")
	else:
		out_path = 'sentiment_output.csv'
		df_out.to_csv(out_path, index=False)
		print(f"Wrote results to {out_path}")


if __name__ == '__main__':
	main()

