"""
Play Store Review Analyzer package (Hugging Face / Gradio edition).
"""
import sys
from pathlib import Path

# Ensure package directory and project root are in sys.path
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import SentimentClassifier, load_resources
from src.preprocessing import preprocess, filter_english_sentences
from src.scraper import scrape_reviews
from src.suggestions import prepare_label_to_reviews, generate_suggestions, query_qwen
from src.pipeline import analyze_reviews, analyze_reviews_data

__all__ = [
    "SentimentClassifier",
    "load_resources",
    "preprocess",
    "filter_english_sentences",
    "scrape_reviews",
    "prepare_label_to_reviews",
    "generate_suggestions",
    "query_qwen",
    "analyze_reviews",
    "analyze_reviews_data",
]
