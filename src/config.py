"""
Configuration and constants for Play Store Review Analyzer.
"""
import os
from pathlib import Path

# Load environment variables from .env file if dotenv is available
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.is_file():
        load_dotenv(dotenv_path=_ENV_PATH)
    else:
        load_dotenv()
except ImportError:
    load_dotenv = None

try:
    from google_play_scraper import Sort
    SORT_OPTIONS = {
        "Most Relevant": Sort.MOST_RELEVANT,
        "Newest": Sort.NEWEST,
    }
except ImportError:
    SORT_OPTIONS = {
        "Most Relevant": 1,
        "Newest": 2,
    }

# Ollama & LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip() or "qwen3.5:0.8b"
MAX_REVIEWS_PER_LABEL_SUGGESTION = 3
MAX_REVIEWS_PER_LABEL_ANALYSIS = 10

# Hugging Face Model Repository
HF_MODEL_REPO_ID = "ssaha007/playstore-models"
HF_MLB_FILE = "ensemble_models/multilabel_binarizer.pkl"
HF_ENSEMBLE_FILE = "ensemble_models/ensemble_model.pkl"
HF_TFIDF_FILE = "ensemble_models/tfidf.pkl"
HF_SENTIMENT_WEIGHTS_FILE = "sentiment_model/best_model_state.bin"

# RoBERTa Configuration
BASE_ROBERTA_MODEL = "roberta-base"
SENTIMENT_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 256


DEFAULT_REVIEW_COUNT = 100
DEFAULT_LANG = "en"
DEFAULT_COUNTRY = "us"

# UI Display
DISPLAY_COLUMNS = ['userName', 'content', 'at', 'appVersion', 'sentiment', 'labels']
