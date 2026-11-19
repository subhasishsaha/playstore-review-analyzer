"""
Pipeline orchestration: connects scraping, preprocessing, ML models, and LLM suggestions.
"""
import sys
import time
import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DISPLAY_COLUMNS, DEFAULT_OLLAMA_MODEL
from src.scraper import scrape_reviews
from src.preprocessing import prepare_cleaned_reviews
from src.models import load_resources
from src.suggestions import prepare_label_to_reviews, generate_suggestions

# Zero-dependency in-memory LRU/TTL cache for scraping and model classification
_ANALYSIS_CACHE: Dict[Tuple[str, str, int], Tuple[float, Dict[str, Any], pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes
_MAX_CACHE_ENTRIES = 20


def _sanitize_for_json(obj):
    """
    Recursively converts pandas/numpy/datetime types into standard JSON-compatible Python types.
    """
    if isinstance(obj, (tuple, set, list, np.ndarray, pd.Series)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (pd.Timestamp, np.datetime64, datetime.date, datetime.datetime)):
        return obj.isoformat() if hasattr(obj, "isoformat") else str(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        return obj
    return obj


def analyze_reviews_data(
    app_id: str,
    sort_order: str = "Most Relevant",
    count: int = 100,
    ollama_model: str = None
) -> dict:
    """
    Core pipeline function that executes the analysis and returns structured JSON-ready data.
    Uses an in-memory TTL cache to return duplicate requests in <10ms with zero extra overhead.
    """
    if not app_id or not app_id.strip():
        raise ValueError("Please provide a valid Google Play App ID.")

    clean_app_id = app_id.strip()
    cache_key = (clean_app_id, sort_order, count)
    target_model = (ollama_model or "").strip() or DEFAULT_OLLAMA_MODEL
    now = time.time()

    # Zero-overhead in-memory cache check
    if cache_key in _ANALYSIS_CACHE:
        cached_ts, cached_result, cached_df = _ANALYSIS_CACHE[cache_key]
        if now - cached_ts < _CACHE_TTL_SECONDS:
            if cached_result.get("model_used") == target_model:
                return dict(cached_result)
            # Re-generate suggestions only if caller requested a different LLM model
            label_to_reviews = prepare_label_to_reviews(cached_df)
            new_suggestions = generate_suggestions(label_to_reviews, model_name=target_model)
            updated = dict(cached_result)
            updated["model_used"] = target_model
            updated["suggestions_markdown"] = new_suggestions
            _ANALYSIS_CACHE[cache_key] = (now, updated, cached_df)
            return updated

    raw_reviews = scrape_reviews(app_id=clean_app_id, sort_order=sort_order, count=count)
    if not raw_reviews:
        raise ValueError(f"No reviews fetched for App ID '{app_id}'. Please check if the App ID exists on Google Play.")

    df = pd.DataFrame(raw_reviews)
    df_processed = prepare_cleaned_reviews(df)

    if df_processed.empty:
        raise ValueError("No valid English review content could be extracted after preprocessing.")

    # Load ML/DL Models
    models = load_resources()

    texts = df_processed["content_processed"].tolist()
    predicted_labels = models.predict_labels(texts)
    predicted_sentiments = models.predict_sentiments(texts)

    df_processed["labels"] = [list(lbls) if isinstance(lbls, (list, tuple)) else [lbls] for lbls in predicted_labels]
    df_processed["sentiment"] = predicted_sentiments

    # Set df to df_processed containing all predictions and columns
    df = df_processed.reset_index(drop=True)

    # Calculate summaries
    total = len(df)
    sentiments = df["sentiment"].tolist()
    positive_count = sum(1 for s in sentiments if s == "Positive")
    negative_count = sum(1 for s in sentiments if s == "Negative")

    # Aspect labels distribution
    all_aspects = []
    for labels in df["labels"]:
        if isinstance(labels, (list, tuple)):
            for l in labels:
                clean_l = str(l).strip().lower()
                if clean_l and clean_l != "unknown":
                    all_aspects.append(clean_l.title())

    labels_distribution = dict(Counter(all_aspects).most_common())

    # Sentiment by aspect breakdown - fast column iteration (eliminates df.iterrows overhead)
    sentiment_by_label = defaultdict(lambda: {"Positive": 0, "Negative": 0})
    for s, row_labels in zip(df["sentiment"], df["labels"]):
        if s not in ("Positive", "Negative") or not isinstance(row_labels, (list, tuple)):
            continue
        for l in row_labels:
            clean_l = str(l).strip().lower()
            if clean_l and clean_l != "unknown":
                sentiment_by_label[clean_l.title()][s] += 1

    # AI Suggestions via Ollama (using user-requested model or default)
    label_to_reviews = prepare_label_to_reviews(df)
    suggestions_md = generate_suggestions(label_to_reviews, model_name=target_model)

    # Format records for JSON via vectorized DataFrame methods
    cols = ["reviewId", "userName", "content", "score", "at", "appVersion", "sentiment", "labels"]
    valid_cols = [c for c in cols if c in df.columns]

    export_df = df[valid_cols].copy()
    if "at" in export_df.columns:
        export_df["at"] = export_df["at"].apply(
            lambda v: v.isoformat() if hasattr(v, "isoformat") else (str(v) if pd.notna(v) else None)
        )
    # Vectorized NaN/NaT null replacement
    export_df = export_df.where(pd.notnull(export_df), None)
    sanitized_records = export_df.to_dict(orient="records")

    result_data = {
        "status": "success",
        "app_id": app_id,
        "model_used": target_model,
        "total": total,
        "summary": {
            "total": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_percentage": round((positive_count / total * 100) if total else 0, 1),
            "negative_percentage": round((negative_count / total * 100) if total else 0, 1),
            "labels_distribution": labels_distribution,
            "sentiment_by_label": dict(sentiment_by_label)
        },
        "reviews": sanitized_records,
        "suggestions_markdown": suggestions_md,
    }

    # Store in bounded in-memory cache
    if len(_ANALYSIS_CACHE) >= _MAX_CACHE_ENTRIES:
        oldest_key = min(_ANALYSIS_CACHE, key=lambda k: _ANALYSIS_CACHE[k][0])
        _ANALYSIS_CACHE.pop(oldest_key, None)
    _ANALYSIS_CACHE[cache_key] = (now, result_data, df)

    return result_data


def analyze_reviews(app_id: str, sort_order: str, count: int, ollama_model: str = None):
    """
    Compatibility wrapper returning (DataFrame, DataFrame, str) for legacy callers.
    """
    res = analyze_reviews_data(app_id=app_id, sort_order=sort_order, count=count, ollama_model=ollama_model)
    df = pd.DataFrame(res["reviews"])
    available_cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    return df[available_cols], df, res["suggestions_markdown"]
