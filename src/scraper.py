"""
Scraper module for fetching Google Play Store user reviews.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from google_play_scraper import reviews, Sort
except ImportError:
    reviews = None
    Sort = None

from src.config import SORT_OPTIONS, DEFAULT_LANG, DEFAULT_COUNTRY, DEFAULT_REVIEW_COUNT



def scrape_reviews(
    app_id: str,
    sort_order: str = "Most Relevant",
    count: int = DEFAULT_REVIEW_COUNT,
    lang: str = DEFAULT_LANG,
    country: str = DEFAULT_COUNTRY
) -> List[Dict[str, Any]]:
    """
    Scrapes user reviews from Google Play Store for a given App ID.

    Args:
        app_id: Google Play package name (e.g., 'com.google.android.gm').
        sort_order: Sort preference ('Most Relevant' or 'Newest').
        count: Number of reviews to fetch.
        lang: Target review language.
        country: Target country code.

    Returns:
        List of review dictionaries.
    """
    if not app_id or not app_id.strip():
        raise ValueError("Please enter a valid App ID.")

    if reviews is None:
        raise RuntimeError("google-play-scraper package is not installed. Install it with 'pip install google-play-scraper'.")

    sort_criteria = SORT_OPTIONS.get(sort_order, Sort.MOST_RELEVANT)
    result, _ = reviews(
        app_id.strip(),
        sort=sort_criteria,
        count=int(count),
        lang=lang,
        country=country
    )
    return result
