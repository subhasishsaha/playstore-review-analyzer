"""
Text preprocessing and language filtering utilities.
"""
import re
import numpy as np

# Pre-compile regular expressions once at module level to eliminate per-call regex parsing overhead
_CLEAN_REGEX = re.compile(r"http\S+|www\S+|<.*?>|\n|\w*\d\w*")

try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


def preprocess(text: str) -> str:
    """
    Cleans raw review text by removing emojis, URLs, HTML tags, and alphanumeric artifacts.
    Uses pre-compiled regex for maximum throughput with zero overhead.
    """
    if not isinstance(text, str) or not text.strip():
        return np.nan

    if HAS_EMOJI:
        text = emoji.replace_emoji(text, "")
    text = _CLEAN_REGEX.sub("", text).strip()
    return text if text else np.nan



def filter_english_sentences(text: str) -> str:
    """
    Splits text into sentences and retains only those identified as English.
    Includes a fast-path whole-text verification that avoids expensive sentence-by-sentence
    profiling when the entire review is homogeneous English.
    """
    if not isinstance(text, str):
        return np.nan

    cleaned = text.strip()
    if not cleaned:
        return np.nan

    if not HAS_LANGDETECT:
        return cleaned

    # Fast-path: verify full text first (1 detector call instead of N sentence calls)
    try:
        if detect(cleaned) == "en":
            return cleaned
    except LangDetectException as lang_err:
        import logging
        logging.getLogger(__name__).debug("Language detection fast-path: %s", lang_err)

    # Fallback: Sentence-level extraction for mixed-language content
    sentences = [s.strip() for s in cleaned.split(".") if len(s.split()) > 1]
    english = []
    for s in sentences:
        try:
            if detect(s) == "en":
                english.append(s)
        except LangDetectException:
            continue
    return ". ".join(english) if english else np.nan


def prepare_cleaned_reviews(df):
    """
    Applies text preprocessing and English filtering to a reviews DataFrame.
    Returns a filtered DataFrame with non-empty 'content_processed' column.
    """
    df_processed = df.copy()
    df_processed["content_processed"] = df_processed["content"].apply(preprocess)
    df_processed.dropna(subset=["content_processed"], inplace=True)

    if not df_processed.empty:
        df_processed["content_processed"] = df_processed["content_processed"].apply(filter_english_sentences)
        df_processed.dropna(subset=["content_processed"], inplace=True)

    return df_processed
