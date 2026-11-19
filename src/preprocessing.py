"""
Text preprocessing and language filtering utilities.
"""
import re
import numpy as np
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
    """
    if not isinstance(text, str) or not text.strip():
        return np.nan

    if HAS_EMOJI:
        text = emoji.replace_emoji(text, "")
    text = re.sub(r"http\S+|www\S+|<.*?>|\n|\w*\d\w*", "", text).strip()
    return text if text else np.nan



def filter_english_sentences(text: str) -> str:
    """
    Splits text into sentences and retains only those identified as English.
    """
    if not isinstance(text, str):
        return np.nan

    if not HAS_LANGDETECT:
        return text.strip() if text.strip() else np.nan

    sentences = [s.strip() for s in text.split(".") if len(s.split()) > 1]
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
