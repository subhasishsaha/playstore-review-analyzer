"""
LLM integration for generating actionable developer recommendations using Qwen via the official Ollama library.
"""
import sys
import re
import ast
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    ollama = None
    HAS_OLLAMA = False

from src.config import (
    OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    MAX_REVIEWS_PER_LABEL_SUGGESTION,
    MAX_REVIEWS_PER_LABEL_ANALYSIS,
)


def prepare_label_to_reviews(df, max_reviews_per_label: int = MAX_REVIEWS_PER_LABEL_ANALYSIS) -> Dict[str, List[str]]:
    """
    Extracts and maps negative reviews grouped by category/aspect label.
    """
    label_to_reviews = defaultdict(list)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "sentiment" not in df.columns:
        return {}

    negative_rows = df[df["sentiment"] == "Negative"]
    for _, row in negative_rows.iterrows():
        raw_labels = row.get("labels")
        if raw_labels is None:
            continue

        if isinstance(raw_labels, (list, tuple, set, np.ndarray, pd.Series)):
            labels = list(raw_labels)
        elif isinstance(raw_labels, str):
            try:
                parsed = ast.literal_eval(raw_labels)
                labels = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
            except (ValueError, SyntaxError):
                labels = [raw_labels]
        else:
            try:
                if pd.isna(raw_labels):
                    continue
            except (ValueError, TypeError):
                continue
            labels = [raw_labels]

        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        for label in labels:
            if label is None:
                continue
            try:
                if pd.isna(label):
                    continue
            except (ValueError, TypeError):
                continue
            clean = str(label).strip().lower()
            if clean and clean != "unknown":
                label_to_reviews[clean].append(content.strip())

    return {k: list(dict.fromkeys(v))[:max_reviews_per_label] for k, v in label_to_reviews.items()}


def _normalize_model_tag(tag: str) -> str:
    """
    Normalizes Ollama model tags for robust comparison (case-insensitive, trims prefixes and tags).
    """
    if not tag or not str(tag).strip():
        return ""
    t = str(tag).strip().lower()
    # Strip registry prefix if present e.g. library/qwen:latest -> qwen:latest
    if "/" in t:
        t = t.split("/")[-1]
    return t if ":" in t else f"{t}:latest"


def _build_ollama_client(base_url: str, timeout: float = 30.0, connect_timeout: float = 5.0):
    """
    Builds an ollama.Client instance with defensive connection and read timeouts.
    """
    clean_url = str(base_url).strip()
    try:
        import httpx
        timeout_cfg = httpx.Timeout(timeout, connect=connect_timeout)
    except Exception:
        timeout_cfg = timeout
    return ollama.Client(host=clean_url, timeout=timeout_cfg)


def get_installed_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """
    Returns a list of locally installed Ollama models.
    Supports Pydantic model objects, dictionaries, and raw string model tags from ollama client.
    """
    if not HAS_OLLAMA:
        return []
    try:
        client = _build_ollama_client(base_url=base_url, timeout=5.0, connect_timeout=3.0)
        res = client.list()
        models_data = res.get("models", []) if isinstance(res, dict) else getattr(res, "models", [])
        models = []
        for m in models_data:
            if isinstance(m, str):
                name = m
            elif isinstance(m, dict):
                name = m.get("model") or m.get("name")
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                models.append(str(name).strip())
        return sorted(list(set(models)))
    except Exception:
        return []


def ensure_model_available(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Verifies that the requested model exists in Ollama; automatically pulls it if missing.
    """
    if not HAS_OLLAMA or not model_name or not str(model_name).strip():
        return False

    target = str(model_name).strip()
    clean_url = str(base_url).strip()
    try:
        # Check installed models
        try:
            installed = get_installed_models(base_url=clean_url)
            normalized_target = _normalize_model_tag(target)
            if any(normalized_target == _normalize_model_tag(name) for name in installed):
                return True
        except Exception as check_err:
            logging.getLogger(__name__).debug("Failed checking installed models: %s", check_err)

        # Model not found locally -> pull automatically
        client = _build_ollama_client(base_url=clean_url, timeout=600.0, connect_timeout=5.0)
        print(f"[INFO] Model '{target}' not found locally in Ollama. Automatically pulling...")
        client.pull(model=target)
        print(f"[OK] Successfully pulled '{target}'.")
        return True
    except Exception as e:
        print(f"[WARNING] Could not pull model '{target}': {e}")
        return False


def query_ollama(
    prompt: str,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    ensure_available: bool = True
) -> str:
    """
    Queries Ollama using the official ollama Python client.
    Automatically checks and pulls the model if not present (unless ensure_available=False).
    Cleans reasoning model thinking tags (<think>...</think>) for polished user output.
    """
    if not HAS_OLLAMA:
        return (
            "⚠️ The 'ollama' Python library is not installed. "
            "Please run 'pip install ollama' to enable local AI suggestions."
        )

    clean_model = (model_name or "").strip() or DEFAULT_OLLAMA_MODEL
    clean_url = str(base_url).strip()

    # Ensure model is downloaded if requested
    if ensure_available:
        available = ensure_model_available(model_name=clean_model, base_url=clean_url)
        if not available:
            return (
                f"⚠️ Model '{clean_model}' is not available in Ollama and could not be retrieved. "
                f"Please ensure Ollama is running (`ollama serve`) and model '{clean_model}' is available."
            )

    try:
        client = _build_ollama_client(base_url=clean_url, timeout=120.0, connect_timeout=5.0)
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        res = client.chat(
            model=clean_model,
            messages=messages,
            options={
                "presence_penalty": 0.0,
                "temperature": 0.3,
                "num_predict": 256,
            }
        )

        msg = res.get("message") if isinstance(res, dict) else getattr(res, "message", None)
        if isinstance(msg, dict):
            content = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""
        elif msg is not None:
            content = getattr(msg, "content", "") or ""
            thinking = getattr(msg, "thinking", "") or ""
        else:
            content = ""
            thinking = ""

        answer = str(content).strip()

        # Handle reasoning model output with <think>...</think> tags in content
        if "<think>" in answer:
            if "</think>" in answer:
                cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
                if cleaned:
                    answer = cleaned
                else:
                    think_match = re.search(r"<think>(.*?)</think>", answer, flags=re.DOTALL)
                    think_text = think_match.group(1) if think_match else ""
                    lines = [l.strip() for l in think_text.split("\n") if l.strip()]
                    candidates = [l for l in lines if any(l.startswith(p) for p in ["1.", "2.", "3.", "* ", "- ", "Fix ", "• "])]
                    if candidates:
                        answer = "\n".join(candidates[:3])
            else:
                raw_think = answer.replace("<think>", "").strip()
                lines = [l.strip() for l in raw_think.split("\n") if l.strip()]
                candidates = [l for l in lines if any(l.startswith(p) for p in ["1.", "2.", "3.", "* ", "- ", "Fix ", "• "])]
                if candidates:
                    answer = "\n".join(candidates[:3])

        if not answer and thinking:
            # Fallback for reasoning models if thinking attribute was used
            lines = [l.strip() for l in str(thinking).split("\n") if l.strip()]
            candidates = []
            for line in reversed(lines):
                if any(line.startswith(p) for p in ["1.", "2.", "3.", "* ", "- ", "Fix ", "• "]):
                    candidates.insert(0, line)
                    if len(candidates) >= 3:
                        break
            if candidates:
                answer = "\n".join(candidates)

        return answer if answer else "No suggestions generated."

    except Exception as e:
        return (
            f"⚠️ Error querying Ollama via ollama library: {e}. "
            f"Please ensure Ollama is running (`ollama serve`) and model '{clean_model}' is available."
        )


def generate_suggestions(
    label_to_reviews: Dict[str, List[str]],
    model_name: str = DEFAULT_OLLAMA_MODEL,
    max_reviews: int = MAX_REVIEWS_PER_LABEL_SUGGESTION,
    base_url: str = OLLAMA_BASE_URL
) -> str:
    """
    Prompts Qwen via the official ollama library to produce actionable recommendations based on negative reviews per category.
    """
    if not HAS_OLLAMA:
        return (
            "⚠️ The 'ollama' Python library is not installed. "
            "Please run 'pip install ollama' to enable local AI suggestions."
        )

    if not label_to_reviews:
        return "No actionable negative reviews found to generate suggestions."

    clean_model = (model_name or "").strip() or DEFAULT_OLLAMA_MODEL
    clean_url = str(base_url).strip()

    # Pre-verify model availability once before iterating through labels
    if not ensure_model_available(model_name=clean_model, base_url=clean_url):
        return (
            f"⚠️ Model '{clean_model}' is not available in Ollama and could not be retrieved. "
            f"Please ensure Ollama is running (`ollama serve`) and the model exists."
        )

    # Prioritize top categories by complaint frequency (up to 5 categories)
    sorted_labels = sorted(label_to_reviews.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    formatted_suggestions = ""
    for label, reviews in sorted_labels:
        if not reviews:
            continue
        reviews_bullet_list = "\n".join(f"- {r}" for r in reviews[:max_reviews])
        prompt = (
            f"Here are user complaints about {label}:\n"
            f"{reviews_bullet_list}\n\n"
            f"List 2 practical, highly actionable improvements for developers to fix these issues:"
        )

        suggestion_text = query_ollama(prompt, model_name=clean_model, base_url=clean_url, ensure_available=False)
        formatted_suggestions += f"### Suggestions for: **{label.title()}**\n{suggestion_text}\n\n---\n\n"

    if not formatted_suggestions:
        return "No actionable negative reviews found to generate suggestions."

    return formatted_suggestions


# Backward compatibility alias
generate_gemini_suggestions = generate_suggestions

