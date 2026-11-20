"""
AI suggestions engine using Qwen/Qwen2.5-1.5B-Instruct via Hugging Face Transformers and ZeroGPU.
"""
import sys
import re
import ast
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure project root is in sys.path
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    HF_SUGGESTIONS_MODEL,
    HF_SUGGESTIONS_MODEL_URL,
    MAX_REVIEWS_PER_LABEL_SUGGESTION,
    MAX_REVIEWS_PER_LABEL_ANALYSIS,
)

# Hugging Face Spaces ZeroGPU integration
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class _MockSpaces:
        def GPU(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
    spaces = _MockSpaces()


_CACHED_MODEL = None
_CACHED_TOKENIZER = None


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


def get_qwen_model(model_id: str = HF_SUGGESTIONS_MODEL) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Lazily loads Qwen/Qwen2.5-1.5B-Instruct on the active accelerator (CUDA with ZeroGPU or CPU).
    """
    global _CACHED_MODEL, _CACHED_TOKENIZER
    if _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None:
        if torch.cuda.is_available() and hasattr(_CACHED_MODEL, "device") and _CACHED_MODEL.device.type != "cuda":
            try:
                _CACHED_MODEL = _CACHED_MODEL.to("cuda")
            except Exception:
                pass
        return _CACHED_MODEL, _CACHED_TOKENIZER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    try:
        _CACHED_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
    except Exception:
        _CACHED_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            _CACHED_MODEL = _CACHED_MODEL.to("cuda")
        else:
            _CACHED_MODEL = _CACHED_MODEL.to("cpu")

    _CACHED_MODEL.eval()
    return _CACHED_MODEL, _CACHED_TOKENIZER


@spaces.GPU(duration=60)
def query_qwen(prompt: str, model_id: str = HF_SUGGESTIONS_MODEL, max_new_tokens: int = 512) -> str:
    """
    Queries Qwen2.5-1.5B-Instruct under ZeroGPU dynamic acceleration.
    """
    clean_model_id = (model_id or "").strip() or HF_SUGGESTIONS_MODEL
    try:
        model, tokenizer = get_qwen_model(clean_model_id)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert mobile application engineer and product manager. "
                    "Analyze the user complaints and provide concise, practical, highly actionable recommendations for developers."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        raw_inputs = tokenizer([text], return_tensors="pt")
        if hasattr(raw_inputs, "to"):
            model_inputs = raw_inputs.to(model.device)
        else:
            model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in raw_inputs.items()}

        input_ids = model_inputs["input_ids"] if isinstance(model_inputs, dict) else model_inputs.input_ids
        infer_ctx = getattr(torch, "inference_mode", torch.no_grad)
        with infer_ctx():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )
        output_tokens = [
            out[len(inp):] for inp, out in zip(input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return response.strip()
    except Exception as e:
        return f"⚠️ Error generating suggestions with {clean_model_id}: {str(e)}"


def generate_suggestions(
    label_to_reviews: Dict[str, List[str]],
    model_name: str = HF_SUGGESTIONS_MODEL,
    max_reviews: int = MAX_REVIEWS_PER_LABEL_SUGGESTION,
    **kwargs
) -> str:
    """
    Generates actionable developer recommendations using Qwen/Qwen2.5-1.5B-Instruct.
    Groups complaints by category and performs ZeroGPU-accelerated inference.
    """
    if not label_to_reviews:
        return "No actionable negative reviews found to generate suggestions."

    target_model = (model_name or "").strip() or HF_SUGGESTIONS_MODEL
    sorted_labels = sorted(label_to_reviews.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    prompt_lines = [
        "Analyze the following negative user reviews grouped by app aspect. "
        "For each category, provide 2 practical, highly actionable improvements for developers to fix these issues. "
        "Use markdown formatting with bold headings and bullet points.\n"
    ]
    for label, reviews in sorted_labels:
        if not reviews:
            continue
        prompt_lines.append(f"### Category: **{label.title()}**")
        for r in reviews[:max_reviews]:
            prompt_lines.append(f"- {r}")
        prompt_lines.append("")

    prompt = "\n".join(prompt_lines).strip()
    return query_qwen(prompt, model_id=target_model, max_new_tokens=512)


# Compatibility aliases for legacy interfaces
query_ollama = query_qwen
generate_gemini_suggestions = generate_suggestions

def ensure_model_available(model_name: str = HF_SUGGESTIONS_MODEL, **kwargs) -> bool:
    return bool(model_name and str(model_name).strip())

def get_installed_models(**kwargs) -> List[str]:
    return [HF_SUGGESTIONS_MODEL]

def _normalize_model_tag(tag: str) -> str:
    return str(tag).strip() if tag else ""
