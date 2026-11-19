"""
Machine learning and deep learning model definitions, loading, and inference.
"""
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

from src.config import (
    HF_MODEL_REPO_ID,
    HF_MLB_FILE,
    HF_ENSEMBLE_FILE,
    HF_TFIDF_FILE,
    HF_SENTIMENT_WEIGHTS_FILE,
    BASE_ROBERTA_MODEL,
    SENTIMENT_BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
)


class SentimentClassifier(nn.Module):
    """
    RoBERTa-based binary sentiment classification model.
    Freezes all base layers except the final two encoder layers.
    """
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(BASE_ROBERTA_MODEL)
        for param in self.roberta.parameters():
            param.requires_grad = False
        for layer in self.roberta.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.last_hidden_state[:, 0, :]
        dropped = self.drop(pooled_output)
        return self.out(dropped)


# Backward compatibility alias
Sentiment_Classifier = SentimentClassifier


def _resolve_resource_path(relative_path: str, repo_id: str = HF_MODEL_REPO_ID) -> str:
    """
    Returns path to a local model file if present in Models/, otherwise downloads from Hugging Face Hub.
    """
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "Models" / relative_path,
        project_root / relative_path,
        Path("Models") / relative_path,
        Path(relative_path),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    # Download from Hugging Face Hub
    return hf_hub_download(repo_id=repo_id, filename=relative_path)


class ModelBundle:
    """
    Encapsulates loaded ML and NLP model artifacts.
    """
    def __init__(self, mlb, ensemble, tfidf, tokenizer, sentiment_model):
        self.mlb = mlb
        self.ensemble = ensemble
        self.tfidf = tfidf
        self.tokenizer = tokenizer
        self.sentiment_model = sentiment_model
        self.softmax = nn.Softmax(dim=1)

    def predict_labels(self, texts: List[str]) -> List[Tuple[str, ...]]:
        """
        Predicts multi-label categories for a list of cleaned text strings.
        Guarantees returned list length matches len(texts).
        """
        if not texts:
            return []

        results: List[Tuple[str, ...]] = [("unknown",)] * len(texts)
        valid_indices = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
        if not valid_indices:
            return results

        valid_texts = [texts[i] for i in valid_indices]
        x_tfidf = self.tfidf.transform(valid_texts)
        y_pred = np.array(self.ensemble.predict(x_tfidf))
        all_labels = self.mlb.inverse_transform(y_pred)
        for idx, labels in zip(valid_indices, all_labels):
            results[idx] = labels if labels else ("unknown",)
        return results

    def predict_sentiments(self, texts: List[str], batch_size: int = SENTIMENT_BATCH_SIZE) -> List[str]:
        """
        Predicts binary sentiment ('Positive' or 'Negative') for a list of texts.
        Guarantees returned list length matches len(texts).
        """
        if not texts:
            return []

        results = ["Negative"] * len(texts)
        valid_indices = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
        if not valid_indices:
            return results

        valid_texts = [texts[i] for i in valid_indices]
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=MAX_SEQUENCE_LENGTH
            )
            with torch.no_grad():
                outputs = self.sentiment_model(encoded["input_ids"], encoded["attention_mask"])
                probs = self.softmax(outputs)
                preds = torch.argmax(probs, dim=1).tolist()
                for j, p in enumerate(preds):
                    global_idx = valid_indices[i + j]
                    results[global_idx] = "Positive" if p == 1 else "Negative"
        return results


_GLOBAL_MODEL_BUNDLE = None


def load_resources() -> ModelBundle:
    """
    Loads all model artifacts, instantiating the bundle singleton if not already loaded.
    """
    global _GLOBAL_MODEL_BUNDLE
    if _GLOBAL_MODEL_BUNDLE is not None:
        return _GLOBAL_MODEL_BUNDLE

    try:
        mlb_path = _resolve_resource_path(HF_MLB_FILE)
        ensemble_path = _resolve_resource_path(HF_ENSEMBLE_FILE)
        tfidf_path = _resolve_resource_path(HF_TFIDF_FILE)
        weights_path = _resolve_resource_path(HF_SENTIMENT_WEIGHTS_FILE)

        mlb = joblib.load(mlb_path)
        ensemble = joblib.load(ensemble_path)
        tfidf = joblib.load(tfidf_path)

        tokenizer = AutoTokenizer.from_pretrained(BASE_ROBERTA_MODEL)

        sentiment_model = SentimentClassifier(n_classes=2)
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location="cpu")
        sentiment_model.load_state_dict(state_dict)
        sentiment_model.eval()

        _GLOBAL_MODEL_BUNDLE = ModelBundle(mlb, ensemble, tfidf, tokenizer, sentiment_model)
        return _GLOBAL_MODEL_BUNDLE

    except Exception as e:
        raise RuntimeError(f"Failed to load model resources: {e}")
