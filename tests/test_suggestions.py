"""
Unit tests for developer suggestion preparation, grouping, and Qwen / ZeroGPU integration.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from src.suggestions import (
    prepare_label_to_reviews,
    generate_suggestions,
    query_qwen,
    get_qwen_model,
    ensure_model_available,
    get_installed_models,
    _normalize_model_tag,
)


class TestSuggestions(unittest.TestCase):
    def test_prepare_label_to_reviews_empty(self):
        self.assertEqual(prepare_label_to_reviews(None), {})
        self.assertEqual(prepare_label_to_reviews(pd.DataFrame()), {})

    def test_prepare_label_to_reviews_no_negative(self):
        df = pd.DataFrame([
            {"sentiment": "Positive", "labels": ["ui/ux"], "content": "Great look!"}
        ])
        self.assertEqual(prepare_label_to_reviews(df), {})

    def test_prepare_label_to_reviews_negative_grouped(self):
        df = pd.DataFrame([
            {"sentiment": "Negative", "labels": ["crashes", "bugs"], "content": "Crashes on startup"},
            {"sentiment": "Negative", "labels": "['crashes', 'performance']", "content": "Very slow and crashes"},
            {"sentiment": "Negative", "labels": ["unknown"], "content": "Just bad"},
            {"sentiment": "Positive", "labels": ["features"], "content": "Good feature set"},
        ])
        grouped = prepare_label_to_reviews(df)
        self.assertIn("crashes", grouped)
        self.assertIn("bugs", grouped)
        self.assertIn("performance", grouped)
        self.assertEqual(len(grouped["crashes"]), 2)

    def test_prepare_label_to_reviews_nan_and_none_labels(self):
        import numpy as np
        df = pd.DataFrame([
            {"sentiment": "Negative", "labels": np.nan, "content": "Nan labels"},
            {"sentiment": "Negative", "labels": None, "content": "None labels"},
            {"sentiment": "Negative", "labels": ["crashes", None, np.nan], "content": "Mixed list"},
            {"sentiment": "Negative", "labels": ["bugs"], "content": ""},  # empty content
        ])
        grouped = prepare_label_to_reviews(df)
        self.assertIn("crashes", grouped)
        self.assertNotIn("bugs", grouped)
        self.assertEqual(grouped["crashes"], ["Mixed list"])

    def test_generate_suggestions_empty(self):
        msg = generate_suggestions({})
        self.assertIn("No actionable negative reviews found", msg)

    @patch("src.suggestions.AutoTokenizer.from_pretrained")
    @patch("src.suggestions.AutoModelForCausalLM.from_pretrained")
    def test_get_qwen_model_mock(self, mock_model_cls, mock_tok_cls):
        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_tok_cls.return_value = mock_tok
        mock_model_cls.return_value = mock_model

        with patch("src.suggestions._CACHED_MODEL", None), patch("src.suggestions._CACHED_TOKENIZER", None):
            model, tokenizer = get_qwen_model("test-model")
            self.assertEqual(tokenizer, mock_tok)
            self.assertEqual(model, mock_model)
            mock_model.eval.assert_called_once()

    @patch("src.suggestions.get_qwen_model")
    def test_query_qwen_mock(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Formatted prompt"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        mock_model.generate.return_value = torch.tensor([[101, 102, 201, 202]])
        mock_tokenizer.batch_decode.return_value = ["1. Optimize database queries.\n2. Add caching layer."]
        mock_get_model.return_value = (mock_model, mock_tokenizer)

        result = query_qwen("Fix database latency")
        self.assertIn("Optimize database queries", result)
        self.assertIn("Add caching layer", result)

    @patch("src.suggestions.query_qwen")
    def test_generate_suggestions_mock(self, mock_query):
        mock_query.return_value = "### Category: **Crashes**\n1. Null check user profile."
        reviews_map = {"crashes": ["App crashes on splash screen"]}
        res = generate_suggestions(reviews_map)
        self.assertIn("Null check user profile", res)
        mock_query.assert_called_once()

    def test_compatibility_aliases(self):
        self.assertTrue(ensure_model_available("Qwen/Qwen2.5-1.5B-Instruct"))
        self.assertFalse(ensure_model_available(""))
        self.assertIn("Qwen/Qwen2.5-1.5B-Instruct", get_installed_models())
        self.assertEqual(_normalize_model_tag(" test "), "test")


if __name__ == "__main__":
    unittest.main()
