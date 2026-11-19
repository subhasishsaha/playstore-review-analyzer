"""
Unit tests for developer suggestion preparation and grouping.
"""
import unittest
import pandas as pd
from src.suggestions import prepare_label_to_reviews, generate_suggestions


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

    def test_get_installed_models_mock(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import get_installed_models

        mock_item1 = MagicMock()
        mock_item1.model = "qwen3.5:0.8b"
        mock_item2 = MagicMock()
        mock_item2.model = "llama3.2:1b"
        mock_list_resp = MagicMock()
        mock_list_resp.models = [mock_item1, mock_item2]

        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.list.return_value = mock_list_resp
            mock_client_cls.return_value = mock_client

            models = get_installed_models()
            self.assertIn("qwen3.5:0.8b", models)
            self.assertIn("llama3.2:1b", models)

    def test_ensure_model_available_installed(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import ensure_model_available

        with patch("src.suggestions.get_installed_models", return_value=["qwen3.5:0.8b"]):
            with patch("ollama.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                available = ensure_model_available("qwen3.5:0.8b")
                self.assertTrue(available)
                mock_client.pull.assert_not_called()

    def test_ensure_model_available_pulls_missing(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import ensure_model_available

        with patch("src.suggestions.get_installed_models", return_value=["qwen3.5:0.8b"]):
            with patch("ollama.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                available = ensure_model_available("mistral:7b")
                self.assertTrue(available)
                mock_client.pull.assert_called_once_with(model="mistral:7b")


    def test_normalize_model_tag(self):
        from src.suggestions import _normalize_model_tag
        self.assertEqual(_normalize_model_tag("qwen3.5:0.8b"), "qwen3.5:0.8b")
        self.assertEqual(_normalize_model_tag("QWEN3.5:0.8B"), "qwen3.5:0.8b")
        self.assertEqual(_normalize_model_tag("qwen3.5"), "qwen3.5:latest")
        self.assertEqual(_normalize_model_tag("  llama3.2:1b  "), "llama3.2:1b")
        self.assertEqual(_normalize_model_tag("library/mistral:7b"), "mistral:7b")
        self.assertEqual(_normalize_model_tag("registry.ollama.ai/library/qwen:latest"), "qwen:latest")
        self.assertEqual(_normalize_model_tag(""), "")
        self.assertEqual(_normalize_model_tag(None), "")

    def test_get_installed_models_dict_format(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import get_installed_models

        mock_list_resp = {
            "models": [
                {"name": "qwen3.5:0.8b"},
                {"model": "llama3.2:1b"}
            ]
        }
        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.list.return_value = mock_list_resp
            mock_client_cls.return_value = mock_client

            models = get_installed_models()
            self.assertIn("qwen3.5:0.8b", models)
            self.assertIn("llama3.2:1b", models)

    def test_ensure_model_available_case_insensitive_and_whitespace(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import ensure_model_available

        with patch("src.suggestions.get_installed_models", return_value=["qwen3.5:0.8b"]):
            with patch("ollama.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                available = ensure_model_available("  Qwen3.5:0.8B  ")
                self.assertTrue(available)
                mock_client.pull.assert_not_called()

    def test_ensure_model_available_empty_or_none(self):
        from src.suggestions import ensure_model_available
        self.assertFalse(ensure_model_available(""))
        self.assertFalse(ensure_model_available(None))
        self.assertFalse(ensure_model_available("   "))

    def test_ensure_model_available_pull_failure(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import ensure_model_available

        with patch("src.suggestions.get_installed_models", return_value=[]):
            with patch("ollama.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.pull.side_effect = RuntimeError("Connection refused")
                mock_client_cls.return_value = mock_client
                available = ensure_model_available("unknown:model")
                self.assertFalse(available)

    def test_query_ollama_success(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import query_ollama

        mock_msg = MagicMock()
        mock_msg.content = "1. Optimize database queries.\n2. Add caching layer."
        mock_msg.thinking = ""
        mock_resp = MagicMock()
        mock_resp.message = mock_msg

        with patch("src.suggestions.ensure_model_available", return_value=True):
            with patch("ollama.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.return_value = mock_resp
                mock_client_cls.return_value = mock_client

                answer = query_ollama("Fix performance issues", model_name="qwen3.5:0.8b")
                self.assertIn("Optimize database queries", answer)
                self.assertIn("Add caching layer", answer)

    def test_query_ollama_reasoning_fallback(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import query_ollama

        # Model exhausted tokens in thinking attribute with empty content
        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.thinking = "Let me think...\n1. Fix memory leak on resume.\n2. Cache view bindings."
        mock_resp = MagicMock()
        mock_resp.message = mock_msg

        with patch("src.suggestions.ensure_model_available", return_value=True):
            with patch("ollama.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.return_value = mock_resp
                mock_client_cls.return_value = mock_client

                answer = query_ollama("Fix crash on resume", model_name="qwen3.5:0.8b")
                self.assertIn("Fix memory leak", answer)

    def test_query_ollama_unavailable_model(self):
        from unittest.mock import patch
        from src.suggestions import query_ollama

        with patch("src.suggestions.ensure_model_available", return_value=False):
            res = query_ollama("Some prompt", model_name="broken:model")
            self.assertIn("⚠️", res)
            self.assertIn("broken:model", res)

    def test_generate_suggestions_unavailable_model(self):
        from unittest.mock import patch
        from src.suggestions import generate_suggestions

        reviews_map = {"crashes": ["App crashes on splash screen"]}
        with patch("src.suggestions.ensure_model_available", return_value=False):
            res = generate_suggestions(reviews_map, model_name="broken:model")
            self.assertIn("⚠️", res)
            self.assertIn("broken:model", res)

    def test_generate_suggestions_success(self):
        from unittest.mock import patch
        from src.suggestions import generate_suggestions

        reviews_map = {
            "crashes": ["Crashes when clicking profile button"],
            "ui/ux": ["Dark mode contrast is too low"]
        }
        with patch("src.suggestions.ensure_model_available", return_value=True):
            with patch("src.suggestions.query_ollama", return_value="1. Null check user profile.\n2. Add crashlytics."):
                res = generate_suggestions(reviews_map, model_name="qwen3.5:0.8b")
                self.assertIn("Suggestions for: **Crashes**", res)
                self.assertIn("Null check user profile", res)

    def test_get_installed_models_string_format(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import get_installed_models

        mock_list_resp = MagicMock()
        mock_list_resp.models = ["qwen3.5:0.8b", "llama3.2:1b"]

        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.list.return_value = mock_list_resp
            mock_client_cls.return_value = mock_client

            models = get_installed_models()
            self.assertIn("qwen3.5:0.8b", models)
            self.assertIn("llama3.2:1b", models)

    def test_query_ollama_content_with_think_tags(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import query_ollama

        mock_msg = MagicMock()
        mock_msg.content = "<think>\nAnalyzing the issue thoroughly...\n</think>\n1. Implement pagination.\n2. Add indexes."
        mock_msg.thinking = ""
        mock_resp = MagicMock()
        mock_resp.message = mock_msg

        with patch("src.suggestions.ensure_model_available", return_value=True):
            with patch("ollama.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.return_value = mock_resp
                mock_client_cls.return_value = mock_client

                answer = query_ollama("Fix slow performance", model_name="qwen3.5:0.8b")
                self.assertNotIn("<think>", answer)
                self.assertIn("Implement pagination", answer)

    def test_query_ollama_content_unclosed_think_tag(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import query_ollama

        mock_msg = MagicMock()
        mock_msg.content = "<think>\nLet me think about how to fix this...\n1. Use background thread.\n2. Add progress bar."
        mock_msg.thinking = ""
        mock_resp = MagicMock()
        mock_resp.message = mock_msg

        with patch("src.suggestions.ensure_model_available", return_value=True):
            with patch("ollama.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.return_value = mock_resp
                mock_client_cls.return_value = mock_client

                answer = query_ollama("Fix frozen UI", model_name="qwen3.5:0.8b")
                self.assertNotIn("<think>", answer)
                self.assertIn("Use background thread", answer)

    def test_query_ollama_skip_ensure_available(self):
        from unittest.mock import patch, MagicMock
        from src.suggestions import query_ollama

        mock_msg = MagicMock()
        mock_msg.content = "1. Cache data locally."
        mock_msg.thinking = ""
        mock_resp = MagicMock()
        mock_resp.message = mock_msg

        with patch("src.suggestions.ensure_model_available") as mock_ensure:
            with patch("ollama.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.return_value = mock_resp
                mock_client_cls.return_value = mock_client

                answer = query_ollama("Fix offline mode", model_name="qwen3.5:0.8b", ensure_available=False)
                mock_ensure.assert_not_called()
                self.assertIn("Cache data locally", answer)

    def test_generate_suggestions_no_ollama_installed(self):
        from unittest.mock import patch
        import src.suggestions as sugg_mod

        with patch.object(sugg_mod, "HAS_OLLAMA", False):
            res = sugg_mod.generate_suggestions({"crashes": ["crash 1"]})
            self.assertIn("The 'ollama' Python library is not installed", res)


if __name__ == "__main__":
    unittest.main()

