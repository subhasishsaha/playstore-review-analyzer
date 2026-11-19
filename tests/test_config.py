"""
Unit tests for configuration constants and path settings.
"""
import unittest
from src.config import (
    SORT_OPTIONS,
    DEFAULT_REVIEW_COUNT,
    DEFAULT_LANG,
    DEFAULT_COUNTRY,
    HF_MODEL_REPO_ID,
    HF_MLB_FILE,
    HF_ENSEMBLE_FILE,
    HF_TFIDF_FILE,
    HF_SENTIMENT_WEIGHTS_FILE,
)


class TestConfig(unittest.TestCase):
    def test_sort_options(self):
        self.assertIn("Most Relevant", SORT_OPTIONS)
        self.assertIn("Newest", SORT_OPTIONS)

    def test_default_values(self):
        self.assertEqual(DEFAULT_REVIEW_COUNT, 100)
        self.assertEqual(DEFAULT_LANG, "en")
        self.assertEqual(DEFAULT_COUNTRY, "us")
        self.assertEqual(HF_MODEL_REPO_ID, "ssaha007/playstore-models")

    def test_model_files_defined(self):
        self.assertTrue(HF_MLB_FILE.endswith(".pkl"))
        self.assertTrue(HF_ENSEMBLE_FILE.endswith(".pkl"))
        self.assertTrue(HF_TFIDF_FILE.endswith(".pkl"))
        self.assertTrue(HF_SENTIMENT_WEIGHTS_FILE.endswith(".bin"))

    def test_ollama_config_defaults(self):
        from src.config import OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL
        self.assertTrue(bool(OLLAMA_BASE_URL))
        self.assertTrue(bool(DEFAULT_OLLAMA_MODEL))
        self.assertTrue(OLLAMA_BASE_URL.startswith("http"))
        self.assertIn("qwen", DEFAULT_OLLAMA_MODEL)


if __name__ == "__main__":
    unittest.main()
