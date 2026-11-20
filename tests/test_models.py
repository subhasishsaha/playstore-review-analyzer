"""
Unit tests for SentimentClassifier architecture and resource resolution.
"""
import unittest
from pathlib import Path
from src.models import _resolve_resource_path
from src.config import (
    HF_MLB_FILE,
    HF_ENSEMBLE_FILE,
    HF_TFIDF_FILE,
    HF_SENTIMENT_WEIGHTS_FILE,
)


class TestModels(unittest.TestCase):
    def test_local_model_paths_exist(self):
        for resource in [HF_MLB_FILE, HF_ENSEMBLE_FILE, HF_TFIDF_FILE, HF_SENTIMENT_WEIGHTS_FILE]:
            resolved = _resolve_resource_path(resource)
            self.assertTrue(Path(resolved).is_file(), f"File {resolved} should exist locally.")

    def test_predict_methods_alignment_and_empty_handling(self):
        from src.models import load_resources
        bundle = load_resources()

        # 1. Empty list
        self.assertEqual(bundle.predict_labels([]), [])
        self.assertEqual(bundle.predict_sentiments([]), [])

        # 2. Mixed inputs: empty string, whitespace, non-string, valid text
        mixed = ["Awesome app, loved the dark mode interface!", "", "   ", "Terrible crash on login."]
        labels = bundle.predict_labels(mixed)
        sentiments = bundle.predict_sentiments(mixed)

        # Output lengths must strictly equal input length
        self.assertEqual(len(labels), len(mixed))
        self.assertEqual(len(sentiments), len(mixed))

        # Check default fallback for empty positions
        self.assertEqual(labels[1], ("unknown",))
        self.assertEqual(labels[2], ("unknown",))
        self.assertIsInstance(labels[0], tuple)
        self.assertIsInstance(labels[3], tuple)

        self.assertIn(sentiments[0], ("Positive", "Negative"))
        self.assertIn(sentiments[3], ("Positive", "Negative"))


if __name__ == "__main__":
    unittest.main()
