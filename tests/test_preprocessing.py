"""
Unit tests for text preprocessing and English language filtering.
"""
import unittest
import pandas as pd
from src.preprocessing import preprocess, filter_english_sentences, prepare_cleaned_reviews


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_cleans_urls_and_html(self):
        text = "Check this out <p>Great app!</p> https://example.com/download www.test.com"
        cleaned = preprocess(text)
        self.assertNotIn("<p>", str(cleaned))
        self.assertNotIn("https://", str(cleaned))
        self.assertNotIn("www.", str(cleaned))
        self.assertIn("Great app!", str(cleaned))

    def test_preprocess_cleans_emojis(self):
        text = "Loving this app so much! 😍🔥🎉"
        cleaned = preprocess(text)
        self.assertNotIn("😍", str(cleaned))
        self.assertIn("Loving this app so much!", str(cleaned))

    def test_preprocess_empty_and_none(self):
        self.assertTrue(pd.isna(preprocess("")))
        self.assertTrue(pd.isna(preprocess(None)))
        self.assertTrue(pd.isna(preprocess("   ")))

    def test_preprocess_non_string_types(self):
        self.assertTrue(pd.isna(preprocess(123)))
        self.assertTrue(pd.isna(preprocess(45.67)))
        self.assertTrue(pd.isna(preprocess(["list"])))
        self.assertTrue(pd.isna(preprocess(object())))

    def test_filter_english_sentences_valid(self):
        text = "This application is exceptionally useful and well designed. It works properly."
        res = filter_english_sentences(text)
        self.assertIsInstance(res, str)
        self.assertIn("useful", res)

    def test_filter_english_sentences_empty_or_none(self):
        self.assertTrue(pd.isna(filter_english_sentences("")))
        self.assertTrue(pd.isna(filter_english_sentences(None)))

    def test_prepare_cleaned_reviews(self):
        df = pd.DataFrame([
            {"reviewId": "1", "content": "Awesome application, totally love the dark mode!"},
            {"reviewId": "2", "content": "   "},
            {"reviewId": "3", "content": "C'est une très bonne application française."},
        ])
        cleaned_df = prepare_cleaned_reviews(df)
        self.assertFalse(cleaned_df.empty)
        self.assertIn("content_processed", cleaned_df.columns)
        self.assertIn("1", cleaned_df["reviewId"].values)


if __name__ == "__main__":
    unittest.main()
