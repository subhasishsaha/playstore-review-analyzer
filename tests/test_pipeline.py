"""
Unit tests for pipeline orchestration utilities and data sanitization.
"""
import unittest
import numpy as np
import pandas as pd
from src.pipeline import _sanitize_for_json, analyze_reviews_data


class TestPipeline(unittest.TestCase):
    def test_sanitize_scalars(self):
        self.assertEqual(_sanitize_for_json(10), 10)
        self.assertEqual(_sanitize_for_json(np.int64(42)), 42)
        self.assertEqual(_sanitize_for_json(3.14), 3.14)
        self.assertIsNone(_sanitize_for_json(float("nan")))
        self.assertIsNone(_sanitize_for_json(np.nan))
        self.assertIsNone(_sanitize_for_json(None))
        self.assertEqual(_sanitize_for_json("hello"), "hello")

    def test_sanitize_collections(self):
        data = [np.int64(1), np.float64(2.5), np.nan, "text"]
        sanitized = _sanitize_for_json(data)
        self.assertEqual(sanitized, [1, 2.5, None, "text"])

    def test_sanitize_nested_dict(self):
        nested = {
            "a": np.int32(100),
            "b": [np.nan, {"c": pd.Timestamp("2026-01-01 00:00:00")}]
        }
        sanitized = _sanitize_for_json(nested)
        self.assertEqual(sanitized["a"], 100)
        self.assertIsNone(sanitized["b"][0])
        self.assertIn("2026-01-01", sanitized["b"][1]["c"])

    def test_sanitize_numpy_and_series(self):
        arr = np.array([1, 2, 3])
        sanitized_arr = _sanitize_for_json(arr)
        self.assertEqual(sanitized_arr, [1, 2, 3])

        series = pd.Series([np.nan, 42.0])
        sanitized_series = _sanitize_for_json(series)
        self.assertEqual(sanitized_series, [None, 42.0])

    def test_sanitize_datetime_and_dates(self):
        import datetime
        now = datetime.datetime(2026, 1, 15, 12, 0, 0)
        today = datetime.date(2026, 1, 15)
        self.assertIn("2026-01-15", _sanitize_for_json(now))
        self.assertIn("2026-01-15", _sanitize_for_json(today))

    def test_sanitize_booleans(self):
        self.assertIs(_sanitize_for_json(True), True)
        self.assertIs(_sanitize_for_json(False), False)
        self.assertIs(_sanitize_for_json(np.bool_(True)), True)

    def test_sanitize_json_dumpable(self):
        import json
        import datetime
        data = {
            "arr": np.array([1, 2]),
            "series": pd.Series([np.nan, 3.5]),
            "timestamp": pd.Timestamp("2026-06-01"),
            "datetime": datetime.datetime.now(),
            "date": datetime.date.today(),
            "flag": np.bool_(False),
            "float_nan": np.nan,
        }
        sanitized = _sanitize_for_json(data)
        # Must serialize cleanly via standard json.dumps
        serialized = json.dumps(sanitized)
        self.assertIsInstance(serialized, str)

    def test_analyze_reviews_data_empty_app_id(self):
        with self.assertRaises(ValueError):
            analyze_reviews_data("")
        with self.assertRaises(ValueError):
            analyze_reviews_data("   ")


if __name__ == "__main__":
    unittest.main()
