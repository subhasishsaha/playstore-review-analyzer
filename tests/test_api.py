"""
Integration tests for FastAPI endpoints and static frontend serving.
"""
import unittest
from fastapi.testclient import TestClient
from src.api import app


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "online")
        self.assertIn("ollama_model", data)

    def test_models_endpoint(self):
        from unittest.mock import patch
        with patch("src.suggestions.get_installed_models", return_value=["qwen3.5:0.8b", "llama3.2:1b"]):
            response = self.client.get("/api/models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("default_model", data)
            self.assertIn("installed_models", data)
            self.assertIn("qwen3.5:0.8b", data["installed_models"])

    def test_models_endpoint_empty_installed(self):
        from unittest.mock import patch
        with patch("src.suggestions.get_installed_models", return_value=[]):
            response = self.client.get("/api/models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("default_model", data)
            self.assertEqual(data["installed_models"], [])


    def test_favicon_endpoint(self):
        response = self.client.get("/favicon.ico")
        self.assertEqual(response.status_code, 200)
        self.assertIn("image/svg+xml", response.headers.get("content-type", ""))

    def test_root_serves_frontend(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("Play Store Review Analyzer", response.text)

    def test_analyze_empty_body(self):
        response = self.client.post("/api/analyze", json={})
        self.assertEqual(response.status_code, 422)

    def test_analyze_empty_app_id(self):
        response = self.client.post("/api/analyze", json={"app_id": ""})
        self.assertEqual(response.status_code, 400)

    def test_analyze_success_response(self):
        from unittest.mock import patch
        mock_payload = {
            "status": "success",
            "app_id": "com.test.app",
            "total": 1,
            "summary": {"total": 1, "positive_count": 1, "negative_count": 0},
            "reviews": [{"reviewId": "1", "content": "Great app", "sentiment": "Positive"}],
            "suggestions_markdown": "None"
        }
        with patch("src.api.analyze_reviews_data", return_value=mock_payload) as mock_analyze:
            response = self.client.post("/api/analyze", json={
                "app_id": "com.test.app",
                "count": 20,
                "ollama_model": "qwen2.5:0.5b"
            })
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["app_id"], "com.test.app")
            mock_analyze.assert_called_once_with(
                app_id="com.test.app",
                sort_order="Most Relevant",
                count=20,
                ollama_model="qwen2.5:0.5b"
            )

    def test_analyze_server_error_500(self):
        from unittest.mock import patch
        with patch("src.api.analyze_reviews_data", side_effect=RuntimeError("Scraper connection timeout")):
            response = self.client.post("/api/analyze", json={"app_id": "com.test.app", "count": 20})
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("Analysis failed: Scraper connection timeout", data["detail"])


if __name__ == "__main__":
    unittest.main()
