import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

# Create test client
client = TestClient(app)


class MockModel:
    def predict(self, image_path):
        return 0

    def predict_with_confidence(self, image_path, top_k=5):
        return [(0, 0.9), (1, 0.05), (2, 0.03), (3, 0.01), (4, 0.01)]


@pytest.fixture
def mock_get_model(monkeypatch):
    mock_model = MockModel()

    def mock_get_model_func():
        return mock_model

    monkeypatch.setattr("src.api.server.get_model", mock_get_model_func)
    return mock_model


# Mock the verify_api_key dependency
@pytest.fixture
def mock_verify_api_key(monkeypatch):
    def mock_verify_func(x_api_key=None):
        return x_api_key

    monkeypatch.setattr("src.api.server.verify_api_key", mock_verify_func)


class TestAPI:
    """Tests for the API endpoints."""

    def test_health(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

    def test_predict(self, mock_get_model, mock_verify_api_key):
        """Test prediction endpoint."""
        # Create a test image
        test_image_path = "tests/test_image.jpg"
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)

        # Patch both predict methods to return fixed values
        with patch("src.model.inference.ONNXModel.predict", return_value=0), patch(
            "src.model.inference.ONNXModel.predict_with_confidence",
            return_value=[(0, 0.9)],
        ):
            try:
                # Create a simple test image
                import numpy as np
                from PIL import Image

                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                img.save(test_image_path)

                # Test the endpoint
                with open(test_image_path, "rb") as f:
                    response = client.post(
                        "/predict", files={"file": ("test_image.jpg", f, "image/jpeg")}
                    )

                # Check response
                assert response.status_code == 200
                data = response.json()
                assert "class_id" in data
                assert "confidence" in data
                assert "processing_time" in data
                assert data["class_id"] == 0
                assert data["confidence"] == 0.9
                assert isinstance(data["processing_time"], float)

            finally:
                # Clean up
                if os.path.exists(test_image_path):
                    os.remove(test_image_path)

    def test_predict_top(self, mock_get_model, mock_verify_api_key):
        """Test top predictions endpoint."""
        # Create a test image
        test_image_path = "tests/test_image.jpg"
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)

        # Mock the predict_with_confidence method
        mock_results = [(0, 0.9), (1, 0.05), (2, 0.03), (3, 0.01), (4, 0.01)]
        with patch(
            "src.model.inference.ONNXModel.predict_with_confidence",
            return_value=mock_results,
        ):
            try:
                # Create a simple test image
                import numpy as np
                from PIL import Image

                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                img.save(test_image_path)

                # Test the endpoint
                with open(test_image_path, "rb") as f:
                    response = client.post(
                        "/predict/top",
                        files={"file": ("test_image.jpg", f, "image/jpeg")},
                    )

                # Check response
                assert response.status_code == 200
                data = response.json()
                assert "predictions" in data
                assert "processing_time" in data
                assert len(data["predictions"]) == 5
                assert data["predictions"][0]["class_id"] == 0
                assert data["predictions"][0]["confidence"] == 0.9
                assert isinstance(data["processing_time"], float)

            finally:
                # Clean up
                if os.path.exists(test_image_path):
                    os.remove(test_image_path)

    def test_api_key_validation(self, mock_get_model):
        """Test API key validation."""
        # Create a test image
        test_image_path = "tests/test_image.jpg"
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)

        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img.save(test_image_path)

        try:
            # Save original API_KEY value
            with patch("src.api.server.API_KEY", "test_key"):
                # Test with invalid API key
                with open(test_image_path, "rb") as f:
                    response = client.post(
                        "/predict",
                        files={"file": ("test_image.jpg", f, "image/jpeg")},
                        headers={"X-API-Key": "invalid_key"},
                    )

                assert response.status_code == 401

                # Test with valid API key
                with open(test_image_path, "rb") as f:
                    response = client.post(
                        "/predict",
                        files={"file": ("test_image.jpg", f, "image/jpeg")},
                        headers={"X-API-Key": "test_key"},
                    )

                assert response.status_code == 200

        finally:
            # Clean up
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
