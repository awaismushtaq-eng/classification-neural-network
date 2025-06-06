import os
import time

import numpy as np
import pytest

from src.model.inference import ONNXModel, Preprocessor

# Test data
TEST_IMAGE_PATH = "data/images/n01440764_tench.jpeg"
TEST_IMAGE_PATH2 = "data/images/n01667114_mud_turtle.JPEG"
MODEL_PATH = "models/model.onnx"


# Fixture for preprocessor
@pytest.fixture
def preprocessor():
    return Preprocessor(target_size=(224, 224))


# Fixture for model
@pytest.fixture
def model():
    return ONNXModel(MODEL_PATH)


class TestPreprocessor:
    """Tests for the Preprocessor class."""

    def test_init(self, preprocessor):
        """Test initialization of the preprocessor."""
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.mean.shape == (1, 1, 3)
        assert preprocessor.std.shape == (1, 1, 3)

    def test_preprocess(self, preprocessor):
        """Test preprocessing of a single image."""
        # Make sure test image exists
        if not os.path.exists(TEST_IMAGE_PATH):
            pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")

        # Preprocess image
        result = preprocessor.preprocess(TEST_IMAGE_PATH)

        # Check shape and type
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32

        # Check value range (after normalization)
        assert -10 < np.min(result) < 10
        assert -10 < np.max(result) < 10

    def test_preprocess_batch(self, preprocessor):
        """Test preprocessing of multiple images."""
        # Make sure test images exist
        if not os.path.exists(TEST_IMAGE_PATH) or not os.path.exists(TEST_IMAGE_PATH2):
            pytest.skip("Test images not found")

        # Preprocess batch
        result = preprocessor.preprocess_batch([TEST_IMAGE_PATH, TEST_IMAGE_PATH2])

        # Check shape and type
        assert result.shape == (2, 3, 224, 224)
        assert result.dtype == np.float32

    def test_preprocess_error(self, preprocessor):
        """Test error handling for preprocessing."""
        with pytest.raises(FileNotFoundError):
            preprocessor.preprocess("nonexistent_image.jpg")


class TestONNXModel:
    """Tests for the ONNXModel class."""

    def test_init(self, model):
        """Test initialization of the model."""
        assert model.session is not None
        assert model.input_name == "input"
        assert model.output_name == "output"
        assert model.preprocessor is not None

    def test_init_error(self):
        """Test error handling for model initialization."""
        with pytest.raises(FileNotFoundError):
            ONNXModel("nonexistent_model.onnx")

    def test_predict(self, model):
        """Test prediction on a single image."""
        # Make sure test image exists
        if not os.path.exists(TEST_IMAGE_PATH):
            pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")

        # Run prediction
        result = model.predict(TEST_IMAGE_PATH)

        # Check result type
        assert isinstance(result, int)
        assert result == 0  # Tench is class 0

    def test_predict_batch(self, model):
        """Test prediction on multiple images."""
        # Make sure test images exist
        if not os.path.exists(TEST_IMAGE_PATH) or not os.path.exists(TEST_IMAGE_PATH2):
            pytest.skip("Test images not found")

        # Run batch prediction
        results = model.predict_batch([TEST_IMAGE_PATH, TEST_IMAGE_PATH2])

        # Check results
        assert len(results) == 2
        assert isinstance(results[0], int)
        assert isinstance(results[1], int)
        assert results[0] == 0  # Tench is class 0
        assert results[1] == 35  # Mud turtle is class 35

    def test_predict_with_confidence(self, model):
        """Test prediction with confidence scores."""
        # Make sure test image exists
        if not os.path.exists(TEST_IMAGE_PATH):
            pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")

        # Run prediction with confidence
        results = model.predict_with_confidence(TEST_IMAGE_PATH, top_k=5)

        # Check results
        assert len(results) == 5
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2
        assert isinstance(results[0][0], int)
        assert isinstance(results[0][1], float)
        assert results[0][0] == 0  # Tench should be the top prediction
        assert 0 <= results[0][1] <= 1  # Confidence should be between 0 and 1

    def test_performance(self, model):
        """Test model performance."""
        # Make sure test image exists
        if not os.path.exists(TEST_IMAGE_PATH):
            pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")

        # Measure single prediction time
        start_time = time.time()
        model.predict(TEST_IMAGE_PATH)
        single_time = time.time() - start_time

        # Measure batch prediction time
        start_time = time.time()
        model.predict_batch([TEST_IMAGE_PATH, TEST_IMAGE_PATH])
        batch_time = time.time() - start_time

        assert (batch_time / single_time) < 4.0  # Batch should not be unreasonably slow

        # Check that predictions are consistent
        single_results = [
            model.predict(TEST_IMAGE_PATH),
            model.predict(TEST_IMAGE_PATH),
        ]
        batch_results = model.predict_batch([TEST_IMAGE_PATH, TEST_IMAGE_PATH])

        assert single_results == batch_results
