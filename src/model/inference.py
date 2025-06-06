import os
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


class Preprocessor:
    """
    Class for preprocessing images before inference.
    Implements the same preprocessing steps as the PyTorch model.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the preprocessor with target image size.

        Args:
            target_size: Target size for the image (height, width)
        """
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess an image from a file path.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image as numpy array with shape (1, 3, height, width)

        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image cannot be processed
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Open image and convert to RGB if needed
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to target size
            img = img.resize(self.target_size, Image.BILINEAR)

            # Convert to numpy array and normalize
            img_np = np.array(img).astype(np.float32) / 255.0

            # Apply mean and std normalization
            img_np = (img_np - self.mean) / self.std

            # Transpose from HWC to CHW format (height, width, channels) -> (channels, height, width)
            img_np = img_np.transpose(2, 0, 1)

            # Add batch dimension
            img_np = np.expand_dims(img_np, axis=0)

            # Ensure float32 data type
            img_np = img_np.astype(np.float32)

            return img_np

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocess multiple images from file paths.

        Args:
            image_paths: List of paths to image files

        Returns:
            Batch of preprocessed images as numpy array with shape (batch_size, 3, height, width)
        """
        batch = [self.preprocess(path) for path in image_paths]
        return np.vstack(batch)


class ONNXModel:
    """
    Class for loading and running inference with an ONNX model.
    """

    def __init__(self, model_path: str, preprocessor: Optional[Preprocessor] = None):
        """
        Initialize the ONNX model.

        Args:
            model_path: Path to the ONNX model file
            preprocessor: Optional preprocessor instance. If not provided, a default one will be created.

        Raises:
            FileNotFoundError: If the model file does not exist
            RuntimeError: If the model cannot be loaded
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            # Get input shape from model
            self.input_shape = self.session.get_inputs()[0].shape

            # Create preprocessor if not provided
            if preprocessor is None:
                target_size = (
                    (self.input_shape[2], self.input_shape[3])
                    if len(self.input_shape) == 4
                    else (224, 224)
                )
                self.preprocessor = Preprocessor(target_size=target_size)
            else:
                self.preprocessor = preprocessor

        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

    def predict(self, image_path: str) -> int:
        """
        Run inference on a single image and return the predicted class ID.

        Args:
            image_path: Path to the image file

        Returns:
            Predicted class ID (int)
        """
        # Preprocess the image
        input_data = self.preprocessor.preprocess(image_path)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})

        # Get the predicted class ID
        predicted_class = np.argmax(outputs[0][0])

        return int(predicted_class)

    def predict_batch(self, image_paths: List[str]) -> List[int]:
        """
        Run inference on multiple images and return the predicted class IDs.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of predicted class IDs
        """
        # Process each image individually since our model has fixed batch size
        results = []
        for image_path in image_paths:
            predicted_class = self.predict(image_path)
            results.append(predicted_class)

        return results

    def predict_with_confidence(
        self, image_path: str, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Run inference on a single image and return the top-k predictions with confidence scores.

        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            List of tuples (class_id, confidence_score)
        """
        # Preprocess the image
        input_data = self.preprocessor.preprocess(image_path)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})

        # Get the top-k predictions
        output = outputs[0][0]

        # Apply softmax to get probabilities
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()

        # Get top-k indices and probabilities
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_probabilities = probabilities[top_indices]

        return [
            (int(idx), float(prob)) for idx, prob in zip(top_indices, top_probabilities)
        ]
