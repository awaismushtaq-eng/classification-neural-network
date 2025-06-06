import argparse
import os
import sys
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms


def load_pytorch_model(weights_path: str) -> torch.nn.Module:
    """
    Load the PyTorch model from weights file.

    Args:
        weights_path: Path to the PyTorch weights file

    Returns:
        Loaded PyTorch model
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Import the model definition
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from pytorch_model import Classifier
    except ImportError:
        # Try to import from the project root
        sys.path.append(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        from pytorch_model import Classifier

    # Create model instance
    model = Classifier()

    # Load weights
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    # Set to evaluation mode
    model.eval()

    return model


def convert_to_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    dynamic_axes: bool = True,
) -> str:
    """
    Convert PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        onnx_path: Path to save the ONNX model
        input_shape: Input shape for the model (batch_size, channels, height, width)
        dynamic_axes: Whether to use dynamic axes for batch size

    Returns:
        Path to the saved ONNX model
    """
    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Define dynamic axes if enabled
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
        opset_version=12,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )

    return onnx_path


def validate_onnx_model(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    test_image_path: Optional[str] = None,
) -> bool:
    """
    Validate the ONNX model against the PyTorch model.

    Args:
        pytorch_model: PyTorch model
        onnx_path: Path to the ONNX model
        test_image_path: Path to a test image (optional)

    Returns:
        True if validation is successful
    """
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Get input name
    input_name = ort_session.get_inputs()[0].name

    # Prepare input data
    if test_image_path and os.path.exists(test_image_path):
        print(f"Using test image: {test_image_path}")

        # Load and preprocess image
        img = Image.open(test_image_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_tensor = preprocess(img).unsqueeze(0)

        # PyTorch prediction
        with torch.no_grad():
            pytorch_output = pytorch_model(img_tensor)
            pytorch_prediction = torch.argmax(pytorch_output, dim=1).item()

        # ONNX prediction
        ort_inputs = {input_name: img_tensor.numpy()}
        ort_output = ort_session.run(None, ort_inputs)
        onnx_prediction = np.argmax(ort_output[0], axis=1)[0]

        print(f"PyTorch prediction: {pytorch_prediction}")
        print(f"ONNX prediction: {onnx_prediction}")

        # Check if predictions match
        if pytorch_prediction == onnx_prediction:
            print(
                "✅ Validation successful: PyTorch and ONNX models produce the same prediction"
            )
            return True
        else:
            print(
                "❌ Validation failed: PyTorch and ONNX models produce different predictions"
            )
            return False
    else:
        print("Using random input for validation")

        # Generate random input
        input_tensor = torch.randn(1, 3, 224, 224)

        # PyTorch prediction
        with torch.no_grad():
            pytorch_output = pytorch_model(input_tensor)

        # ONNX prediction
        ort_inputs = {input_name: input_tensor.numpy()}
        ort_output = ort_session.run(None, ort_inputs)

        # Compare outputs
        np_pytorch_output = pytorch_output.numpy()
        np_ort_output = ort_output[0]

        # Check if outputs are close
        if np.allclose(np_pytorch_output, np_ort_output, rtol=1e-03, atol=1e-05):
            print("✅ Validation successful: PyTorch and ONNX model outputs match")
            return True
        else:
            print("❌ Validation failed: PyTorch and ONNX model outputs differ")
            max_diff = np.max(np.abs(np_pytorch_output - np_ort_output))
            print(f"Maximum absolute difference: {max_diff}")
            return False


def main():
    """Main function for the converter script."""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        "--weights", required=True, help="Path to PyTorch model weights"
    )
    parser.add_argument(
        "--output", default="model.onnx", help="Path to save ONNX model"
    )
    parser.add_argument("--test-image", help="Path to test image for validation")
    parser.add_argument(
        "--dynamic-axes", action="store_true", help="Enable dynamic axes for batch size"
    )

    args = parser.parse_args()

    # Load PyTorch model
    print(f"Loading PyTorch model from {args.weights}")
    pytorch_model = load_pytorch_model(args.weights)

    # Convert to ONNX
    print(f"Converting PyTorch model to ONNX format and saving to {args.output}")
    onnx_path = convert_to_onnx(
        pytorch_model, args.output, dynamic_axes=args.dynamic_axes
    )

    print("ONNX model exported successfully")

    # Validate ONNX model
    print("Validating ONNX model...")
    validate_onnx_model(pytorch_model, onnx_path, args.test_image)


if __name__ == "__main__":
    main()
