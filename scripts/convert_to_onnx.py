import argparse
import os
import sys

from src.utils.converter import convert_to_onnx, load_pytorch_model, validate_onnx_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main function for the converter script."""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        "--weights", required=True, help="Path to PyTorch model weights"
    )
    parser.add_argument(
        "--output", default="models/model.onnx", help="Path to save ONNX model"
    )
    parser.add_argument("--test-image", help="Path to test image for validation")
    parser.add_argument(
        "--dynamic-axes", action="store_true", help="Enable dynamic axes for batch size"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

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
