# Image Classification Model Deployment on Cerebrium

This project implements an end-to-end MLOps pipeline for deploying a PyTorch image classification model on Cerebrium's serverless GPU platform. The model is converted to ONNX format for optimized inference.

## Project Structure

```
├── src/                    # Source code package
│   ├── model/              # Model implementation
│   │   └── inference.py    # Preprocessor and ONNXModel classes
│   ├── api/                # API implementation
│   │   └── server.py       # FastAPI server implementation
│   └── utils/              # Utility functions
│       └── converter.py    # PyTorch to ONNX converter
├── scripts/                # Executable scripts
│   ├── convert_to_onnx.py  # Script to convert PyTorch model to ONNX
│   └── test_server.py      # Script to test the deployed model
├── tests/                  # Test suite
│   ├── test_model.py       # Tests for model implementation
│   └── test_api.py         # Tests for API implementation
├── models/                 # Model files
│   └── model.onnx          # Converted ONNX model (generated)
├── data/                   # Data files
│   └── images/             # Test images
│       ├── n01440764_tench.jpeg
│       └── n01667114_mud_turtle.JPEG
├── logs/                   # Log files
├── main.py                 # Main entry point
├── Dockerfile              # Docker configuration for Cerebrium deployment
├── cerebrium.toml          # Cerebrium deployment configuration
└── requirements.txt        # Project dependencies
```

## Setup and Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- ONNX Runtime

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Converting the PyTorch Model to ONNX

```bash
python scripts/convert_to_onnx.py --weights pytorch_model_weights.pth --output models/model.onnx --test-image data/images/n01440764_tench.jpeg
```

Options:
- `--weights`: Path to the PyTorch model weights (required)
- `--output`: Path to save the ONNX model (default: models/model.onnx)
- `--test-image`: Path to an image for validation (optional)
- `--dynamic-axes`: Enable dynamic axes for batch size (optional)

### Running Tests

```bash
python -m pytest tests/
```

### Running the FastAPI Server Locally

```bash
python main.py
```

Or directly with uvicorn:

```bash
uvicorn src.api.server:app --reload
```

The API will be available at http://localhost:8192 with the following endpoints:
- `GET /health`: Health check endpoint
- `POST /predict`: Predict the class of an uploaded image
- `POST /predict/top`: Get top-k predictions for an uploaded image

### Deploying to Cerebrium

1. Make sure you have the Cerebrium CLI installed and configured:
   ```bash
   pip install cerebrium
   cerebrium login
   ```

2. Deploy the model:
   ```bash
   cerebrium deploy
   ```

3. The deployment will use the configuration specified in `cerebrium.toml`.

### Testing the Deployed Model

Once deployed, you can test the model using the `scripts/test_server.py` script:

```bash
python scripts/test_server.py --url https://your-deployment-url.cerebrium.ai --api-key your-api-key
```

Options:
- `--url`: Base URL of the deployed model API (required)
- `--api-key`: API key for authentication (optional)
- `--images`: Paths to test images (optional)

## Model Details

The model is a ResNet-based image classifier trained on the ImageNet dataset, capable of classifying images into 1000 different categories.

### Preprocessing Steps

1. Resize image to 224x224 pixels
2. Convert to RGB format if needed
3. Normalize with ImageNet mean and standard deviation:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

### API Response Format

#### /predict Endpoint

```json
{
  "class_id": 0,
  "confidence": 0.8267,
  "processing_time": 0.1234
}
```

#### /predict/top Endpoint

```json
{
  "predictions": [
    {"class_id": 0, "confidence": 0.8267},
    {"class_id": 1, "confidence": 0.0261},
    {"class_id": 29, "confidence": 0.0207},
    {"class_id": 397, "confidence": 0.0182},
    {"class_id": 389, "confidence": 0.0084}
  ],
  "processing_time": 0.1234
}
```

## Performance Considerations

- The model is converted to ONNX format for optimized inference
- The API supports batch processing for improved throughput
- The Cerebrium deployment is configured with GPU support for faster inference

## Security Considerations

- API key authentication is implemented for all prediction endpoints
- The API key can be set via the `API_KEY` environment variable
- Temporary files are properly cleaned up after processing

## Future Improvements

- Implement caching for improved performance
- Add model versioning and A/B testing capabilities
- Implement monitoring and logging for model performance
- Add support for custom preprocessing options
