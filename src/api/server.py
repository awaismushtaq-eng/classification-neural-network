import io
import os
import time
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel

# Import from our package structure
from src.model.inference import ONNXModel

# Configure logger
logger.add("logs/api.log", rotation="10 MB")

# Initialize FastAPI app
app = FastAPI(title="Image Classification API")

# Global variables
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.onnx")
API_KEY = os.environ.get(
    "API_KEY", ""
)  # This would be set in the Cerebrium environment

# Model instance (lazy loading)
_model = None


def get_model() -> ONNXModel:
    """
    Lazy loading of the model to avoid loading it multiple times.
    """
    global _model
    if _model is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        start_time = time.time()
        try:
            _model = ONNXModel(MODEL_PATH)
            logger.info(
                f"Model loaded successfully in {time.time() - start_time:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )
    return _model


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Verify the API key if one is set in the environment.
    """
    if API_KEY and (not x_api_key or x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint.
    """

    class_id: int
    confidence: float
    processing_time: float


class TopPredictionsResponse(BaseModel):
    """
    Response model for top predictions endpoint.
    """

    predictions: list[dict[str, Any]]
    processing_time: float


@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "OK"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict(file: UploadFile = File(...), model: ONNXModel = Depends(get_model)):
    """
    Predict the class of an uploaded image.
    """
    start_time = time.time()

    try:
        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Save the image temporarily
        os.makedirs("temp", exist_ok=True)
        temp_path = f"temp/temp_{int(time.time())}.jpg"
        image.save(temp_path)

        try:
            # Run prediction
            class_id = model.predict(temp_path)

            # Get confidence score for the predicted class
            top_predictions = model.predict_with_confidence(temp_path, top_k=1)
            confidence = top_predictions[0][1]

            processing_time = time.time() - start_time
            logger.info(
                f"Prediction completed in {processing_time:.4f} seconds: class_id={class_id}, confidence={confidence:.4f}"
            )

            return {
                "class_id": class_id,
                "confidence": confidence,
                "processing_time": processing_time,
            }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post(
    "/predict/top",
    response_model=TopPredictionsResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict_top(
    file: UploadFile = File(...), top_k: int = 5, model: ONNXModel = Depends(get_model)
):
    """
    Predict the top-k classes of an uploaded image.
    """
    start_time = time.time()

    try:
        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Save the image temporarily
        os.makedirs("temp", exist_ok=True)
        temp_path = f"temp/temp_{int(time.time())}.jpg"
        image.save(temp_path)

        try:
            # Run prediction
            top_predictions = model.predict_with_confidence(temp_path, top_k=top_k)

            # Format the results
            results = [
                {"class_id": class_id, "confidence": float(confidence)}
                for class_id, confidence in top_predictions
            ]

            processing_time = time.time() - start_time
            logger.info(
                f"Top-{top_k} prediction completed in {processing_time:.4f} seconds"
            )

            return {"predictions": results, "processing_time": processing_time}
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def run_server(host: str = "0.0.0.0", port: int = 8192):
    """
    Run the FastAPI server.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
