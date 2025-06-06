import argparse
import json
import os
import time
from typing import Any, Optional

import requests


def test_health(base_url: str) -> dict[str, Any]:
    """
    Test the health check endpoint.

    Args:
        base_url: Base URL of the API

    Returns:
        Response from the health check endpoint
    """
    url = f"{base_url}/health"
    start_time = time.time()
    response = requests.get(url)
    elapsed_time = time.time() - start_time

    if response.status_code == 200:
        print(f"Health check response ({elapsed_time:.2f}s): {response.json()}")
        return response.json()
    else:
        print(f"Error: Health check failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return {"status": "ERROR", "code": response.status_code}


def test_predict(
    base_url: str, image_path: str, api_key: Optional[str] = None
) -> dict[str, Any]:
    """
    Test the prediction endpoint.

    Args:
        base_url: Base URL of the API
        image_path: Path to the image file
        api_key: Optional API key for authentication

    Returns:
        Response from the prediction endpoint
    """
    url = f"{base_url}/predict"

    # Prepare headers
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    # Prepare file
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}

        # Send request
        start_time = time.time()
        response = requests.post(url, files=files, headers=headers)
        elapsed_time = time.time() - start_time

    if response.status_code == 200:
        print(
            f"Prediction ({elapsed_time:.2f}s): {json.dumps(response.json(), indent=2)}"
        )
        return response.json()
    else:
        print(f"Error: Prediction failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return {"status": "ERROR", "code": response.status_code}


def test_predict_top(
    base_url: str, image_path: str, top_k: int = 5, api_key: Optional[str] = None
) -> dict[str, Any]:
    """
    Test the top predictions endpoint.

    Args:
        base_url: Base URL of the API
        image_path: Path to the image file
        top_k: Number of top predictions to return
        api_key: Optional API key for authentication

    Returns:
        Response from the top predictions endpoint
    """
    url = f"{base_url}/predict/top"

    # Prepare headers
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    # Prepare file and parameters
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        params = {"top_k": top_k}

        # Send request
        start_time = time.time()
        response = requests.post(url, files=files, params=params, headers=headers)
        elapsed_time = time.time() - start_time

    if response.status_code == 200:
        print(
            f"Top predictions ({elapsed_time:.2f}s): {json.dumps(response.json(), indent=2)}"
        )
        return response.json()
    else:
        print(f"Error: Top predictions failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return {"status": "ERROR", "code": response.status_code}


def main():
    """Main function for the test script."""
    parser = argparse.ArgumentParser(description="Test the deployed model API")
    parser.add_argument("--url", required=True, help="Base URL of the API")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--images",
        nargs="+",
        default=[
            "data/images/n01440764_tench.jpeg",
            "data/images/n01667114_mud_turtle.JPEG",
        ],
        help="Paths to test images",
    )

    args = parser.parse_args()

    # Test health check
    print("\n=== Testing Health Check ===")
    test_health(args.url)

    # Test prediction endpoint
    print("\n=== Testing Prediction Endpoint ===")
    for image_path in args.images:
        if os.path.exists(image_path):
            print(f"\nTesting with image: {os.path.basename(image_path)}")
            test_predict(args.url, image_path, args.api_key)
        else:
            print(f"Warning: Image not found: {image_path}")

    # Test top predictions endpoint
    print("\n=== Testing Top Predictions Endpoint ===")
    for image_path in args.images:
        if os.path.exists(image_path):
            print(f"\nTesting with image: {os.path.basename(image_path)}")
            test_predict_top(args.url, image_path, api_key=args.api_key)
        else:
            print(f"Warning: Image not found: {image_path}")


if __name__ == "__main__":
    main()
