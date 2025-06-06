FROM python:3.10-slim

# Install dumb-init and other dependencies including build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    dumb-init \
    build-essential \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p /app/models /app/logs /app/data/images /app/temp

# Copy model and source code
COPY models/ /app/models/
COPY main.py .
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY pyproject.toml .

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/model.onnx

# Expose the port the app will run on
EXPOSE 8192

# Use dumb-init as the entrypoint to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Run FastAPI server
CMD ["python", "-m", "src.api.server"]
