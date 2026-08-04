# Dockerfile
# Dog Breed + OOD Detector — FastAPI inference server
# Owner: Devreet
#
# Build:
#   docker build -t mai202-dog-ood .
#
# Run (macOS -- port 5001, Windows/Linux -- port 8000):
#   docker compose up
#
# Health check:
#   curl http://localhost:5001/health

FROM python:3.11-slim

# System dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch CPU (no CUDA needed for inference container)
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY params.yaml .

# Copy model artifacts and data if they exist
# These are optional at build time -- mounted via volume in docker-compose
COPY models/ ./models/ 2>/dev/null || true
COPY data/processed/class_names.json ./data/processed/class_names.json 2>/dev/null || true
COPY reports/ood/ ./reports/ood/ 2>/dev/null || true

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Health check built into container
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start server
# Port is always 8000 inside the container
# docker-compose maps 5001:8000 on macOS, 8000:8000 on Windows/Linux
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
