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

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check built into container
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start server
# Port is always 8000 inside the container
# docker-compose maps 5001:8000 on macOS, 8000:8000 on Windows/Linux
ENV PYTHONPATH=/app
CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
