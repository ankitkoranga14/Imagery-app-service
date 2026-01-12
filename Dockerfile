# =============================================================================
# Imagery Guardrail & Hybrid Pipeline - API Dockerfile
# =============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install CPU PyTorch first (for development/CPU workers)
RUN pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.0+cpu \
    torchvision==0.19.0+cpu

# Install remaining dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ml_cache data/storage

# Expose port
EXPOSE 8000

# Default command
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "src.main:app"]
