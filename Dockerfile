# =============================================================================
# Imagery Guardrail & Hybrid Pipeline - Optimized API Dockerfile
# =============================================================================
# 
# Features:
# - Pre-downloaded ML models (no download at runtime)
# - Safetensors format for 4-7x faster weight loading
# - Multi-stage build for smaller final image
# - Parallel model loading enabled by default
#
# Models (standard pip packages only):
# - Text: all-MiniLM-L6-v2 (sentence-transformers)
# - Vision: YOLOE-26n-seg - Unified L3+L4 (replaces YOLOv11n + MobileCLIP2)
#
# Build args:
#   PREDOWNLOAD_MODELS: true/false (default: true)
#
# =============================================================================

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    # Model loading optimization
    GUARDRAIL_PARALLEL_LOADING=true \
    GUARDRAIL_BACKGROUND_LOADING=false \
    ML_MODEL_CACHE_DIR=/app/ml_cache \
    # CLIP temperature calibration (improves F1)
    CLIP_TEMPERATURE=1.2

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# =============================================================================
# Dependencies Stage - Build dependencies separately for caching
# =============================================================================
FROM base AS dependencies

# Copy requirements first for layer caching
COPY requirements.txt .

# Install CPU PyTorch first (for development/CPU workers)
RUN pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.0+cpu \
    torchvision==0.19.0+cpu

# Install remaining dependencies
RUN pip install -r requirements.txt

# =============================================================================
# Model Download Stage (separate for caching)
# =============================================================================
FROM dependencies AS model-downloader

# Build arguments
ARG PREDOWNLOAD_MODELS=true

# Copy only what's needed for model download
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create cache directory
RUN mkdir -p /app/ml_cache

# Pre-download and optimize models during build (if enabled)
# Uses STANDARD configuration (single optimal model)
RUN if [ "${PREDOWNLOAD_MODELS}" = "true" ]; then \
        echo "Pre-downloading ML models (STANDARD configuration)..." && \
        python scripts/setup_models.py \
            --cache-dir /app/ml_cache && \
        echo "Models downloaded successfully!" && \
        du -sh /app/ml_cache; \
    else \
        echo "Skipping model pre-download (PREDOWNLOAD_MODELS=false)"; \
    fi

# =============================================================================
# Final Stage - Minimal runtime image
# =============================================================================
FROM dependencies AS final

# Copy pre-downloaded models from downloader stage
COPY --from=model-downloader /app/ml_cache /app/ml_cache

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/storage

# Expose port
EXPOSE 8000

# Health check - does NOT load models (lazy loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command with parallel loading enabled
# 2 workers for development, scale up in production
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "--timeout", "300", "-b", "0.0.0.0:8000", "src.main:app"]
