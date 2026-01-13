"""
Imagery Guardrail & Hybrid Pipeline - Main Application

Production-ready FastAPI application with:
- API versioning (/api/v1/)
- Structured logging with structlog
- Prometheus metrics
- Global exception handling
- Storage abstraction (local + Azure ready)
- Optimized model loading (parallel + background)
"""

import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import redis.asyncio as redis

from src.core.config import settings
from src.core.database import create_db_and_tables
from src.core.logging import setup_logging, get_logger
from src.core.exceptions import register_exception_handlers, GlobalExceptionMiddleware
from src.core.metrics import set_app_info, http_requests_total, http_request_duration_seconds
from src.api.v1 import api_v1_router
from src.api.dependencies import (
    preload_models,
    preload_models_async,
    start_background_model_loading,
    wait_for_models_ready,
    get_model_loading_status,
    USE_BACKGROUND_LOADING,
)

import time


# =============================================================================
# Initialize Logging
# =============================================================================
setup_logging(
    log_level=settings.LOG_LEVEL,
    json_format=settings.LOG_FORMAT_JSON
)
logger = get_logger(__name__)


# =============================================================================
# Lifespan Handler
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown.
    
    Supports two loading modes:
    1. Background loading: Start loading immediately, app starts accepting requests
    2. Blocking loading: Wait for models to load before accepting requests
    """
    startup_start = time.time()
    
    logger.info(
        "application_starting",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT
    )
    
    # Initialize database
    await create_db_and_tables()
    logger.info("database_initialized")
    
    # Initialize Redis connection
    app.state.redis = redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    logger.info("redis_connected", url=settings.REDIS_URL)
    
    # Set Prometheus app info
    set_app_info(
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT
    )
    
    # Pre-load ML models
    logger.info("preloading_ml_models", background=USE_BACKGROUND_LOADING)
    
    if USE_BACKGROUND_LOADING:
        # Start loading in background (non-blocking)
        # App will start immediately, models load in background
        await start_background_model_loading()
        logger.info("ml_models_loading_in_background")
    else:
        # Load models synchronously (blocking)
        # App waits for models before accepting requests
        try:
            success = await preload_models_async()
            if success:
                logger.info("ml_models_preloaded")
            else:
                logger.warning("ml_models_preload_failed")
        except Exception as e:
            logger.warning("ml_models_preload_failed", error=str(e))
    
    startup_time = time.time() - startup_start
    logger.info("application_ready", startup_time_seconds=startup_time)
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    await app.state.redis.aclose()
    logger.info("application_shutdown_complete")


# =============================================================================
# Create FastAPI Application
# =============================================================================
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    Production-ready Imagery Processing Pipeline with:
    
    - **Guardrail Validation**: CLIP-based NSFW/Food detection, injection checks
    - **Background Removal**: Rembg integration
    - **4K Upscaling**: RealESRGAN with tiled processing
    - **Smart Placement**: Nano Banana API ($0.08/image)
    - **Observability**: Structured logging, Prometheus metrics
    
    ## API Versioning
    
    All endpoints are versioned under `/api/v1/`
    
    ## Pipeline Stages
    
    1. **Guardrail** (sync) - Content validation
    2. **Rembg** (async) - Background removal
    3. **RealESRGAN** (async) - 4K upscaling
    4. **Nano Banana** (async) - Smart placement
    
    ## Model Loading
    
    Models are loaded with parallel loading for 50-60% faster startup.
    Set `GUARDRAIL_BACKGROUND_LOADING=true` to start accepting requests
    immediately while models load in background.
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# =============================================================================
# Middleware
# =============================================================================

# CORS
cors_origins = settings.CORS_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Track request timing for metrics."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Record metrics
    http_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    # Add timing header
    response.headers["X-Process-Time"] = str(duration)
    
    return response


# =============================================================================
# Model Loading Middleware (for background loading mode)
# =============================================================================

@app.middleware("http")
async def ensure_models_ready(request: Request, call_next):
    """Ensure models are loaded before processing guardrail requests.
    
    Only applies to guardrail endpoints when using background loading.
    Other endpoints work immediately.
    """
    # Skip for non-guardrail endpoints
    if not request.url.path.startswith("/api/v1/guardrail") and not request.url.path.startswith("/v1/guardrail"):
        return await call_next(request)
    
    # Check if models are loading
    status = get_model_loading_status()
    
    if status["is_loading"]:
        # Wait for models (with timeout)
        logger.info("waiting_for_models", path=request.url.path)
        ready = await wait_for_models_ready(timeout=120.0)
        
        if not ready:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service initializing",
                    "message": "ML models are still loading. Please retry in a moment.",
                    "retry_after": 30
                },
                headers={"Retry-After": "30"}
            )
    
    return await call_next(request)


# =============================================================================
# Register Exception Handlers
# =============================================================================
register_exception_handlers(app)


# =============================================================================
# Include API Routers
# =============================================================================
app.include_router(api_v1_router)

# Legacy route compatibility (redirect to v1)
from src.api.v1.guardrail import router as legacy_guardrail_router
app.include_router(
    legacy_guardrail_router,
    prefix="/v1/guardrail",
    tags=["guardrail-legacy"],
    deprecated=True
)


# =============================================================================
# Static Files & UI
# =============================================================================

# Mount storage directory for serving processed images
storage_dir = os.path.join(os.path.dirname(__file__), "..", "data", "storage")
if os.path.exists(storage_dir):
    app.mount("/static/storage", StaticFiles(directory=storage_dir), name="storage")

# Mount frontend UI
ui_dist_dir = os.path.join(os.path.dirname(__file__), "..", "food-guard-ui", "dist")
if os.path.exists(ui_dist_dir):
    app.mount("/ui", StaticFiles(directory=ui_dist_dir, html=True), name="ui")


# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/api/docs",
        "ui": "/ui",
        "api_v1": "/api/v1",
        "metrics": "/api/v1/metrics"
    }


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION
    }


@app.get("/ready", tags=["health"])
async def ready(request: Request):
    """Readiness check - verifies dependencies are available."""
    checks = {
        "redis": False,
        "database": False,
        "ml_models": False
    }
    
    # Check Redis
    try:
        await request.app.state.redis.ping()
        checks["redis"] = True
    except Exception:
        pass
    
    # Check database (simple check)
    try:
        from src.core.database import engine
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        pass
    
    # Check ML models
    model_status = get_model_loading_status()
    checks["ml_models"] = model_status["models_loaded"]
    
    all_ready = all(checks.values())
    
    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "ready": all_ready,
            "checks": checks,
            "model_status": {
                "loaded": model_status["models_loaded"],
                "loading": model_status["is_loading"],
            }
        }
    )


@app.get("/models/status", tags=["health"])
async def models_status():
    """Get detailed ML model loading status."""
    return get_model_loading_status()


# =============================================================================
# Development Server
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
