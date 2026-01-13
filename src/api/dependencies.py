"""
FastAPI Dependencies for Guardrail Service

Provides dependency injection for:
- ML Repository (singleton with parallel loading)
- Text/Image Guardrail Services (singleton)
- Cache Repository (per-request)
- Log Repository (per-request with session)
- Feedback Repository (per-request with session)

Optimized Features:
- Parallel model loading (50-60% faster startup)
- Background loading support
- Configurable model sizes
"""

import asyncio
import logging
import os

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database import get_session
from src.engines.guardrail.repositories import (
    MLRepository,
    CacheRepository,
    LogRepository,
    FeedbackRepository,
    ConfigVariantRepository,
    ModelSize,
)
from src.engines.guardrail.services import (
    TextGuardrailService,
    ImageGuardrailService,
    GuardrailService
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Always use STANDARD model configuration - single model, no options
MODEL_SIZE = ModelSize.STANDARD

# Use parallel loading by default (can be disabled via env)
USE_PARALLEL_LOADING = os.environ.get("GUARDRAIL_PARALLEL_LOADING", "true").lower() == "true"

# Background loading (start loading immediately, don't block startup)
USE_BACKGROUND_LOADING = os.environ.get("GUARDRAIL_BACKGROUND_LOADING", "false").lower() == "true"


# =============================================================================
# Global Singletons - ML models are expensive, only load once per process
# =============================================================================

_ml_repo = MLRepository.get_instance(settings.ML_MODEL_CACHE_DIR)
_text_service = TextGuardrailService(_ml_repo)
_image_service = ImageGuardrailService(_ml_repo)

# Track if models are being loaded
_models_loading = False
_models_load_task = None


# =============================================================================
# Model Preloading Functions
# =============================================================================

def preload_models() -> bool:
    """Pre-load ML models at startup (parallel or sequential).
    
    Call this in the FastAPI startup event.
    Uses parallel loading for 50-60% faster startup.
    """
    global _models_loading
    _models_loading = True
    
    logger.info(f"Pre-loading ML models (parallel={USE_PARALLEL_LOADING}, size={MODEL_SIZE.value})...")
    
    try:
        # Use parallel loading for faster startup
        if USE_PARALLEL_LOADING:
            success = _ml_repo.preload_all_models_parallel()
        else:
            success = _ml_repo.preload_all_models()
        
        if success:
            # Pre-compute embeddings after models are loaded
            _text_service._precompute_food_embeddings()
            logger.info("Food embeddings computed")
            
            _text_service._precompute_injection_embeddings()
            logger.info("Injection embeddings computed")
            
            # Log loading stats
            stats = _ml_repo.get_loading_stats()
            logger.info(
                f"All ML models pre-loaded successfully | "
                f"Total time: {stats['total_loading_time']:.1f}s | "
                f"Device: {stats['device']} | "
                f"Model size: {stats['model_size']}"
            )
        
        _models_loading = False
        return success
        
    except Exception as e:
        logger.warning(f"Model pre-loading failed (will load on first request): {e}")
        _models_loading = False
        return False


async def preload_models_async() -> bool:
    """Async version of model preloading.
    
    Runs model loading in thread pool to avoid blocking.
    """
    return await asyncio.to_thread(preload_models)


async def start_background_model_loading():
    """Start loading models in background (non-blocking).
    
    This allows the app to start accepting requests immediately
    while models load in the background.
    """
    global _models_load_task
    
    logger.info("Starting background model loading...")
    _models_load_task = asyncio.create_task(preload_models_async())
    return _models_load_task


async def wait_for_models_ready(timeout: float = 180.0) -> bool:
    """Wait for models to be ready.
    
    Call this before handling requests that need ML models.
    Returns True if models are loaded, False if timed out or failed.
    """
    global _models_load_task
    
    # If models already loaded, return immediately
    if _ml_repo.models_loaded:
        return True
    
    # If loading in background, wait for it
    if _models_load_task is not None:
        try:
            result = await asyncio.wait_for(_models_load_task, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Model loading timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    # No loading started, load now
    return await preload_models_async()


def get_model_loading_status() -> dict:
    """Get current model loading status."""
    return {
        "models_loaded": _ml_repo.models_loaded,
        "is_loading": _models_loading or (_models_load_task is not None and not _models_load_task.done()),
        "loading_stats": _ml_repo.get_loading_stats() if _ml_repo.models_loaded else None,
        "config": {
            "model_size": MODEL_SIZE.value,
            "parallel_loading": USE_PARALLEL_LOADING,
            "background_loading": USE_BACKGROUND_LOADING,
        }
    }


# =============================================================================
# ML Repository
# =============================================================================

def get_ml_repo() -> MLRepository:
    """Returns singleton ML repository."""
    return _ml_repo


# =============================================================================
# Cache Repository
# =============================================================================

def get_cache_repo(request: Request) -> CacheRepository:
    """Returns cache repository with Redis client from app state."""
    return CacheRepository(request.app.state.redis, settings.CACHE_TTL_SECONDS)


# =============================================================================
# Database Repositories
# =============================================================================

def get_log_repo(session: AsyncSession = Depends(get_session)) -> LogRepository:
    """Returns log repository with async session."""
    return LogRepository(session)


def get_feedback_repo(session: AsyncSession = Depends(get_session)) -> FeedbackRepository:
    """Returns feedback repository with async session."""
    return FeedbackRepository(session)


def get_config_variant_repo(session: AsyncSession = Depends(get_session)) -> ConfigVariantRepository:
    """Returns config variant repository with async session."""
    return ConfigVariantRepository(session)


# =============================================================================
# Guardrail Services
# =============================================================================

def get_text_service() -> TextGuardrailService:
    """Returns singleton text guardrail service."""
    return _text_service


def get_image_service() -> ImageGuardrailService:
    """Returns singleton image guardrail service."""
    return _image_service


def get_guardrail_service(
    cache_repo: CacheRepository = Depends(get_cache_repo),
    log_repo: LogRepository = Depends(get_log_repo),
) -> GuardrailService:
    """Returns GuardrailService with singleton ML services.
    
    The GuardrailService is created per-request with:
    - Singleton ML repo, text service, and image service
    - Per-request cache repo and log repo
    """
    return GuardrailService(
        ml_repo=_ml_repo,
        cache_repo=cache_repo,
        log_repo=log_repo,
        text_service=_text_service,
        image_service=_image_service
    )
