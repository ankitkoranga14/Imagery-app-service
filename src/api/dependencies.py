"""
FastAPI Dependencies for Guardrail Service

Provides dependency injection for:
- ML Repository (singleton)
- Text/Image Guardrail Services (singleton)
- Cache Repository (per-request)
- Log Repository (per-request with session)
- Feedback Repository (per-request with session)
"""

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.config import settings
from src.core.database import get_session
from src.engines.guardrail.repositories import (
    MLRepository,
    CacheRepository,
    LogRepository,
    FeedbackRepository,
    ConfigVariantRepository
)
from src.engines.guardrail.services import (
    TextGuardrailService,
    ImageGuardrailService,
    GuardrailService
)
import logging
import asyncio

logger = logging.getLogger(__name__)


# =============================================================================
# Global Singletons - ML models are expensive, only load once per process
# =============================================================================

_ml_repo = MLRepository(settings.ML_MODEL_CACHE_DIR)
_text_service = TextGuardrailService(_ml_repo)
_image_service = ImageGuardrailService(_ml_repo)


def preload_models():
    """Pre-load ML models at startup to avoid first-request delay.
    
    Call this in the FastAPI startup event.
    """
    logger.info("Pre-loading ML models...")
    try:
        # Load text model
        _ml_repo.get_text_model()
        logger.info("Text model loaded successfully")
        
        # Pre-compute food embeddings
        _text_service._precompute_food_embeddings()
        logger.info("Food embeddings computed")
        
        # Pre-compute injection embeddings for security
        _text_service._precompute_injection_embeddings()
        logger.info("Injection embeddings computed")
        
        # Load CLIP model
        _ml_repo.get_clip_model()
        logger.info("CLIP model loaded successfully")
        
        # Load YOLO model
        _ml_repo.get_yolo_model()
        logger.info("YOLO model loaded successfully")
        
        logger.info("All ML models pre-loaded successfully")
        return True
        
    except Exception as e:
        logger.warning(f"Model pre-loading failed (will load on first request): {e}")
        return False


async def preload_models_async():
    """Async version of model preloading.
    
    Runs model loading in thread pool to avoid blocking.
    """
    return await asyncio.to_thread(preload_models)


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
