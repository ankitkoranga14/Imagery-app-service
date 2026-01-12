from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.config import settings
from src.core.database import get_session
from src.engines.guardrail.repositories import MLRepository, CacheRepository, LogRepository
from src.engines.guardrail.services import TextGuardrailService, ImageGuardrailService, GuardrailService
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Global Singletons - ML models are expensive, only load once per process
# =============================================================================
_ml_repo = MLRepository(settings.ML_MODEL_CACHE_DIR)
_text_service = TextGuardrailService(_ml_repo)
_image_service = ImageGuardrailService(_ml_repo)


def preload_models():
    """Pre-load ML models at startup to avoid first-request delay."""
    logger.info("Pre-loading ML models...")
    try:
        # Load text model
        _ml_repo.get_text_model()
        logger.info("Text model loaded successfully")
        
        # Pre-compute food embeddings
        _text_service._precompute_food_embeddings()
        logger.info("Food embeddings computed")
        
        # Load CLIP model (this can take time on first download)
        _ml_repo.get_clip_model()
        logger.info("CLIP model loaded successfully")
        
    except Exception as e:
        logger.warning(f"Model pre-loading failed (will load on first request): {e}")


def get_ml_repo() -> MLRepository:
    return _ml_repo


def get_cache_repo(request: Request) -> CacheRepository:
    return CacheRepository(request.app.state.redis, settings.CACHE_TTL_SECONDS)


def get_log_repo(session: AsyncSession = Depends(get_session)) -> LogRepository:
    return LogRepository(session)


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
    """Returns GuardrailService with singleton ML services."""
    return GuardrailService(
        ml_repo=_ml_repo,
        cache_repo=cache_repo,
        log_repo=log_repo,
        text_service=_text_service,
        image_service=_image_service
    )
