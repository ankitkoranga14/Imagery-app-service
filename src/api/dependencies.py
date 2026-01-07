from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.config import settings
from src.core.database import get_session
from src.engines.guardrail.repositories import MLRepository, CacheRepository, LogRepository
from src.engines.guardrail.services import TextGuardrailService, ImageGuardrailService, GuardrailService

# Singletons
_ml_repo = MLRepository(settings.ML_MODEL_CACHE_DIR)

def get_ml_repo() -> MLRepository:
    return _ml_repo

def get_cache_repo(request: Request) -> CacheRepository:
    return CacheRepository(request.app.state.redis, settings.CACHE_TTL_SECONDS)

def get_log_repo(session: AsyncSession = Depends(get_session)) -> LogRepository:
    return LogRepository(session)

def get_text_service(ml_repo: MLRepository = Depends(get_ml_repo)) -> TextGuardrailService:
    return TextGuardrailService(ml_repo)

def get_image_service(ml_repo: MLRepository = Depends(get_ml_repo)) -> ImageGuardrailService:
    return ImageGuardrailService(ml_repo)

def get_guardrail_service(
    ml_repo: MLRepository = Depends(get_ml_repo),
    cache_repo: CacheRepository = Depends(get_cache_repo),
    log_repo: LogRepository = Depends(get_log_repo),
    text_service: TextGuardrailService = Depends(get_text_service),
    image_service: ImageGuardrailService = Depends(get_image_service)
) -> GuardrailService:
    return GuardrailService(ml_repo, cache_repo, log_repo, text_service, image_service)
