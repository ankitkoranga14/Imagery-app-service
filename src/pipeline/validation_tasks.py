"""
Celery Tasks for Guardrail Microservice

Implements a single validation task with:
- Fail-fast logic: L0 (Cache) -> L1 (Text) -> L2 (Physics) -> L3/L4 (parallel if borderline)
- Lazy model loading to prevent OOM
- Database logging for all validation attempts
"""

import traceback
import base64
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime

from src.core.celery_app import celery_app
from src.core.logging import get_logger, set_job_context, clear_job_context
from src.core.config import settings

logger = get_logger(__name__)


def update_validation_job_sync(
    job_id: str,
    status: str,
    result_status: Optional[str] = None,
    result_reason: Optional[str] = None,
    result_scores: Optional[Dict[str, Any]] = None,
    result_latency_ms: Optional[int] = None,
    validation_log_id: Optional[str] = None,
    error_message: Optional[str] = None
):
    """
    Synchronously update validation job status in database.
    Called from Celery tasks.
    """
    from sqlmodel import Session, select
    from sqlalchemy import create_engine
    from src.modules.validation.models import ValidationJob
    
    # Create sync engine for Celery tasks
    sync_url = settings.DATABASE_URL.replace("+aiosqlite", "").replace("sqlite+", "sqlite:///")
    if "postgresql" in settings.DATABASE_URL:
        sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
    
    engine = create_engine(sync_url)
    
    with Session(engine) as session:
        statement = select(ValidationJob).where(ValidationJob.id == job_id)
        job = session.exec(statement).first()
        
        if job:
            job.status = status
            
            if status == "COMPLETED":
                job.result_status = result_status
                job.result_reason = result_reason
                job.result_scores = result_scores or {}
                job.result_latency_ms = result_latency_ms
                job.validation_log_id = validation_log_id
                job.completed_at = datetime.utcnow()
            elif status == "FAILED":
                job.error_message = error_message
                job.completed_at = datetime.utcnow()
            
            session.add(job)
            session.commit()
            
            logger.info(
                "validation_job_updated",
                job_id=job_id,
                status=status
            )


def save_validation_log_sync(
    image_hash: str,
    prompt: str,
    status: str,
    failure_reason: Optional[str],
    failure_level: Optional[str],
    scores: Dict[str, Any],
    latency_ms: int,
    latency_breakdown: Dict[str, float],
    cache_hit: bool,
    parallel_execution: bool,
    levels_executed: str,
    clip_variant: Optional[str] = None,
    yolo_variant: Optional[str] = None,
    image_url: Optional[str] = None
) -> str:
    """
    Synchronously save validation log to database.
    Returns the log ID.
    """
    from sqlmodel import Session
    from sqlalchemy import create_engine
    from src.modules.validation.models import ValidationLog
    
    # Create sync engine for Celery tasks
    sync_url = settings.DATABASE_URL.replace("+aiosqlite", "").replace("sqlite+", "sqlite:///")
    if "postgresql" in settings.DATABASE_URL:
        sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
    
    engine = create_engine(sync_url)
    
    log = ValidationLog(
        image_hash=image_hash,
        image_url=image_url,
        prompt=prompt,
        status=status,
        failure_reason=failure_reason,
        failure_level=failure_level,
        scores=scores,
        latency_ms=latency_ms,
        latency_breakdown=latency_breakdown,
        cache_hit=cache_hit,
        parallel_execution=parallel_execution,
        levels_executed=levels_executed,
        clip_variant=clip_variant,
        yolo_variant=yolo_variant
    )
    
    with Session(engine) as session:
        session.add(log)
        session.commit()
        session.refresh(log)
        
        logger.info(
            "validation_log_saved",
            log_id=log.id,
            status=status,
            latency_ms=latency_ms
        )
        
        return log.id


@celery_app.task(
    bind=True,
    name="src.pipeline.validation_tasks.validate_image_task",
    max_retries=1,
    default_retry_delay=5,
    acks_late=True
)
def validate_image_task(
    self,
    job_id: str,
    image_base64: str,
    prompt: str,
    image_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Celery task for image validation (Guardrail Microservice).
    
    Implements fail-fast logic:
    - L0: Cache check (instant)
    - L1: Text validation (NLP)
    - L2: Physics check (OpenCV)
    - L3/L4: Geometry + Context (YOLO + CLIP, parallel if L2 borderline)
    
    Args:
        job_id: Async job ID for status tracking
        image_base64: Base64 encoded image
        prompt: Text prompt for validation
        image_url: Optional URL if image was fetched
        
    Returns:
        Dict with status, failure_reason, scores, latency
    """
    set_job_context(job_id, "validation")
    
    try:
        logger.info("validate_image_task_started", job_id=job_id, prompt_length=len(prompt))
        
        # Update job status to PROCESSING
        update_validation_job_sync(job_id, "PROCESSING")
        
        # Compute image hash for caching/deduplication
        image_bytes = base64.b64decode(image_base64)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Run validation using GuardrailService
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Import and initialize services (lazy loading)
            from src.engines.guardrail.services import (
                GuardrailService,
                TextGuardrailService,
                ImageGuardrailService
            )
            from src.engines.guardrail.repositories import MLRepository, CacheRepository, LogRepository
            from src.engines.guardrail.schemas import GuardrailRequestDTO, GuardrailStatus
            
            # Initialize repositories
            ml_repo = MLRepository()
            cache_repo = CacheRepository()
            log_repo = LogRepository()
            text_service = TextGuardrailService(ml_repo)
            image_service = ImageGuardrailService(ml_repo)
            
            guardrail_service = GuardrailService(
                ml_repo, cache_repo, log_repo, text_service, image_service
            )
            
            # Create request
            request = GuardrailRequestDTO(
                prompt=prompt,
                image_bytes=image_base64
            )
            
            # Validate
            result = loop.run_until_complete(guardrail_service.validate(request))
            
        finally:
            loop.close()
        
        # Extract result data
        status = result.status.value
        failure_reason = ", ".join(result.reasons) if result.reasons else None
        scores = result.scores
        latency_ms = result.metadata.get("processing_time_ms", 0)
        
        # Extract metadata
        trace = result.metadata.get("validation_trace", {})
        latency_breakdown = trace.get("timings", {})
        cache_hit = result.metadata.get("cache_hit", False)
        parallel_execution = trace.get("parallel_execution", False)
        levels_executed = ",".join(trace.get("levels_executed", []))
        levels_failed = trace.get("levels_failed", [])
        failure_level = levels_failed[0] if levels_failed else None
        
        model_variants = result.metadata.get("model_variants", {})
        clip_variant = model_variants.get("clip")
        yolo_variant = model_variants.get("yolo")
        
        # Save validation log
        log_id = save_validation_log_sync(
            image_hash=image_hash,
            prompt=prompt,
            status=status,
            failure_reason=failure_reason,
            failure_level=failure_level,
            scores=scores,
            latency_ms=latency_ms,
            latency_breakdown=latency_breakdown,
            cache_hit=cache_hit,
            parallel_execution=parallel_execution,
            levels_executed=levels_executed,
            clip_variant=clip_variant,
            yolo_variant=yolo_variant,
            image_url=image_url
        )
        
        # Update job with result
        update_validation_job_sync(
            job_id=job_id,
            status="COMPLETED",
            result_status=status,
            result_reason=failure_reason,
            result_scores=scores,
            result_latency_ms=latency_ms,
            validation_log_id=log_id
        )
        
        logger.info(
            "validate_image_task_completed",
            job_id=job_id,
            status=status,
            latency_ms=latency_ms,
            cache_hit=cache_hit
        )
        
        return {
            "status": status,
            "failure_reason": failure_reason,
            "scores": scores,
            "latency_ms": latency_ms,
            "validation_log_id": log_id,
            "cache_hit": cache_hit
        }
        
    except Exception as e:
        logger.error(
            "validate_image_task_failed",
            job_id=job_id,
            error=str(e),
            traceback=traceback.format_exc()
        )
        
        # Update job as failed
        update_validation_job_sync(
            job_id=job_id,
            status="FAILED",
            error_message=str(e)
        )
        
        raise
        
    finally:
        clear_job_context()
