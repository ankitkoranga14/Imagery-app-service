"""
Celery Tasks for Guardrail Microservice

This module now focuses solely on validation tasks.
The complex pipeline chain (rembg, realesrgan, nano_banana, generation) has been removed.

Primary task: validate_image_task (in validation_tasks.py)

Legacy pipeline tasks are deprecated and will be removed in future versions.
"""

import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from celery import shared_task

from src.core.celery_app import celery_app
from src.core.logging import get_logger, set_job_context, clear_job_context
from src.core.config import settings

logger = get_logger(__name__)


# =============================================================================
# DEPRECATED - Legacy pipeline support
# =============================================================================
# These functions are kept for backward compatibility but are deprecated.
# New code should use validate_image_task from validation_tasks.py

def update_job_status_sync(
    job_id: str,
    status: str,
    stage: Optional[str] = None,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    storage_key: Optional[str] = None,
    storage_field: Optional[str] = None
):
    """
    DEPRECATED: Synchronously update job status in database.
    Use validation_tasks.update_validation_job_sync for new code.
    """
    from sqlmodel import Session, select
    from sqlalchemy import create_engine
    
    # Try to use new ValidationJob model first
    try:
        from src.modules.validation.models import ValidationJob
        
        sync_url = settings.DATABASE_URL.replace("+aiosqlite", "").replace("sqlite+", "sqlite:///")
        if "postgresql" in settings.DATABASE_URL:
            sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
        
        engine = create_engine(sync_url)
        
        with Session(engine) as session:
            statement = select(ValidationJob).where(ValidationJob.id == job_id)
            job = session.exec(statement).first()
            
            if job:
                job.status = status
                session.add(job)
                session.commit()
                logger.info("job_status_updated", job_id=job_id, status=status)
                return
    except Exception as e:
        logger.warning(f"Could not update using new model: {e}")
    
    # Fallback to legacy ImageJob
    try:
        from src.modules.imagery.models import ImageJob
        
        sync_url = settings.DATABASE_URL.replace("+aiosqlite", "").replace("sqlite+", "sqlite:///")
        if "postgresql" in settings.DATABASE_URL:
            sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
        
        engine = create_engine(sync_url)
        
        with Session(engine) as session:
            statement = select(ImageJob).where(ImageJob.id == job_id)
            job = session.exec(statement).first()
            
            if job:
                job.status = status
                job.updated_at = datetime.utcnow()
                
                if storage_key and storage_field:
                    setattr(job, storage_field, storage_key)
                
                if stage:
                    current_meta = job.ai_metadata or {}
                    current_meta["current_stage"] = stage
                    current_meta["stage_updated_at"] = datetime.utcnow().isoformat()
                    
                    if metadata:
                        stages = current_meta.get("stages", {})
                        stages[stage] = metadata
                        current_meta["stages"] = stages
                    
                    if error_message:
                        current_meta["error"] = error_message
                        current_meta["failed_stage"] = stage
                    
                    job.ai_metadata = current_meta
                
                session.add(job)
                session.commit()
                
                logger.info("job_status_updated", job_id=job_id, status=status, stage=stage)
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")


# =============================================================================
# Stage 0: Quality Check (Guardrail) - KEPT for backward compatibility
# =============================================================================

@celery_app.task(
    bind=True,
    name="src.pipeline.tasks.process_guardrail",
    max_retries=1,
    default_retry_delay=5,
    acks_late=True
)
def process_guardrail(
    self,
    job_id: str,
    prompt: str,
    input_storage_key: Optional[str] = None,
    use_simulation: bool = False
) -> Dict[str, Any]:
    """
    Celery task for Guardrail validation.
    
    This is the primary validation task for the Guardrail Microservice.
    Can be used standalone or as part of a larger pipeline.
    """
    set_job_context(job_id, "guardrail")
    
    try:
        logger.info("task_guardrail_started", prompt=prompt)
        
        # Update job status
        update_job_status_sync(job_id, "PROCESSING", stage="guardrail")
        
        # Load image if present
        image_bytes = None
        if input_storage_key:
            try:
                with open(f"./data/storage/{input_storage_key}", "rb") as f:
                    image_bytes = f.read()
            except FileNotFoundError:
                logger.warning(f"Input file not found: {input_storage_key}")
        
        # Process using guardrail implementation
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        from src.pipeline.stages import process_guardrail_stage
        logger.info("guardrail_using_actual_implementation", job_id=job_id)
        is_passed, metadata = loop.run_until_complete(
            process_guardrail_stage(prompt, image_bytes, job_id)
        )
        
        loop.close()
        
        # Update job status
        update_job_status_sync(
            job_id,
            "PROCESSING" if is_passed else "FAILED",
            stage="guardrail",
            metadata=metadata,
            error_message=None if is_passed else f"Guardrail blocked: {', '.join(metadata.get('reasons', []))}"
        )
        
        if not is_passed:
            logger.warning("task_guardrail_blocked", job_id=job_id, reasons=metadata.get("reasons"))
            from src.core.exceptions import PipelineStageError
            raise PipelineStageError(
                f"Guardrail blocked: {', '.join(metadata.get('reasons', []))}",
                stage="guardrail",
                job_id=job_id
            )
            
        logger.info("task_guardrail_passed", job_id=job_id)
        
        return {
            "job_id": job_id,
            "stage": "guardrail",
            "output_storage_key": input_storage_key,
            "metadata": metadata,
            "use_simulation": use_simulation
        }
        
    except Exception as e:
        if "PipelineStageError" not in str(type(e)):
            logger.error("task_guardrail_unexpected_error", error=str(e), traceback=traceback.format_exc())
            update_job_status_sync(job_id, "FAILED", stage="guardrail", error_message=str(e))
        raise
    finally:
        clear_job_context()


# =============================================================================
# DEPRECATED Pipeline Stages - Removed from Guardrail Microservice
# =============================================================================
# The following pipeline stages have been removed:
# - process_rembg (background removal)
# - process_realesrgan (4K upscaling)
# - process_nano_banana (smart placement)
# - process_generation (hero generation)
# - run_full_pipeline (pipeline chain)
#
# These were part of the generative image pipeline and are not needed
# for the lean validation microservice.
#
# If you need these features, they should be implemented in a separate
# "Generation Microservice" that calls this validation service first.
# =============================================================================
