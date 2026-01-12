"""
Celery Tasks for Image Processing Pipeline

Implements async task chaining with:
- Exponential backoff for Nano Banana API
- Circuit breaker for stage failures
- Database status updates on failure
"""

import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from celery import shared_task, chain, group
from celery.exceptions import MaxRetriesExceededError

from src.core.celery_app import celery_app
from src.core.logging import get_logger, set_job_context, clear_job_context
from src.core.metrics import (
    record_job_completion,
    pipeline_total_duration,
    active_jobs_gauge
)
from src.core.exceptions import (
    PipelineStageError,
    ExternalAPIError,
    GPUMemoryError,
    get_circuit_breaker
)
from src.core.storage import StorageFactory
from src.core.config import settings

logger = get_logger(__name__)


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
    Synchronously update job status in database.
    Called from Celery tasks.
    """
    from sqlmodel import Session, select
    from sqlalchemy import create_engine
    from src.modules.imagery.models import ImageJob
    
    # Create sync engine for Celery tasks
    sync_url = settings.DATABASE_URL.replace("+aiosqlite", "").replace("sqlite+", "sqlite:///")
    if "postgresql" in settings.DATABASE_URL:
        sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
    
    engine = create_engine(sync_url)
    
    with Session(engine) as session:
        # Find job
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
                
                # Also update stages_metadata for UI
                stage_status = "in_progress"
                if status == "COMPLETED":
                    stage_status = "completed"
                elif status == "FAILED":
                    stage_status = "failed"
                elif status == "PROCESSING" and metadata:
                    # If we have metadata (like duration), it means the stage finished 
                    # but the overall job is still PROCESSING
                    stage_status = "completed"
                
                job.update_stage(
                    stage=stage,
                    status=stage_status,
                    duration_ms=metadata.get("duration_ms") if metadata else None,
                    metadata=metadata,
                    error=error_message
                )
            
            session.add(job)
            session.commit()
            
            logger.info(
                "job_status_updated",
                job_id=job_id,
                status=status,
                stage=stage
            )


# =============================================================================
# Stage 0: Quality Check (Guardrail)
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
                logger.warning(f"Input file not found for guardrail: {input_storage_key}")
        
        # Process using ACTUAL guardrail implementation (never simulated)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Guardrail ALWAYS uses actual implementation for content validation
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
            record_job_completion("failed", failure_stage="guardrail")
            # Raise error to stop the chain
            raise PipelineStageError(
                f"Guardrail blocked: {', '.join(metadata.get('reasons', []))}",
                stage="guardrail",
                job_id=job_id
            )
            
        logger.info("task_guardrail_passed", job_id=job_id)
        
        return {
            "job_id": job_id,
            "stage": "guardrail",
            "output_storage_key": input_storage_key, # Pass through
            "metadata": metadata,
            "use_simulation": use_simulation
        }
        
    except PipelineStageError:
        raise
    except Exception as e:
        logger.error("task_guardrail_unexpected_error", error=str(e), traceback=traceback.format_exc())
        update_job_status_sync(job_id, "FAILED", stage="guardrail", error_message=str(e))
        record_job_completion("failed", failure_stage="guardrail")
        raise
    finally:
        clear_job_context()



# =============================================================================
# Stage 1: Background Removal (Rembg)
# =============================================================================

@celery_app.task(
    bind=True,
    name="src.pipeline.tasks.process_rembg",
    max_retries=2,
    default_retry_delay=10,
    acks_late=True
)
def process_rembg(
    self,
    prev_result: Dict[str, Any],
    use_simulation: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Celery task for background removal.
    
    Args:
        prev_result: Result from previous stage (guardrail)
        use_simulation: Override simulation setting
        
    Returns:
        Dict with output_storage_key and metadata
    """
    job_id = prev_result["job_id"]
    input_storage_key = prev_result["output_storage_key"]
    if use_simulation is None:
        use_simulation = prev_result.get("use_simulation", False)
    set_job_context(job_id, "rembg")
    
    try:
        logger.info("task_rembg_started", input_key=input_storage_key)
        
        # Update job status
        update_job_status_sync(job_id, "PROCESSING", stage="rembg")
        active_jobs_gauge.inc()
        
        # Get storage
        storage = StorageFactory.get_storage()
        
        # Load input image (sync wrapper for async storage)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Read from storage
            with open(f"./data/storage/{input_storage_key}", "rb") as f:
                image_bytes = f.read()
        except FileNotFoundError:
            raise PipelineStageError(
                f"Input file not found: {input_storage_key}",
                stage="rembg",
                job_id=job_id
            )
        
        # Process
        if use_simulation:
            from src.pipeline.stages import simulate_rembg_stage
            output_bytes, metadata = simulate_rembg_stage(image_bytes, job_id)
        else:
            from src.pipeline.stages import process_rembg_stage
            output_bytes, metadata = process_rembg_stage(image_bytes, job_id)
        
        # Save output
        output_key = loop.run_until_complete(
            storage.upload(
                output_bytes,
                f"{job_id}_nobg.png",
                folder=f"jobs/{job_id}",
                content_type="image/png"
            )
        )
        
        loop.close()
        
        # Update job with transparent URL
        update_job_status_sync(
            job_id,
            "PROCESSING",
            stage="rembg",
            metadata=metadata,
            storage_key=output_key,
            storage_field="transparent_storage_key"
        )
        
        logger.info(
            "task_rembg_completed",
            output_key=output_key,
            duration_ms=metadata.get("duration_ms")
        )
        
        return {
            "job_id": job_id,
            "stage": "rembg",
            "output_storage_key": output_key,
            "metadata": metadata,
            "use_simulation": use_simulation
        }
        
    except GPUMemoryError as e:
        logger.error("task_rembg_oom", error=str(e))
        update_job_status_sync(job_id, "FAILED", stage="rembg", error_message="GPU out of memory")
        record_job_completion("failed", failure_stage="rembg")
        raise
        
    except PipelineStageError as e:
        logger.error("task_rembg_failed", error=str(e))
        update_job_status_sync(job_id, "FAILED", stage="rembg", error_message=str(e))
        record_job_completion("failed", failure_stage="rembg")
        raise
        
    except Exception as e:
        logger.error(
            "task_rembg_unexpected_error",
            error=str(e),
            traceback=traceback.format_exc()
        )
        
        # Retry on transient errors
        try:
            self.retry(exc=e, countdown=10 * (2 ** self.request.retries))
        except MaxRetriesExceededError:
            update_job_status_sync(
                job_id,
                "FAILED",
                stage="rembg",
                error_message=f"Max retries exceeded: {str(e)}"
            )
            record_job_completion("failed", failure_stage="rembg")
            raise
    
    finally:
        clear_job_context()


# =============================================================================
# Stage 2: 4K Upscaling (RealESRGAN)
# =============================================================================

@celery_app.task(
    bind=True,
    name="src.pipeline.tasks.process_realesrgan",
    max_retries=2,
    default_retry_delay=15,
    acks_late=True
)
def process_realesrgan(
    self,
    prev_result: Dict[str, Any],
    scale: int = 4
) -> Dict[str, Any]:
    """
    Celery task for 4K upscaling.
    
    Args:
        prev_result: Result from previous stage (rembg)
        scale: Upscale factor
        
    Returns:
        Dict with output_storage_key and metadata
    """
    job_id = prev_result["job_id"]
    input_storage_key = prev_result["output_storage_key"]
    use_simulation = prev_result.get("use_simulation", False)
    
    set_job_context(job_id, "realesrgan")
    
    try:
        logger.info("task_realesrgan_started", input_key=input_storage_key)
        
        # Update job status
        update_job_status_sync(job_id, "PROCESSING", stage="realesrgan")
        
        # Get storage
        storage = StorageFactory.get_storage()
        
        # Load input image
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            with open(f"./data/storage/{input_storage_key}", "rb") as f:
                image_bytes = f.read()
        except FileNotFoundError:
            raise PipelineStageError(
                f"Input file not found: {input_storage_key}",
                stage="realesrgan",
                job_id=job_id
            )
        
        # Process
        if use_simulation:
            from src.pipeline.stages import simulate_realesrgan_stage
            output_bytes, metadata = simulate_realesrgan_stage(image_bytes, job_id, scale)
        else:
            from src.pipeline.stages import process_realesrgan_stage
            output_bytes, metadata = process_realesrgan_stage(image_bytes, job_id, scale)
        
        # Save output
        output_key = loop.run_until_complete(
            storage.upload(
                output_bytes,
                f"{job_id}_upscaled.png",
                folder=f"jobs/{job_id}",
                content_type="image/png"
            )
        )
        
        loop.close()
        
        # Update job
        update_job_status_sync(
            job_id,
            "PROCESSING",
            stage="realesrgan",
            metadata=metadata,
            storage_key=output_key,
            storage_field="upscaled_storage_key"
        )
        
        logger.info(
            "task_realesrgan_completed",
            output_key=output_key,
            duration_ms=metadata.get("duration_ms")
        )
        
        return {
            "job_id": job_id,
            "stage": "realesrgan",
            "output_storage_key": output_key,
            "metadata": metadata,
            "use_simulation": use_simulation
        }
        
    except GPUMemoryError as e:
        logger.error("task_realesrgan_oom", error=str(e))
        update_job_status_sync(job_id, "FAILED", stage="realesrgan", error_message="GPU out of memory")
        record_job_completion("failed", failure_stage="realesrgan")
        raise
        
    except PipelineStageError as e:
        logger.error("task_realesrgan_failed", error=str(e))
        update_job_status_sync(job_id, "FAILED", stage="realesrgan", error_message=str(e))
        record_job_completion("failed", failure_stage="realesrgan")
        raise
        
    except Exception as e:
        logger.error(
            "task_realesrgan_unexpected_error",
            error=str(e),
            traceback=traceback.format_exc()
        )
        
        try:
            self.retry(exc=e, countdown=15 * (2 ** self.request.retries))
        except MaxRetriesExceededError:
            update_job_status_sync(
                job_id,
                "FAILED",
                stage="realesrgan",
                error_message=f"Max retries exceeded: {str(e)}"
            )
            record_job_completion("failed", failure_stage="realesrgan")
            raise
    
    finally:
        clear_job_context()


# =============================================================================
# Stage 3: Smart Placement (Nano Banana API)
# =============================================================================

@celery_app.task(
    bind=True,
    name="src.pipeline.tasks.process_nano_banana",
    max_retries=5,
    default_retry_delay=5,
    acks_late=True,
    autoretry_for=(ExternalAPIError,),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True
)
def process_nano_banana(
    self,
    prev_result: Dict[str, Any],
    placement_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Celery task for Nano Banana smart placement.
    
    Configured with exponential backoff for network 503s.
    
    Args:
        prev_result: Result from previous stage (realesrgan)
        placement_options: Optional placement configuration
        
    Returns:
        Dict with output_storage_key and metadata
    """
    job_id = prev_result["job_id"]
    input_storage_key = prev_result["output_storage_key"]
    use_simulation = prev_result.get("use_simulation", False)
    
    set_job_context(job_id, "nano_banana")
    
    try:
        logger.info("task_nano_banana_started", input_key=input_storage_key)
        
        # Update job status
        update_job_status_sync(job_id, "PROCESSING", stage="nano_banana")
        
        # Get storage
        storage = StorageFactory.get_storage()
        
        # Load input image
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            with open(f"./data/storage/{input_storage_key}", "rb") as f:
                image_bytes = f.read()
        except FileNotFoundError:
            raise PipelineStageError(
                f"Input file not found: {input_storage_key}",
                stage="nano_banana",
                job_id=job_id
            )
        
        # Process (async function needs to run in loop)
        if use_simulation:
            from src.pipeline.stages import simulate_nano_banana_stage
            output_bytes, metadata = loop.run_until_complete(
                simulate_nano_banana_stage(image_bytes, job_id, placement_options)
            )
        else:
            from src.pipeline.stages import process_nano_banana_stage
            output_bytes, metadata = loop.run_until_complete(
                process_nano_banana_stage(image_bytes, job_id, placement_options)
            )
        
        # Save final output
        output_key = loop.run_until_complete(
            storage.upload(
                output_bytes,
                f"{job_id}_final.png",
                folder=f"jobs/{job_id}",
                content_type="image/png"
            )
        )
        
        loop.close()
        
        # Update job to completed
        update_job_status_sync(
            job_id,
            "COMPLETED",
            stage="nano_banana",
            metadata=metadata,
            storage_key=output_key,
            storage_field="final_storage_key"
        )
        
        record_job_completion("completed")
        active_jobs_gauge.dec()
        
        logger.info(
            "task_nano_banana_completed",
            output_key=output_key,
            duration_ms=metadata.get("duration_ms"),
            cost_usd=metadata.get("cost_usd")
        )
        
        return {
            "job_id": job_id,
            "stage": "nano_banana",
            "output_storage_key": output_key,
            "metadata": metadata,
            "status": "COMPLETED"
        }
        
    except ExternalAPIError as e:
        logger.warning(
            "task_nano_banana_api_error",
            error=str(e),
            retry=self.request.retries
        )
        
        # Exponential backoff: 5, 10, 20, 40, 80 seconds
        countdown = 5 * (2 ** self.request.retries)
        
        try:
            self.retry(exc=e, countdown=countdown)
        except MaxRetriesExceededError:
            update_job_status_sync(
                job_id,
                "FAILED",
                stage="nano_banana",
                error_message=f"Nano Banana API unavailable after {self.request.retries} retries"
            )
            record_job_completion("failed", failure_stage="nano_banana")
            raise
        
    except Exception as e:
        logger.error(
            "task_nano_banana_unexpected_error",
            error=str(e),
            traceback=traceback.format_exc()
        )
        
        update_job_status_sync(
            job_id,
            "FAILED",
            stage="nano_banana",
            error_message=str(e)
        )
        record_job_completion("failed", failure_stage="nano_banana")
        raise
    
    finally:
        clear_job_context()


# =============================================================================
# Stage 4: Hero Generation (Mocked)
# =============================================================================

@celery_app.task(
    bind=True,
    name="src.pipeline.tasks.process_generation",
    max_retries=2,
    default_retry_delay=10,
    acks_late=True
)
def process_generation(
    self,
    prev_result: Dict[str, Any],
    num_versions: int = 2
) -> Dict[str, Any]:
    """
    Celery task for Hero Generative Model (Mocked).
    """
    job_id = prev_result["job_id"]
    input_storage_key = prev_result["output_storage_key"]
    use_simulation = prev_result.get("use_simulation", False)
    
    set_job_context(job_id, "generation")
    
    try:
        logger.info("task_generation_started", input_key=input_storage_key)
        
        # Update job status
        update_job_status_sync(job_id, "PROCESSING", stage="generation")
        
        # Load input image
        try:
            with open(f"./data/storage/{input_storage_key}", "rb") as f:
                image_bytes = f.read()
        except FileNotFoundError:
            raise PipelineStageError(
                f"Input file not found: {input_storage_key}",
                stage="generation",
                job_id=job_id
            )
        
        # Process
        if use_simulation:
            from src.pipeline.stages import simulate_generation_stage
            output_bytes, metadata = simulate_generation_stage(image_bytes, job_id, num_versions)
        else:
            from src.pipeline.stages import process_generation_stage
            output_bytes, metadata = process_generation_stage(image_bytes, job_id, num_versions)
        
        # In a real implementation, we would save multiple versions.
        # For now, we just pass through or save a "generated" version.
        storage = StorageFactory.get_storage()
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        output_key = loop.run_until_complete(
            storage.upload(
                output_bytes,
                f"{job_id}_generated.png",
                folder=f"jobs/{job_id}",
                content_type="image/png"
            )
        )
        loop.close()
        
        # Update job status
        update_job_status_sync(
            job_id,
            "PROCESSING",
            stage="generation",
            metadata=metadata,
            storage_key=output_key,
            storage_field="generated_storage_key"
        )
        
        logger.info("task_generation_completed", output_key=output_key)
        
        return {
            "job_id": job_id,
            "stage": "generation",
            "output_storage_key": output_key,
            "metadata": metadata,
            "use_simulation": use_simulation
        }
        
    except Exception as e:
        logger.error("task_generation_failed", error=str(e))
        update_job_status_sync(job_id, "FAILED", stage="generation", error_message=str(e))
        record_job_completion("failed", failure_stage="generation")
        raise
    finally:
        clear_job_context()



# =============================================================================
# Full Pipeline Chain
# =============================================================================

@celery_app.task(name="src.pipeline.tasks.run_full_pipeline")
def run_full_pipeline(
    job_id: str,
    input_storage_key: str,
    prompt: str = "",
    options: Optional[Dict[str, Any]] = None
) -> str:
    """
    Dispatch the full pipeline as a chain of tasks.
    
    Pipeline follows Imagery Board PDF flow:
    1. Quality Check (Guardrail) - ACTUAL implementation
    2. Background Removal (Rembg) - Mocked for now
    3. Apply design/theme/background (Nano Banana) - Mocked for now
    4. Generate 2 versions (Generation) - Mocked for now
    5. Auto enhancement (RealESRGAN) - Mocked for now
    
    Args:
        job_id: Job ID
        input_storage_key: Storage key for input image
        prompt: User prompt for guardrail and generation
        options: Optional pipeline options
        
    Returns:
        Task chain ID
    """
    options = options or {}
    # Force simulation for all stages EXCEPT guardrail (which uses actual implementation)
    # Guardrail always uses actual implementation (use_simulation=False)
    # Other stages use simulation (mocked) until fully implemented
    use_simulation_for_other_stages = True  # Mock remaining stages
    scale = options.get("scale", 4)
    placement_options = options.get("placement")
    num_versions = options.get("num_versions", 2)
    
    logger.info(
        "pipeline_chain_dispatched",
        job_id=job_id,
        input_key=input_storage_key,
        guardrail="actual",
        other_stages="mocked"
    )
    
    # Create task chain following the Imagery Board PDF:
    # 1. Quality Check (Guardrail) - ACTUAL (use_simulation=False)
    # 2. Background Removal (Rembg) - Mocked
    # 3. Apply design/theme/background (Nano Banana) - Mocked
    # 4. Generate 2 versions (Generation) - Mocked
    # 5. Auto enhancement (RealESRGAN) - Mocked
    
    pipeline = chain(
        process_guardrail.s(job_id, prompt, input_storage_key, False),  # Guardrail is ALWAYS actual
        process_rembg.s(use_simulation=use_simulation_for_other_stages),
        process_nano_banana.s(placement_options=placement_options),
        process_generation.s(num_versions=num_versions),
        process_realesrgan.s(scale=scale)
    )
    
    # Execute chain
    result = pipeline.apply_async()
    
    return str(result.id)

