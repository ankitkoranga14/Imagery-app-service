"""
Validation Endpoint - Guardrail Microservice

Single POST /validate endpoint with support for:
- Synchronous mode: Wait for result and return immediately
- Asynchronous mode: Return job ID for polling

This replaces the complex pipeline processing endpoint with a lean validation service.
"""

import uuid
import base64
import hashlib
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.database import get_session
from src.core.config import settings
from src.core.logging import get_logger
from src.core.metrics import guardrail_latency_seconds, record_guardrail_validation
from src.engines.guardrail.schemas import GuardrailRequestDTO, GuardrailStatus
from src.engines.guardrail.services import GuardrailService
from src.api.dependencies import get_guardrail_service
from src.modules.validation.models import ValidationLog, ValidationJob

import time

logger = get_logger(__name__)
router = APIRouter()

# Constants
MAX_IMAGE_SIZE_BYTES = getattr(settings, 'MAX_IMAGE_SIZE_BYTES', 10 * 1024 * 1024)  # 10MB
MAX_IMAGE_SIZE_MB = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)


# =============================================================================
# Request/Response Schemas
# =============================================================================

class ValidateRequest(BaseModel):
    """Request for image validation."""
    prompt: str = Field(..., max_length=2000, description="Text prompt describing the expected content")
    image_bytes: str = Field(..., description="Base64 encoded image")
    async_mode: bool = Field(default=False, description="If true, return job_id for polling instead of waiting")
    
    @field_validator('image_bytes')
    @classmethod
    def validate_image_size(cls, v: str) -> str:
        """Validate image doesn't exceed max size."""
        decoded_size_bytes = len(v) * 3 / 4  # Approximate decoded size
        if decoded_size_bytes > MAX_IMAGE_SIZE_BYTES:
            actual_size_mb = decoded_size_bytes / (1024 * 1024)
            raise ValueError(
                f"Image size ({actual_size_mb:.2f}MB) exceeds maximum ({MAX_IMAGE_SIZE_MB:.0f}MB). "
                "Please compress or resize."
            )
        return v


class ValidateResponse(BaseModel):
    """Synchronous validation response."""
    status: str = Field(..., description="PASS or BLOCK")
    failure_reason: Optional[str] = Field(None, description="Reason for blocking, null if passed")
    scores: Dict[str, Any] = Field(default_factory=dict, description="Validation scores from all levels")
    latency_ms: int = Field(..., description="Total processing time in milliseconds")
    validation_id: Optional[str] = Field(None, description="ID of the validation log for reference")


class AsyncValidateResponse(BaseModel):
    """Async validation response (returns job ID)."""
    job_id: str = Field(..., description="Job ID for polling status")
    status: str = Field(default="PENDING", description="Initial job status")
    poll_url: str = Field(..., description="URL to poll for results")


class JobStatusResponse(BaseModel):
    """Job status for async validation."""
    job_id: str
    status: str = Field(..., description="PENDING, PROCESSING, COMPLETED, or FAILED")
    result: Optional[ValidateResponse] = Field(None, description="Result when completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: str
    completed_at: Optional[str] = None


class ValidationLogsResponse(BaseModel):
    """Response for validation history."""
    logs: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=None)
async def validate_image(
    request: ValidateRequest,
    session: AsyncSession = Depends(get_session),
    guardrail_service: GuardrailService = Depends(get_guardrail_service)
):
    """
    Validate an image against guardrail rules.
    
    This is the primary endpoint for the Guardrail Microservice.
    
    Validation Pipeline:
    - L0: Cache check (instant, serves cached results)
    - L1: Text validation (injection, policy, food domain)
    - L2: Physics check (brightness, glare, blur, contrast)
    - L3: Geometry check (YOLO food detection) - runs if L2 borderline
    - L4: Context check (CLIP food/NSFW) - runs parallel with L3 if borderline
    
    Args:
        request: Validation request with prompt and base64 image
        
    Returns:
        ValidateResponse (sync) or AsyncValidateResponse (async mode)
    """
    start_time = time.time()
    
    # Calculate image hash for deduplication
    try:
        image_data = base64.b64decode(request.image_bytes)
        image_hash = hashlib.sha256(image_data).hexdigest()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    if request.async_mode:
        # Async mode: Create job and dispatch to Celery
        job_id = str(uuid.uuid4())
        
        # Create job record
        job = ValidationJob(
            id=job_id,
            status="PENDING",
            prompt=request.prompt,
            image_hash=image_hash
        )
        session.add(job)
        await session.commit()
        
        # Dispatch to Celery
        from src.pipeline.validation_tasks import validate_image_task
        validate_image_task.delay(
            job_id=job_id,
            image_base64=request.image_bytes,
            prompt=request.prompt
        )
        
        logger.info("validation_job_created", job_id=job_id)
        
        return AsyncValidateResponse(
            job_id=job_id,
            status="PENDING",
            poll_url=f"/api/v1/validate/jobs/{job_id}"
        )
    
    # Sync mode: Run validation directly
    try:
        guardrail_request = GuardrailRequestDTO(
            prompt=request.prompt,
            image_bytes=request.image_bytes
        )
        
        result = await guardrail_service.validate(guardrail_request)
        
        # Extract result data
        status = result.status.value
        failure_reason = ", ".join(result.reasons) if result.reasons else None
        scores = result.scores
        latency_ms = result.metadata.get("processing_time_ms", 0)
        
        # Record metrics
        guardrail_latency_seconds.observe(latency_ms / 1000.0)
        block_reason = result.reasons[0].split(":")[0] if result.reasons else "none"
        record_guardrail_validation(status=status, block_reason=block_reason)
        
        # Save validation log
        trace = result.metadata.get("validation_trace", {})
        log = ValidationLog(
            image_hash=image_hash,
            prompt=request.prompt,
            status=status,
            failure_reason=failure_reason,
            failure_level=trace.get("levels_failed", [None])[0] if trace.get("levels_failed") else None,
            scores=scores,
            latency_ms=latency_ms,
            latency_breakdown=trace.get("timings", {}),
            cache_hit=result.metadata.get("cache_hit", False),
            parallel_execution=trace.get("parallel_execution", False),
            levels_executed=",".join(trace.get("levels_executed", [])),
            clip_variant=result.metadata.get("model_variants", {}).get("clip"),
            yolo_variant=result.metadata.get("model_variants", {}).get("yolo")
        )
        session.add(log)
        await session.commit()
        await session.refresh(log)
        
        logger.info(
            "validation_completed",
            status=status,
            latency_ms=latency_ms,
            cache_hit=result.metadata.get("cache_hit", False)
        )
        
        return ValidateResponse(
            status=status,
            failure_reason=failure_reason,
            scores=scores,
            latency_ms=latency_ms,
            validation_id=log.id
        )
        
    except Exception as e:
        logger.error("validation_failed", error=str(e))
        record_guardrail_validation(status="ERROR", block_reason="internal_error")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Get status of an async validation job.
    
    Poll this endpoint after submitting with async_mode=true.
    """
    result = await session.execute(
        select(ValidationJob).where(ValidationJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = JobStatusResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at.isoformat() if job.created_at else "",
        completed_at=job.completed_at.isoformat() if job.completed_at else None
    )
    
    if job.status == "COMPLETED":
        response.result = ValidateResponse(
            status=job.result_status or "UNKNOWN",
            failure_reason=job.result_reason,
            scores=job.result_scores or {},
            latency_ms=job.result_latency_ms or 0,
            validation_id=job.validation_log_id
        )
    elif job.status == "FAILED":
        response.error = job.error_message
    
    return response


@router.get("/logs", response_model=ValidationLogsResponse)
async def get_validation_logs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None, description="Filter by PASS or BLOCK"),
    session: AsyncSession = Depends(get_session)
):
    """
    Get validation history (audit logs).
    
    Returns paginated list of all validation attempts.
    """
    from sqlalchemy import func, desc
    
    # Build query
    query = select(ValidationLog)
    count_query = select(func.count(ValidationLog.id))
    
    if status:
        query = query.where(ValidationLog.status == status)
        count_query = count_query.where(ValidationLog.status == status)
    
    # Get total count
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0
    
    # Get paginated results
    offset = (page - 1) * page_size
    query = query.order_by(desc(ValidationLog.created_at)).offset(offset).limit(page_size)
    
    result = await session.execute(query)
    logs = result.scalars().all()
    
    return ValidationLogsResponse(
        logs=[log.to_response_dict() for log in logs],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/logs/{log_id}")
async def get_validation_log(
    log_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Get details of a specific validation log.
    """
    result = await session.execute(
        select(ValidationLog).where(ValidationLog.id == log_id)
    )
    log = result.scalar_one_or_none()
    
    if not log:
        raise HTTPException(status_code=404, detail="Validation log not found")
    
    return log.to_response_dict()


@router.get("/stats")
async def get_validation_stats(
    session: AsyncSession = Depends(get_session)
):
    """
    Get validation statistics summary.
    
    Returns counts, pass/block rates, and average latencies.
    """
    from sqlalchemy import func
    
    # Total counts
    total_result = await session.execute(select(func.count(ValidationLog.id)))
    total = total_result.scalar() or 0
    
    pass_result = await session.execute(
        select(func.count(ValidationLog.id)).where(ValidationLog.status == "PASS")
    )
    pass_count = pass_result.scalar() or 0
    
    block_result = await session.execute(
        select(func.count(ValidationLog.id)).where(ValidationLog.status == "BLOCK")
    )
    block_count = block_result.scalar() or 0
    
    # Average latency
    avg_latency_result = await session.execute(
        select(func.avg(ValidationLog.latency_ms))
    )
    avg_latency = avg_latency_result.scalar() or 0
    
    # Cache hit rate
    cache_hit_result = await session.execute(
        select(func.count(ValidationLog.id)).where(ValidationLog.cache_hit == True)
    )
    cache_hits = cache_hit_result.scalar() or 0
    
    return {
        "total_validations": total,
        "pass_count": pass_count,
        "block_count": block_count,
        "pass_rate": pass_count / total if total > 0 else 0,
        "block_rate": block_count / total if total > 0 else 0,
        "average_latency_ms": round(avg_latency, 2),
        "cache_hit_count": cache_hits,
        "cache_hit_rate": cache_hits / total if total > 0 else 0
    }
