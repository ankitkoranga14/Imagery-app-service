"""
Process Endpoint - Full Pipeline Dispatch

POST /api/v1/process - Submit image for full pipeline processing:
1. Guardrail validation (sync)
2. Dispatch async pipeline if validation passes
"""

import base64
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_session
from src.core.storage import get_storage, IStorage
from src.core.config import settings
from src.core.logging import get_logger, set_job_context, LogContext
from src.core.metrics import record_job_completion
from src.engines.guardrail.schemas import GuardrailRequestDTO
from src.engines.guardrail.services import GuardrailService
from src.api.dependencies import get_guardrail_service
from src.modules.imagery.models import ImageJob, JobStatus, PipelineStage, StageStatus

# Constants for file size limits
MAX_IMAGE_SIZE_BYTES = settings.MAX_IMAGE_SIZE_BYTES  # 10MB from config
MAX_IMAGE_SIZE_MB = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Request/Response Schemas
# =============================================================================

class ProcessRequest(BaseModel):
    """Request for image processing pipeline."""
    prompt: str = Field(..., max_length=2000, description="Description of the image")
    image_base64: str = Field(..., description="Base64 encoded image")
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pipeline options: {simulate: bool, scale: int, placement: {...}}"
    )
    
    @field_validator('image_base64')
    @classmethod
    def validate_image_size(cls, v: str) -> str:
        """Validate that the base64 image doesn't exceed the maximum size."""
        # Calculate approximate decoded size (base64 is ~33% larger than binary)
        decoded_size_bytes = len(v) * 3 / 4
        
        if decoded_size_bytes > MAX_IMAGE_SIZE_BYTES:
            actual_size_mb = decoded_size_bytes / (1024 * 1024)
            raise ValueError(
                f"Image size ({actual_size_mb:.2f}MB) exceeds maximum allowed size ({MAX_IMAGE_SIZE_MB:.0f}MB). "
                f"Please compress or resize your image."
            )
        return v


class ProcessResponse(BaseModel):
    """Response from process endpoint."""
    job_id: str
    status: str
    message: str
    guardrail: Optional[Dict[str, Any]] = None
    estimated_cost_usd: float = 0.08
    estimated_time_seconds: int = 30


class JobStatusResponse(BaseModel):
    """Full job status response."""
    id: str
    status: str
    current_stage: Optional[str]
    progress: Dict[str, Any]
    storage_urls: Optional[Dict[str, str]]
    error: Optional[Dict[str, str]]
    cost_usd: float
    processing_time_ms: int
    created_at: str
    completed_at: Optional[str]


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=ProcessResponse)
async def process_image(
    request: ProcessRequest,
    session: AsyncSession = Depends(get_session),
    storage: IStorage = Depends(get_storage),
    guardrail_service: GuardrailService = Depends(get_guardrail_service)
):
    """
    Submit an image for full pipeline processing.
    
    Flow:
    1. Create job record
    2. Run synchronous guardrail validation
    3. If passed, upload image and dispatch async pipeline
    4. Return job ID for status tracking
    
    The async pipeline will execute:
    - Stage 1: Rembg (background removal)
    - Stage 2: RealESRGAN (4K upscaling)
    - Stage 3: Nano Banana (smart placement, $0.08/image)
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Calculate image size for logging
    image_size_bytes = len(request.image_base64) * 3 / 4  # Approximate decoded size
    image_size_mb = image_size_bytes / (1024 * 1024)
    
    with LogContext(job_id=job_id, stage="validation"):
        logger.info(
            "process_request_received", 
            prompt_length=len(request.prompt),
            image_size_mb=round(image_size_mb, 2),
            max_allowed_mb=MAX_IMAGE_SIZE_MB
        )
        
        # Create job record
        job = ImageJob(
            id=job_id,
            prompt=request.prompt,
            status=JobStatus.VALIDATING.value,
            current_stage=PipelineStage.GUARDRAIL.value
        )
        job.update_stage(PipelineStage.GUARDRAIL.value, StageStatus.IN_PROGRESS.value)
        
        session.add(job)
        await session.commit()
        await session.refresh(job)
        
        # Run full guardrail validation (same as standalone guardrail endpoint)
        try:
            guardrail_request = GuardrailRequestDTO(
                prompt=request.prompt,
                image_bytes=request.image_base64  # Full validation including image
            )
            logger.info("guardrail_validation_started")
            guardrail_result = await guardrail_service.validate(guardrail_request)
            logger.info("guardrail_validation_completed", 
                       status=guardrail_result.status.value,
                       processing_time_ms=guardrail_result.metadata.get("processing_time_ms"))
            
            # Update job with guardrail result
            job.guardrail_status = guardrail_result.status.value
            job.guardrail_scores = guardrail_result.scores
            
            if guardrail_result.status.value == "BLOCK":
                # Content blocked
                job.mark_blocked(
                    reasons=guardrail_result.reasons,
                    scores=guardrail_result.scores
                )
                job.update_stage(
                    PipelineStage.GUARDRAIL.value,
                    StageStatus.COMPLETED.value,
                    duration_ms=guardrail_result.metadata.get("processing_time_ms")
                )
                
                session.add(job)
                await session.commit()
                
                logger.warning(
                    "content_blocked",
                    reasons=guardrail_result.reasons
                )
                
                return ProcessResponse(
                    job_id=job_id,
                    status="BLOCKED",
                    message=f"Content blocked: {', '.join(guardrail_result.reasons)}",
                    guardrail={
                        "status": "BLOCK",
                        "reasons": guardrail_result.reasons,
                        "scores": guardrail_result.scores,
                        "metadata": guardrail_result.metadata
                    },
                    estimated_cost_usd=0.0,
                    estimated_time_seconds=0
                )
            
            # Guardrail passed
            job.update_stage(
                PipelineStage.GUARDRAIL.value,
                StageStatus.COMPLETED.value,
                duration_ms=guardrail_result.metadata.get("processing_time_ms")
            )
            job.status = JobStatus.VALIDATED.value
            await session.commit()
            await session.refresh(job)
            
        except Exception as e:
            logger.error("guardrail_validation_failed", error=str(e))
            job.mark_failed(str(e), PipelineStage.GUARDRAIL.value)
            session.add(job)
            await session.commit()
            raise HTTPException(status_code=500, detail=f"Guardrail validation failed: {str(e)}")
        
        # Upload image to storage
        try:
            job.update_stage(PipelineStage.UPLOAD.value, StageStatus.IN_PROGRESS.value)
            
            image_bytes = base64.b64decode(request.image_base64)
            storage_key = await storage.upload(
                image_bytes,
                f"{job_id}_original.png",
                folder=f"jobs/{job_id}",
                content_type="image/png"
            )
            
            job.original_storage_key = storage_key
            job.update_stage(PipelineStage.UPLOAD.value, StageStatus.COMPLETED.value)
            job.status = JobStatus.PROCESSING.value
            await session.commit()
            await session.refresh(job)
            
            logger.info("image_uploaded", storage_key=storage_key)
            
        except Exception as e:
            logger.error("image_upload_failed", error=str(e))
            job.mark_failed(str(e), PipelineStage.UPLOAD.value)
            session.add(job)
            await session.commit()
            raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
        
        # Dispatch async pipeline
        try:
            from src.pipeline.tasks import run_full_pipeline
            
            # Extract options
            pipeline_options = request.options or {}
            
            # Dispatch
            task_id = run_full_pipeline(
                job_id=job_id,
                input_storage_key=storage_key,
                prompt=request.prompt,
                options=pipeline_options
            )
            
            # Update job with task ID
            job.celery_task_id = task_id
            job.status = JobStatus.PROCESSING.value
            session.add(job)
            await session.commit()
            
            logger.info("pipeline_dispatched", job_id=job_id, task_id=task_id)
            
            return ProcessResponse(
                job_id=job_id,
                status="PROCESSING",
                message="Pipeline dispatched successfully",
                guardrail={
                    "status": "PASS",
                    "scores": guardrail_result.scores
                },
                estimated_cost_usd=0.08,
                estimated_time_seconds=30
            )
            
        except Exception as e:
            logger.error("pipeline_dispatch_failed", error=str(e))
            job.mark_failed(str(e), "dispatch")
            session.add(job)
            await session.commit()
            raise HTTPException(status_code=500, detail=f"Pipeline dispatch failed: {str(e)}")


@router.post("/upload", response_model=ProcessResponse)
async def process_uploaded_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    options: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
    storage: IStorage = Depends(get_storage),
    guardrail_service: GuardrailService = Depends(get_guardrail_service)
):
    """
    Submit an image via file upload for processing.
    
    Alternative to base64 endpoint for direct file uploads.
    """
    # Read file and convert to base64
    file_content = await file.read()
    image_base64 = base64.b64encode(file_content).decode("utf-8")
    
    # Parse options if provided
    import json
    parsed_options = json.loads(options) if options else None
    
    # Create request and delegate
    request = ProcessRequest(
        prompt=prompt,
        image_base64=image_base64,
        options=parsed_options
    )
    
    return await process_image(
        request=request,
        session=session,
        storage=storage,
        guardrail_service=guardrail_service
    )

