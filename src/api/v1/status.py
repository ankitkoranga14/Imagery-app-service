"""
Status Endpoint - Job Status Tracking

GET /api/v1/status/{job_id} - Get current status of a processing job
GET /api/v1/status/{job_id}/logs - Stream logs for a job
"""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from src.core.database import get_session
from src.core.storage import get_storage, IStorage
from src.core.logging import get_logger
from src.modules.imagery.models import ImageJob, JobStatus

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Response Schemas
# =============================================================================

class StageProgress(BaseModel):
    """Individual stage progress."""
    stage: str
    status: str
    duration_ms: Optional[int] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class StorageUrls(BaseModel):
    """Storage URLs for job outputs."""
    original: Optional[str] = None
    transparent: Optional[str] = None
    upscaled: Optional[str] = None
    generated: Optional[str] = None
    final: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Full job status response."""
    id: str
    status: str
    current_stage: Optional[str] = None
    stages: List[StageProgress]
    storage_urls: Optional[StorageUrls] = None
    guardrail: Optional[dict] = None
    error: Optional[dict] = None
    cost_usd: float = 0.0
    processing_time_ms: int = 0
    created_at: str
    completed_at: Optional[str] = None


class JobListResponse(BaseModel):
    """List of jobs response."""
    jobs: List[JobStatusResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    session: AsyncSession = Depends(get_session),
    storage: IStorage = Depends(get_storage)
):
    """
    Get the current status of a processing job.
    
    Returns:
        Full job status including:
        - Current stage and progress
        - Storage URLs for outputs
        - Cost and timing information
        - Error details if failed
    """
    # Find job
    statement = select(ImageJob).where(ImageJob.id == job_id)
    result = await session.execute(statement)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    # Build stages progress
    stage_order = ["upload", "guardrail", "rembg", "nano_banana", "generation", "realesrgan", "finalize"]
    stages = []
    
    for stage_name in stage_order:
        stage_data = job.stages_metadata.get(stage_name, {}) if job.stages_metadata else {}
        stages.append(StageProgress(
            stage=stage_name,
            status=stage_data.get("status", "pending"),
            duration_ms=stage_data.get("duration_ms"),
            started_at=stage_data.get("started_at"),
            completed_at=stage_data.get("completed_at"),
            error=stage_data.get("error")
        ))
    
    # Get storage URLs
    storage_urls = None
    if job.status in [JobStatus.PROCESSING.value, JobStatus.COMPLETED.value]:
        try:
            urls = {}
            if job.original_storage_key:
                urls["original"] = await storage.get_url(job.original_storage_key)
            if job.transparent_storage_key:
                urls["transparent"] = await storage.get_url(job.transparent_storage_key)
            if job.upscaled_storage_key:
                urls["upscaled"] = await storage.get_url(job.upscaled_storage_key)
            if job.generated_storage_key:
                urls["generated"] = await storage.get_url(job.generated_storage_key)
            if job.final_storage_key:
                urls["final"] = await storage.get_url(job.final_storage_key)
            storage_urls = StorageUrls(**urls)
        except Exception as e:
            logger.warning("failed_to_get_storage_urls", error=str(e))
    
    # Build response
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        current_stage=job.current_stage,
        stages=stages,
        storage_urls=storage_urls,
        guardrail={
            "status": job.guardrail_status,
            "reasons": job.guardrail_reasons.split(", ") if job.guardrail_reasons else [],
            "scores": job.guardrail_scores
        } if job.guardrail_status else None,
        error={
            "message": job.error_message,
            "stage": job.error_stage
        } if job.error_message else None,
        cost_usd=job.total_cost_usd,
        processing_time_ms=job.total_processing_time_ms,
        created_at=job.created_at.isoformat() if job.created_at else "",
        completed_at=job.completed_at.isoformat() if job.completed_at else None
    )


@router.get("/{job_id}/progress")
async def get_job_progress(
    job_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Get lightweight progress info for polling.
    
    Optimized for frequent polling from frontend.
    """
    statement = select(ImageJob).where(ImageJob.id == job_id)
    result = await session.execute(statement)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return job.get_pipeline_progress()


@router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Get structured logs for a specific job.
    
    Returns logs in JSON format for frontend display.
    """
    # Find job
    statement = select(ImageJob).where(ImageJob.id == job_id)
    result = await session.execute(statement)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    # Build log entries from stages metadata
    logs = []
    
    if job.stages_metadata:
        for stage_name, stage_data in job.stages_metadata.items():
            # Stage started
            if stage_data.get("started_at"):
                logs.append({
                    "timestamp": stage_data["started_at"],
                    "level": "info",
                    "event": "stage_started",
                    "stage": stage_name,
                    "job_id": job_id
                })
            
            # Stage completed
            if stage_data.get("completed_at"):
                log_entry = {
                    "timestamp": stage_data["completed_at"],
                    "level": "info" if stage_data.get("status") == "completed" else "error",
                    "event": "stage_completed" if stage_data.get("status") == "completed" else "stage_failed",
                    "stage": stage_name,
                    "job_id": job_id,
                    "duration_ms": stage_data.get("duration_ms")
                }
                
                if stage_data.get("error"):
                    log_entry["error"] = stage_data["error"]
                
                # Add stage-specific metadata
                if stage_name == "nano_banana" and stage_data.get("cost_usd"):
                    log_entry["cost_usd"] = stage_data["cost_usd"]
                
                if stage_data.get("vram_used_gb"):
                    log_entry["vram_used_gb"] = stage_data["vram_used_gb"]
                
                logs.append(log_entry)
    
    # Sort by timestamp
    logs.sort(key=lambda x: x.get("timestamp", ""))
    
    return {
        "job_id": job_id,
        "logs": logs,
        "status": job.status
    }


@router.get("", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session)
):
    """
    List all jobs with pagination.
    """
    # Build query
    query = select(ImageJob)
    
    if status:
        query = query.where(ImageJob.status == status)
    
    # Order by created_at descending
    query = query.order_by(ImageJob.created_at.desc())
    
    # Pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    
    result = await session.execute(query)
    jobs = result.scalars().all()
    
    # Get total count
    count_query = select(ImageJob)
    if status:
        count_query = count_query.where(ImageJob.status == status)
    count_result = await session.execute(count_query)
    total = len(count_result.scalars().all())
    
    # Convert to response
    job_responses = []
    for job in jobs:
        stages = []
        stage_order = ["upload", "guardrail", "rembg", "nano_banana", "generation", "realesrgan"]
        for stage_name in stage_order:
            stage_data = job.stages_metadata.get(stage_name, {}) if job.stages_metadata else {}
            stages.append(StageProgress(
                stage=stage_name,
                status=stage_data.get("status", "pending"),
                duration_ms=stage_data.get("duration_ms")
            ))
        
        job_responses.append(JobStatusResponse(
            id=job.id,
            status=job.status,
            current_stage=job.current_stage,
            stages=stages,
            cost_usd=job.total_cost_usd,
            processing_time_ms=job.total_processing_time_ms,
            created_at=job.created_at.isoformat() if job.created_at else "",
            completed_at=job.completed_at.isoformat() if job.completed_at else None
        ))
    
    return JobListResponse(
        jobs=job_responses,
        total=total,
        page=page,
        page_size=page_size
    )

