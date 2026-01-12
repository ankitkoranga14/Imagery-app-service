"""
Enhanced ImageJob Model with Pipeline Status Tracking

Tracks full pipeline state with:
- Per-stage status and timing
- Cost tracking
- Error state management
"""

import uuid
from enum import Enum
from sqlmodel import SQLModel, Field, Column, JSON
from typing import Optional, Dict, Any, List
from datetime import datetime


class JobStatus(str, Enum):
    """Pipeline job status states."""
    PENDING = "PENDING"           # Job created, not started
    VALIDATING = "VALIDATING"     # Running guardrail validation
    VALIDATED = "VALIDATED"       # Passed guardrail, queued for processing
    BLOCKED = "BLOCKED"           # Blocked by guardrail
    PROCESSING = "PROCESSING"     # Pipeline is running
    COMPLETED = "COMPLETED"       # All stages completed successfully
    FAILED = "FAILED"             # A stage failed
    CANCELLED = "CANCELLED"       # Job was cancelled


class PipelineStage(str, Enum):
    """Pipeline stages."""
    UPLOAD = "upload"
    GUARDRAIL = "guardrail"
    REMBG = "rembg"
    NANO_BANANA = "nano_banana"
    GENERATION = "generation"
    REALESRGAN = "realesrgan"
    FINALIZE = "finalize"


class StageStatus(str, Enum):
    """Individual stage status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ImageJob(SQLModel, table=True):
    """
    Enhanced ImageJob model for pipeline tracking.
    
    Stores:
    - Input/output storage keys
    - Per-stage status and metadata
    - Timing and cost information
    - Error tracking
    """
    __tablename__ = "image_jobs"
    
    # Primary Key
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    
    # User association
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    
    # Input Data
    prompt: str = Field(default="")
    original_filename: Optional[str] = None
    
    # Pipeline Status
    status: str = Field(default=JobStatus.PENDING.value)
    current_stage: Optional[str] = None
    
    # Storage Keys (relative paths in storage)
    original_storage_key: Optional[str] = None
    transparent_storage_key: Optional[str] = None  # After rembg
    upscaled_storage_key: Optional[str] = None     # After realesrgan
    generated_storage_key: Optional[str] = None    # After generation
    final_storage_key: Optional[str] = None        # After nano_banana
    
    # Guardrail Result
    guardrail_status: Optional[str] = None  # PASS or BLOCK
    guardrail_reasons: Optional[str] = None
    guardrail_scores: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # Per-Stage Metadata
    # Structure: {stage_name: {status, duration_ms, started_at, completed_at, ...}}
    stages_metadata: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # AI/Processing Metadata
    ai_metadata: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # Error Tracking
    error_message: Optional[str] = None
    error_stage: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Cost Tracking
    total_cost_usd: float = Field(default=0.0)
    
    # Performance Tracking
    total_processing_time_ms: int = Field(default=0)
    
    # Celery Task ID
    celery_task_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def update_stage(
        self,
        stage: str,
        status: str,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Update a specific stage's status and metadata."""
        # Create a copy to ensure SQLAlchemy detects the change
        current_metadata = dict(self.stages_metadata) if self.stages_metadata else {}
        
        stage_data = current_metadata.get(stage, {}).copy()
        stage_data["status"] = status
        
        if status == StageStatus.IN_PROGRESS.value:
            stage_data["started_at"] = datetime.utcnow().isoformat()
        elif status in [StageStatus.COMPLETED.value, StageStatus.FAILED.value]:
            stage_data["completed_at"] = datetime.utcnow().isoformat()
        
        if duration_ms is not None:
            stage_data["duration_ms"] = duration_ms
        
        if metadata:
            stage_data.update(metadata)
        
        if error:
            stage_data["error"] = error
        
        current_metadata[stage] = stage_data
        self.stages_metadata = current_metadata  # Replace dict to trigger update
        self.current_stage = stage
        self.updated_at = datetime.utcnow()
    
    def mark_started(self):
        """Mark job as started."""
        self.status = JobStatus.PROCESSING.value
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def mark_completed(self, total_cost: float = 0.0):
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED.value
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.total_cost_usd = total_cost
        
        # Calculate total processing time
        if self.started_at:
            self.total_processing_time_ms = int(
                (self.completed_at - self.started_at).total_seconds() * 1000
            )
    
    def mark_failed(self, error_message: str, error_stage: str, traceback: Optional[str] = None):
        """Mark job as failed."""
        self.status = JobStatus.FAILED.value
        self.error_message = error_message
        self.error_stage = error_stage
        self.error_traceback = traceback
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Update stage metadata
        self.update_stage(error_stage, StageStatus.FAILED.value, error=error_message)
    
    def mark_blocked(self, reasons: List[str], scores: Dict[str, float]):
        """Mark job as blocked by guardrail."""
        self.status = JobStatus.BLOCKED.value
        self.guardrail_status = "BLOCK"
        self.guardrail_reasons = ", ".join(reasons)
        self.guardrail_scores = scores
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def get_stage_status(self, stage: str) -> Optional[str]:
        """Get the status of a specific stage."""
        if self.stages_metadata and stage in self.stages_metadata:
            return self.stages_metadata[stage].get("status")
        return None
    
    def get_pipeline_progress(self) -> Dict[str, Any]:
        """Get pipeline progress summary."""
        stages = [
            PipelineStage.UPLOAD,
            PipelineStage.GUARDRAIL,
            PipelineStage.REMBG,
            PipelineStage.NANO_BANANA,
            PipelineStage.GENERATION,
            PipelineStage.REALESRGAN,
            PipelineStage.FINALIZE
        ]
        
        progress = []
        for stage in stages:
            stage_name = stage.value
            stage_data = self.stages_metadata.get(stage_name, {}) if self.stages_metadata else {}
            
            progress.append({
                "stage": stage_name,
                "status": stage_data.get("status", StageStatus.PENDING.value),
                "duration_ms": stage_data.get("duration_ms"),
                "started_at": stage_data.get("started_at"),
                "completed_at": stage_data.get("completed_at"),
                "error": stage_data.get("error")
            })
        
        return {
            "job_id": self.id,
            "status": self.status,
            "current_stage": self.current_stage,
            "stages": progress,
            "total_cost_usd": self.total_cost_usd,
            "total_processing_time_ms": self.total_processing_time_ms
        }
    
    def to_response_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "id": self.id,
            "status": self.status,
            "current_stage": self.current_stage,
            "prompt": self.prompt,
            "original_filename": self.original_filename,
            "guardrail": {
                "status": self.guardrail_status,
                "reasons": self.guardrail_reasons.split(", ") if self.guardrail_reasons else [],
                "scores": self.guardrail_scores
            } if self.guardrail_status else None,
            "storage_keys": {
                "original": self.original_storage_key,
                "transparent": self.transparent_storage_key,
                "upscaled": self.upscaled_storage_key,
                "generated": self.generated_storage_key,
                "final": self.final_storage_key
            },
            "error": {
                "message": self.error_message,
                "stage": self.error_stage
            } if self.error_message else None,
            "cost_usd": self.total_cost_usd,
            "processing_time_ms": self.total_processing_time_ms,
            "stages": self.stages_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
