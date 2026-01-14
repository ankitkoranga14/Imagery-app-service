"""
ValidationLogs Model for Guardrail Microservice

Stores every validation check performed, including:
- Image URL/hash
- Guardrail scores for all levels
- Pass/Block status with reason
- Latency metrics
"""

import uuid
import hashlib
from sqlmodel import SQLModel, Field, Column, JSON
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASS = "PASS"
    BLOCK = "BLOCK"


class ValidationLog(SQLModel, table=True):
    """
    ValidationLog model for audit trail of all guardrail checks.
    
    Stores:
    - Image identification (URL or hash)
    - Prompt text
    - Validation result and reason
    - Per-level scores (L1-L4)
    - Latency breakdown
    """
    __tablename__ = "validation_logs"
    
    # Primary Key
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    
    # Request identification
    image_hash: str = Field(index=True, description="SHA256 hash of the image for deduplication")
    image_url: Optional[str] = Field(default=None, description="Optional URL if image was fetched")
    prompt: str = Field(default="", description="Text prompt for validation")
    
    # Validation Result
    status: str = Field(default=ValidationStatus.PASS.value, description="PASS or BLOCK")
    failure_reason: Optional[str] = Field(default=None, description="Reason for blocking, null if passed")
    failure_level: Optional[str] = Field(default=None, description="Which level caused the block (L1/L2/L3/L4)")
    exit_level: Optional[str] = Field(default=None, description="Specific exit point in the tiered flow (e.g., L3_EARLY_EXIT)")
    
    # Scores from each validation level
    # L1: Text validation (injection, policy, food domain)
    # L2: Physics (brightness, glare, blur, contrast)
    # L3: Geometry (YOLO food detection)
    # L4: Context (CLIP food/NSFW detection)
    scores: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # Latency breakdown
    latency_ms: int = Field(default=0, description="Total processing time in milliseconds")
    latency_breakdown: Dict[str, float] = Field(
        default={},
        sa_column=Column(JSON),
        description="Per-level latency in ms: {cache_ms, l1_ms, l2_ms, l3_ms, l4_ms}"
    )
    
    # Execution metadata
    cache_hit: bool = Field(default=False, description="Whether result was served from cache")
    parallel_execution: bool = Field(default=False, description="Whether L3/L4 ran in parallel")
    levels_executed: str = Field(default="", description="Comma-separated list of executed levels")
    
    # Model variants used (for A/B testing)
    clip_variant: Optional[str] = Field(default=None, description="CLIP model variant used")
    yolo_variant: Optional[str] = Field(default=None, description="YOLO model variant used")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    @classmethod
    def compute_image_hash(cls, image_bytes: bytes) -> str:
        """Compute SHA256 hash of image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()
    
    def to_response_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "id": self.id,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "failure_level": self.failure_level,
            "scores": self.scores,
            "latency_ms": self.latency_ms,
            "latency_breakdown": self.latency_breakdown,
            "cache_hit": self.cache_hit,
            "parallel_execution": self.parallel_execution,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ValidationJob(SQLModel, table=True):
    """
    Async validation job for when client doesn't want to wait.
    
    Used for async mode where client gets a job_id and polls for result.
    """
    __tablename__ = "validation_jobs"
    
    # Primary Key
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    
    # Job state
    status: str = Field(default="PENDING", description="PENDING, PROCESSING, COMPLETED, FAILED")
    
    # Request data (stored for async processing)
    prompt: str = Field(default="")
    image_hash: str = Field(default="")
    image_storage_key: Optional[str] = Field(default=None, description="Where image is stored temporarily")
    
    # Result (populated when completed)
    validation_log_id: Optional[str] = Field(default=None, foreign_key="validation_logs.id")
    result_status: Optional[str] = Field(default=None, description="PASS or BLOCK")
    result_reason: Optional[str] = Field(default=None)
    result_scores: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    result_latency_ms: Optional[int] = Field(default=None)
    
    # Error tracking
    error_message: Optional[str] = Field(default=None)
    
    # Celery task tracking
    celery_task_id: Optional[str] = Field(default=None)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    completed_at: Optional[datetime] = Field(default=None)
    
    def mark_completed(
        self,
        validation_log_id: str,
        status: str,
        reason: Optional[str],
        scores: Dict[str, Any],
        latency_ms: int
    ):
        """Mark job as completed with result."""
        self.status = "COMPLETED"
        self.validation_log_id = validation_log_id
        self.result_status = status
        self.result_reason = reason
        self.result_scores = scores
        self.result_latency_ms = latency_ms
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error_message: str):
        """Mark job as failed."""
        self.status = "FAILED"
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
    
    def to_response_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "job_id": self.id,
            "status": self.status,
            "result": {
                "status": self.result_status,
                "failure_reason": self.result_reason,
                "scores": self.result_scores,
                "latency_ms": self.result_latency_ms
            } if self.status == "COMPLETED" else None,
            "error": self.error_message if self.status == "FAILED" else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
