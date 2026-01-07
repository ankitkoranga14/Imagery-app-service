import uuid
from sqlmodel import SQLModel, Field, Column, JSON
from typing import Optional, Dict, Any
from datetime import datetime

class ImageJob(SQLModel, table=True):
    __tablename__ = "image_jobs"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    
    # Input Data
    prompt: str
    status: str = Field(default="PENDING") # PENDING, PROCESSING, COMPLETED, FAILED
    
    # Azure Blob Storage URLs
    original_url: Optional[str] = None
    transparent_url: Optional[str] = None
    upscaled_url: Optional[str] = None
    final_url: Optional[str] = None
    
    # AI Metadata
    guardrail_result: Optional[str] = None
    ai_metadata: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # Performance Tracking
    processing_time_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
