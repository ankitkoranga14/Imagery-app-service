from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class GuardrailLog(SQLModel, table=True):
    __tablename__ = "guardrail_logs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    prompt: str
    status: str  # PASS, BLOCK
    reasons: Optional[str] = None
    processing_time_ms: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
