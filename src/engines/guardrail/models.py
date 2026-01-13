from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class GuardrailLog(SQLModel, table=True):
    """Log entry for guardrail validation requests."""
    __tablename__ = "guardrail_logs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    prompt: str
    status: str  # PASS, BLOCK
    reasons: Optional[str] = None
    processing_time_ms: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Enhanced logging fields
    input_hash: Optional[str] = Field(default=None, index=True)
    levels_executed: Optional[str] = Field(default=None)  # Comma-separated
    levels_failed: Optional[str] = Field(default=None)  # Comma-separated
    
    # Intermediate scores for analysis
    darkness_score: Optional[float] = Field(default=None)
    glare_percentage: Optional[float] = Field(default=None)
    food_score: Optional[float] = Field(default=None)
    nsfw_score: Optional[float] = Field(default=None)
    dish_count: Optional[int] = Field(default=None)


class FeedbackType(str, Enum):
    """Types of feedback for guardrail validation."""
    FALSE_POSITIVE = "false_positive"  # Blocked but should have passed
    FALSE_NEGATIVE = "false_negative"  # Passed but should have blocked
    CORRECT = "correct"  # Validation was correct


class GuardrailFeedback(SQLModel, table=True):
    """User feedback on guardrail validation decisions.
    
    Used to:
    1. Calculate precision/recall per validation level
    2. A/B test threshold changes
    3. Identify systematic failures
    4. Improve model accuracy over time
    """
    __tablename__ = "guardrail_feedback"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Link to original validation
    validation_id: Optional[int] = Field(default=None, foreign_key="guardrail_logs.id")
    input_hash: Optional[str] = Field(default=None, index=True)
    
    # User info
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    
    # Feedback details
    feedback_type: str  # FeedbackType value
    failed_level: Optional[str] = Field(default=None)  # Which level was wrong
    user_comment: Optional[str] = Field(default=None)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional: Store the original request for analysis
    original_prompt: Optional[str] = Field(default=None)
    original_status: Optional[str] = Field(default=None)


class GuardrailConfigVariant(SQLModel, table=True):
    """A/B testing variants for guardrail configuration.
    
    Allows testing different threshold configurations
    without affecting the entire user base.
    """
    __tablename__ = "guardrail_config_variants"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: Optional[str] = Field(default=None)
    
    # Threshold overrides (JSON string)
    config_json: str  # JSON with threshold overrides
    
    # Rollout percentage (0-100)
    rollout_percentage: int = Field(default=0)
    
    # Status
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    
    # Performance metrics
    total_validations: int = Field(default=0)
    false_positives: int = Field(default=0)
    false_negatives: int = Field(default=0)
