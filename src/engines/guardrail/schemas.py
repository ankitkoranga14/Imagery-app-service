from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class GuardrailStatus(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"


class GuardrailRequestDTO(BaseModel):
    """Request for guardrail validation."""
    prompt: str = Field(..., max_length=2000)
    image_bytes: Optional[str] = Field(None)  # base64 encoded


class BatchGuardrailRequestDTO(BaseModel):
    """Batch request for multiple guardrail validations."""
    requests: List[GuardrailRequestDTO] = Field(..., max_items=50)
    max_concurrent: int = Field(default=5, ge=1, le=10)


class GuardrailScoresDTO(BaseModel):
    """Detailed scores from the guardrail validation pipeline.
    
    All fields are optional since they depend on which validation levels were executed.
    """
    # Text validation scores
    injection_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prompt injection detection score")
    semantic_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Semantic similarity to known injections")
    policy_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Policy violation score")
    food_domain_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Food domain relevance score")
    
    # Physics (OpenCV) scores - Enhanced
    darkness_score: Optional[float] = Field(None, ge=0, le=255, description="Mean luminance (0-255)")
    glare_percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Percentage of overexposed pixels")
    blur_variance: Optional[float] = Field(None, ge=0, description="Laplacian variance for blur detection")
    contrast_std_dev: Optional[float] = Field(None, ge=0, description="Standard deviation of luminance (contrast)")
    dynamic_range: Optional[float] = Field(None, ge=0, description="Dynamic range (p95 - p5)")
    
    # Geometry (YOLO) scores - Enhanced
    food_object_count: Optional[int] = Field(None, ge=0, description="Number of food objects detected")
    distinct_dish_count: Optional[int] = Field(None, ge=0, description="Number of spatially distinct dishes")
    liquid_dish_count: Optional[int] = Field(None, ge=0, description="Number of liquid dishes (soups, beverages)")
    
    # Context (CLIP) scores - Enhanced
    food_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Composite food classification score")
    ready_to_eat_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Ready-to-eat meal probability")
    nsfw_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="NSFW content probability")
    
    # Advanced Context (CLIP) scores - Enhanced
    angle_quality_score: Optional[float] = Field(None, description="Image angle quality (+ top-down, - side-view)")
    quality_score: Optional[float] = Field(None, description="Overall image quality score")
    foreign_object_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Foreign object detection probability")


class PhysicsResultDTO(BaseModel):
    """Detailed physics check results."""
    is_too_dark: bool = False
    is_glary: bool = False
    is_blurry: bool = False
    has_contrast: bool = True
    
    darkness_reason: Optional[str] = None
    glare_reason: Optional[str] = None
    contrast_reason: Optional[str] = None
    
    # Metrics
    mean_brightness: Optional[float] = None
    glare_percentage: Optional[float] = None
    contrast_std_dev: Optional[float] = None
    dynamic_range: Optional[float] = None


class GeometryResultDTO(BaseModel):
    """Detailed geometry check results."""
    food_object_count: int = 0
    distinct_dish_count: int = 0
    main_dish_count: int = 0
    main_dish_clusters: int = 0
    surfaces_detected: int = 0
    liquid_dish_count: int = 0
    expected_count: int = 1
    
    detected_foods: List[Dict[str, Any]] = Field(default_factory=list)


class ContextResultDTO(BaseModel):
    """Detailed context check results."""
    is_food: bool = False
    is_nsfw: bool = False
    is_raw_food: bool = False
    is_packaged_food: bool = False
    is_pet_food: bool = False
    is_spoiled_food: bool = False
    
    top_category: Optional[str] = None
    quality_issues: List[str] = Field(default_factory=list)


class IntermediateResultsDTO(BaseModel):
    """Intermediate results from all validation levels for debugging."""
    text: Optional[Dict[str, Any]] = None
    physics: Optional[Dict[str, Any]] = None
    geometry: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    context_advanced: Optional[Dict[str, Any]] = None


class ValidationTraceDTO(BaseModel):
    """Execution trace showing which validation levels were performed."""
    levels_executed: List[str] = Field(default_factory=list, description="All levels that were executed")
    levels_passed: List[str] = Field(default_factory=list, description="Levels that passed validation")
    levels_failed: List[str] = Field(default_factory=list, description="Levels that failed validation")
    levels_skipped: List[str] = Field(default_factory=list, description="Levels that were skipped")
    timings: Dict[str, float] = Field(default_factory=dict, description="Time taken for each level in ms")
    detected_foods: Optional[List[Dict[str, Any]]] = Field(None, description="Food objects detected by YOLO")
    intermediate_results: Optional[Dict[str, Any]] = Field(None, description="Detailed intermediate results")


class GuardrailMetadataDTO(BaseModel):
    """Metadata about the validation process."""
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    cache_hit: bool = Field(False, description="Whether the result was served from cache")
    validation_trace: Optional[ValidationTraceDTO] = Field(None, description="Detailed execution trace")


class GuardrailResponseDTO(BaseModel):
    """Response from the guardrail validation pipeline."""
    status: GuardrailStatus = Field(..., description="PASS or BLOCK")
    reasons: List[str] = Field(default_factory=list, description="List of failure reasons (empty if PASS)")
    scores: Dict[str, Union[float, int]] = Field(default_factory=dict, description="Validation scores from all levels")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata including validation trace")


class BatchGuardrailResponseDTO(BaseModel):
    """Response from batch guardrail validation."""
    results: List[GuardrailResponseDTO] = Field(default_factory=list)
    total_processing_time_ms: int = Field(0, description="Total time for all validations")
    success_count: int = Field(0, description="Number of PASS results")
    block_count: int = Field(0, description="Number of BLOCK results")


# =============================================================================
# Feedback Schemas
# =============================================================================

class FeedbackType(str, Enum):
    """Types of feedback for guardrail validation."""
    FALSE_POSITIVE = "false_positive"  # Blocked but should have passed
    FALSE_NEGATIVE = "false_negative"  # Passed but should have blocked
    CORRECT = "correct"  # Validation was correct


class GuardrailFeedbackRequestDTO(BaseModel):
    """Request to submit feedback on a guardrail validation."""
    validation_id: Optional[int] = Field(None, description="ID of the original validation log")
    input_hash: Optional[str] = Field(None, description="Hash of the original input")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    failed_level: Optional[str] = Field(None, description="Which validation level was wrong")
    user_comment: Optional[str] = Field(None, max_length=1000, description="Optional user comment")


class GuardrailFeedbackResponseDTO(BaseModel):
    """Response after submitting feedback."""
    success: bool
    feedback_id: Optional[int] = None
    message: str


# =============================================================================
# Config Variant Schemas (for A/B Testing)
# =============================================================================

class ConfigVariantDTO(BaseModel):
    """Configuration variant for A/B testing."""
    name: str = Field(..., description="Unique variant name")
    description: Optional[str] = None
    
    # Physics thresholds
    darkness_threshold: Optional[float] = Field(None, ge=0, le=100)
    glare_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    contrast_threshold: Optional[float] = Field(None, ge=0, le=100)
    
    # Context thresholds
    food_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    nsfw_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    foreign_object_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    poor_angle_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Geometry thresholds
    min_main_dishes: Optional[int] = Field(None, ge=1, le=10)
    min_any_dishes: Optional[int] = Field(None, ge=1, le=20)
    
    # Rollout
    rollout_percentage: int = Field(default=0, ge=0, le=100)
    is_active: bool = Field(default=True)


class ConfigVariantStatsDTO(BaseModel):
    """Statistics for a configuration variant."""
    name: str
    total_validations: int = 0
    pass_count: int = 0
    block_count: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: Optional[float] = None  # TP / (TP + FP)
    recall: Optional[float] = None  # TP / (TP + FN)
