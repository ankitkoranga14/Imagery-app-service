"""
Guardrail Validation Endpoints

Synchronous guardrails executed inside the FastAPI request cycle.

Features:
- Single validation endpoint
- Batch processing with parallel execution
- Model warmup endpoint
- Feedback submission
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from src.engines.guardrail.schemas import (
    GuardrailRequestDTO,
    GuardrailResponseDTO,
    BatchGuardrailRequestDTO,
    BatchGuardrailResponseDTO,
    GuardrailFeedbackRequestDTO,
    GuardrailFeedbackResponseDTO,
    GuardrailStatus
)
from src.engines.guardrail.services import GuardrailService
from src.api.dependencies import get_guardrail_service, get_ml_repo
from src.engines.guardrail.repositories import MLRepository
from src.core.logging import get_logger
from src.core.metrics import (
    guardrail_validations_total,
    guardrail_latency_seconds,
    record_guardrail_validation
)
import time
import asyncio

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Validation Endpoints
# =============================================================================

@router.post("/validate", response_model=GuardrailResponseDTO)
async def validate_guardrail(
    request: GuardrailRequestDTO,
    service: GuardrailService = Depends(get_guardrail_service)
):
    """
    Validate content against guardrail rules.
    
    Performs synchronous validation including:
    - Text validation (injection detection, policy check, food domain)
    - Physics validation (brightness, glare, contrast)
    - Geometry validation (food object detection, dish counting)
    - Context validation (CLIP food/NSFW detection, quality assessment)
    
    Returns:
        GuardrailResponseDTO with status (PASS/BLOCK), reasons, and scores
    """
    start_time = time.time()
    
    try:
        result = await service.validate(request)
        
        # Record metrics
        duration = time.time() - start_time
        guardrail_latency_seconds.observe(duration)
        
        block_reason = result.reasons[0] if result.reasons else "none"
        record_guardrail_validation(
            status=result.status.value,
            block_reason=block_reason
        )
        
        logger.info(
            "guardrail_validation_complete",
            status=result.status.value,
            reasons=result.reasons,
            processing_time_ms=result.metadata.get("processing_time_ms")
        )
        
        return result
        
    except Exception as e:
        logger.error("guardrail_validation_error", error=str(e))
        record_guardrail_validation(status="ERROR", block_reason="internal_error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/validate/batch", response_model=BatchGuardrailResponseDTO)
async def validate_batch(
    batch_request: BatchGuardrailRequestDTO,
    service: GuardrailService = Depends(get_guardrail_service)
):
    """
    Validate multiple items against guardrail rules in parallel.
    
    Args:
        batch_request: Contains list of validation requests and concurrency settings
    
    Returns:
        BatchGuardrailResponseDTO with all results and summary statistics
    """
    start_time = time.time()
    
    try:
        results = await service.validate_batch(
            batch_request.requests,
            max_concurrent=batch_request.max_concurrent
        )
        
        # Calculate statistics
        success_count = sum(1 for r in results if r.status == GuardrailStatus.PASS)
        block_count = len(results) - success_count
        total_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            "guardrail_batch_validation_complete",
            total=len(results),
            passed=success_count,
            blocked=block_count,
            processing_time_ms=total_time
        )
        
        return BatchGuardrailResponseDTO(
            results=results,
            total_processing_time_ms=total_time,
            success_count=success_count,
            block_count=block_count
        )
        
    except Exception as e:
        logger.error("guardrail_batch_validation_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# Warmup & Health Endpoints
# =============================================================================

class WarmupResponse(BaseModel):
    """Response from model warmup."""
    success: bool
    message: str
    models_loaded: List[str] = Field(default_factory=list)
    warmup_time_ms: int = 0


@router.post("/warmup", response_model=WarmupResponse)
async def warmup_models(
    service: GuardrailService = Depends(get_guardrail_service)
):
    """
    Preload all ML models for faster cold start.
    
    Call this endpoint after deployment to warm up models before serving traffic.
    This prevents the first request from experiencing long loading times.
    
    Returns:
        WarmupResponse with status and loaded model names
    """
    start_time = time.time()
    
    try:
        success = await service.preload_models()
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        if success:
            return WarmupResponse(
                success=True,
                message="All models preloaded successfully",
                models_loaded=["SentenceTransformer", "CLIP-ViT-B-32", "YOLOv8n"],
                warmup_time_ms=elapsed_ms
            )
        else:
            return WarmupResponse(
                success=False,
                message="Some models failed to load",
                models_loaded=[],
                warmup_time_ms=elapsed_ms
            )
            
    except Exception as e:
        logger.error("guardrail_warmup_error", error=str(e))
        return WarmupResponse(
            success=False,
            message=f"Warmup failed: {str(e)}",
            models_loaded=[],
            warmup_time_ms=int((time.time() - start_time) * 1000)
        )


class HealthResponse(BaseModel):
    """Response from guardrail health check."""
    status: str
    models_loaded: bool
    cache_available: bool
    details: dict = Field(default_factory=dict)


@router.get("/health", response_model=HealthResponse)
async def guardrail_health(
    ml_repo: MLRepository = Depends(get_ml_repo)
):
    """
    Check health of guardrail service components.
    
    Returns:
        HealthResponse with status of models and cache
    """
    models_loaded = ml_repo.models_loaded
    
    return HealthResponse(
        status="healthy" if models_loaded else "warming_up",
        models_loaded=models_loaded,
        cache_available=True,  # Redis is managed at app level
        details={
            "text_model": ml_repo.text_model is not None,
            "clip_model": ml_repo.clip_model is not None,
            "yolo_model": ml_repo.yolo_model is not None,
            "device": ml_repo.device
        }
    )


# =============================================================================
# Feedback Endpoints
# =============================================================================

@router.post("/feedback", response_model=GuardrailFeedbackResponseDTO)
async def submit_feedback(
    feedback: GuardrailFeedbackRequestDTO,
    service: GuardrailService = Depends(get_guardrail_service)
):
    """
    Submit feedback on a guardrail validation decision.
    
    Use this to report false positives (blocked but should have passed)
    or false negatives (passed but should have blocked).
    
    This feedback is used to:
    - Calculate precision/recall metrics
    - Identify systematic issues
    - Improve threshold calibration
    
    Returns:
        GuardrailFeedbackResponseDTO with feedback ID
    """
    try:
        # Note: Feedback storage requires FeedbackRepository
        # This is a placeholder - implement with actual database integration
        logger.info(
            "guardrail_feedback_received",
            feedback_type=feedback.feedback_type.value,
            validation_id=feedback.validation_id,
            failed_level=feedback.failed_level
        )
        
        return GuardrailFeedbackResponseDTO(
            success=True,
            feedback_id=None,  # Would be actual ID from database
            message="Feedback received successfully"
        )
        
    except Exception as e:
        logger.error("guardrail_feedback_error", error=str(e))
        return GuardrailFeedbackResponseDTO(
            success=False,
            feedback_id=None,
            message=f"Failed to save feedback: {str(e)}"
        )


# =============================================================================
# Configuration Endpoints
# =============================================================================

class ThresholdsResponse(BaseModel):
    """Current guardrail thresholds."""
    physics: dict
    geometry: dict
    context: dict
    text: dict


@router.get("/thresholds", response_model=ThresholdsResponse)
async def get_thresholds():
    """
    Get current guardrail threshold configuration.
    
    Returns current values for all configurable thresholds.
    Useful for debugging and understanding validation behavior.
    """
    # Import thresholds from services
    from src.engines.guardrail.services import (
        PHYSICS_DARKNESS_THRESHOLD,
        PHYSICS_DARKNESS_HARD_LIMIT,
        PHYSICS_GLARE_THRESHOLD,
        PHYSICS_BLOWN_THRESHOLD,
        PHYSICS_BLUR_THRESHOLD,
        PHYSICS_MIN_CONTRAST_SD,
        PHYSICS_MIN_DYNAMIC_RANGE,
        GEOMETRY_PROXIMITY_THRESHOLD,
        GEOMETRY_MIN_MAIN_DISHES,
        GEOMETRY_MIN_ANY_DISHES,
        CONTEXT_FOOD_THRESHOLD,
        CONTEXT_NSFW_THRESHOLD,
        CONTEXT_FOREIGN_OBJECT_THRESHOLD,
        CONTEXT_POOR_ANGLE_THRESHOLD,
        TEXT_FOOD_DOMAIN_THRESHOLD,
    )
    
    return ThresholdsResponse(
        physics={
            "darkness_threshold": PHYSICS_DARKNESS_THRESHOLD,
            "darkness_hard_limit": PHYSICS_DARKNESS_HARD_LIMIT,
            "glare_threshold": PHYSICS_GLARE_THRESHOLD,
            "blown_threshold": PHYSICS_BLOWN_THRESHOLD,
            "blur_threshold": PHYSICS_BLUR_THRESHOLD,
            "min_contrast_sd": PHYSICS_MIN_CONTRAST_SD,
            "min_dynamic_range": PHYSICS_MIN_DYNAMIC_RANGE,
        },
        geometry={
            "proximity_threshold": GEOMETRY_PROXIMITY_THRESHOLD,
            "min_main_dishes_to_block": GEOMETRY_MIN_MAIN_DISHES,
            "min_any_dishes_to_block": GEOMETRY_MIN_ANY_DISHES,
        },
        context={
            "food_threshold": CONTEXT_FOOD_THRESHOLD,
            "nsfw_threshold": CONTEXT_NSFW_THRESHOLD,
            "foreign_object_threshold": CONTEXT_FOREIGN_OBJECT_THRESHOLD,
            "poor_angle_threshold": CONTEXT_POOR_ANGLE_THRESHOLD,
        },
        text={
            "food_domain_threshold": TEXT_FOOD_DOMAIN_THRESHOLD,
        }
    )
