"""
Guardrail Validation Endpoints

Synchronous guardrails executed inside the FastAPI request cycle.
"""

from fastapi import APIRouter, Depends, HTTPException
from src.engines.guardrail.schemas import GuardrailRequestDTO, GuardrailResponseDTO
from src.engines.guardrail.services import GuardrailService
from src.api.dependencies import get_guardrail_service
from src.core.logging import get_logger
from src.core.metrics import (
    guardrail_validations_total,
    guardrail_latency_seconds,
    record_guardrail_validation
)
import time

logger = get_logger(__name__)
router = APIRouter()


@router.post("/validate", response_model=GuardrailResponseDTO)
async def validate_guardrail(
    request: GuardrailRequestDTO,
    service: GuardrailService = Depends(get_guardrail_service)
):
    """
    Validate content against guardrail rules.
    
    Performs synchronous validation including:
    - CLIP-based NSFW/Food detection for images
    - Transformer-based injection checks for text
    - Policy and domain validation
    
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


@router.post("/validate/batch")
async def validate_batch(
    requests: list[GuardrailRequestDTO],
    service: GuardrailService = Depends(get_guardrail_service)
):
    """
    Validate multiple items against guardrail rules.
    
    Returns:
        List of GuardrailResponseDTO results
    """
    results = []
    
    for request in requests:
        try:
            result = await service.validate(request)
            results.append({
                "index": len(results),
                "result": result
            })
        except Exception as e:
            results.append({
                "index": len(results),
                "error": str(e)
            })
    
    return {"results": results, "total": len(requests)}
