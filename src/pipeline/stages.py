"""
Pipeline Stage Implementations - Guardrail Microservice

This module now contains only the guardrail validation stage.
All generative pipeline stages have been removed.

Validation Pipeline:
- L0: Cache check (instant)
- L1: Text validation (NLP)
- L2: Physics check (OpenCV)
- L3: Geometry check (YOLO)
- L4: Context check (CLIP)
"""

import io
import base64
import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from src.core.logging import get_logger, set_job_context
from src.core.metrics import track_stage_latency
from src.core.exceptions import PipelineStageError

from src.engines.guardrail.services import GuardrailService
from src.engines.guardrail.repositories import MLRepository, CacheRepository, LogRepository
from src.engines.guardrail.schemas import GuardrailRequestDTO, GuardrailStatus
from src.engines.guardrail.services import TextGuardrailService, ImageGuardrailService

logger = get_logger(__name__)


# =============================================================================
# Stage 0: Quality Check (Guardrail) - Primary validation entry point
# =============================================================================

async def process_guardrail_stage(
    prompt: str,
    image_bytes: Optional[bytes],
    job_id: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate prompt and image using GuardrailService.
    
    This is the primary validation function for the Guardrail Microservice.
    
    Implements fail-fast logic:
    - L0: Cache check (instant)
    - L1: Text validation (injection, policy, food domain)
    - L2: Physics check (brightness, glare, blur, contrast)
    - L3/L4: Geometry + Context (run parallel if L2 borderline)
    
    Args:
        prompt: Text prompt describing expected content
        image_bytes: Raw image bytes (optional)
        job_id: Job ID for logging
        
    Returns:
        Tuple of (is_passed, metadata)
        - is_passed: True if validation passed, False if blocked
        - metadata: Dict with scores, reasons, and latency info
    """
    set_job_context(job_id, "guardrail")
    start_time = datetime.utcnow()
    
    try:
        # Initialize repositories and services (lazy loading)
        from src.core.config import settings
        
        # Use singleton MLRepository to avoid re-downloading models
        ml_repo = MLRepository.get_instance(settings.ML_MODEL_CACHE_DIR)
        
        # Note: CacheRepository and LogRepository require dependencies (Redis, DB Session)
        # that are not easily available here without a larger refactor.
        # For now, we initialize them with None/Mock to prevent immediate crashes,
        # but this stage might need further work to fully support logging/caching.
        # The primary fix here is for MLRepository downloads.
        
        # Mock/Placeholder for now as we focus on ML model loading
        class MockRepo:
            async def get(self, *args, **kwargs): return None
            async def set(self, *args, **kwargs): pass
            async def save(self, *args, **kwargs): pass
            async def compute_hash(self, *args, **kwargs): return "hash"
            
        cache_repo = MockRepo()
        log_repo = MockRepo()
        
        text_service = TextGuardrailService(ml_repo)
        image_service = ImageGuardrailService(ml_repo)
        
        guardrail_service = GuardrailService(
            ml_repo, cache_repo, log_repo, text_service, image_service
        )

        
        # Prepare request
        image_b64 = None
        if image_bytes:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
        request = GuardrailRequestDTO(
            prompt=prompt,
            image_bytes=image_b64
        )
        
        # Validate
        with track_stage_latency("guardrail"):
            response = await guardrail_service.validate(request)
        
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        is_passed = response.status == GuardrailStatus.PASS
        
        # Build metadata
        metadata = {
            "stage": "guardrail",
            "status": response.status.value,
            "reasons": response.reasons,
            "scores": response.scores,
            "duration_ms": duration_ms,
            "cache_hit": response.metadata.get("cache_hit", False),
            "validation_trace": response.metadata.get("validation_trace", {}),
            "model_variants": response.metadata.get("model_variants", {})
        }
        
        logger.info(
            "guardrail_completed",
            status=response.status.value,
            duration_ms=duration_ms,
            passed=is_passed,
            cache_hit=metadata["cache_hit"]
        )
        
        return is_passed, metadata
        
    except Exception as e:
        logger.error("guardrail_failed", error=str(e))
        raise PipelineStageError(
            f"Guardrail validation failed: {str(e)}",
            stage="guardrail",
            job_id=job_id
        )


async def validate_image(
    prompt: str,
    image_bytes: bytes,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for direct validation without Celery.
    
    Args:
        prompt: Text prompt
        image_bytes: Raw image bytes
        job_id: Optional job ID (generated if not provided)
        
    Returns:
        Validation result dict with status, scores, and latency
    """
    import uuid
    
    if not job_id:
        job_id = str(uuid.uuid4())
    
    is_passed, metadata = await process_guardrail_stage(prompt, image_bytes, job_id)
    
    return {
        "status": "PASS" if is_passed else "BLOCK",
        "failure_reason": ", ".join(metadata.get("reasons", [])) if not is_passed else None,
        "scores": metadata.get("scores", {}),
        "latency_ms": metadata.get("duration_ms", 0),
        "cache_hit": metadata.get("cache_hit", False),
        "validation_trace": metadata.get("validation_trace", {})
    }


# =============================================================================
# REMOVED - Generative Pipeline Stages
# =============================================================================
# The following stages have been removed from the Guardrail Microservice:
#
# - process_rembg_stage: Background removal using rembg
# - process_realesrgan_stage: 4K upscaling using RealESRGAN
# - process_nano_banana_stage: Smart placement via Nano Banana API
# - process_generation_stage: Hero image generation
#
# And their simulation counterparts:
# - simulate_rembg_stage
# - simulate_realesrgan_stage
# - simulate_nano_banana_stage
# - simulate_generation_stage
#
# These stages are part of the generative image pipeline and should be
# implemented in a separate "Generation Microservice" if needed.
#
# The Generation Microservice should:
# 1. Call this Validation Microservice first (POST /validate)
# 2. Only proceed with generation if validation passes
# 3. Handle GPU resources separately for generation workloads
# =============================================================================
