"""
Pipeline Stage Implementations

Each stage is a separate function that can be called independently.
Implements the actual processing logic with GPU handling.
"""

import io
import base64
import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from PIL import Image

import httpx

from src.core.logging import get_logger, LogContext, set_job_context
from src.core.metrics import (
    track_stage_latency,
    update_gpu_metrics,
    record_nano_banana_call,
    gpu_vram_usage_gb_gauge
)
from src.core.exceptions import (
    PipelineStageError,
    ExternalAPIError,
    GPUMemoryError,
    get_circuit_breaker
)
from src.core.config import settings

from src.engines.guardrail.services import GuardrailService
from src.engines.guardrail.repositories import MLRepository, CacheRepository, LogRepository
from src.engines.guardrail.schemas import GuardrailRequestDTO, GuardrailStatus
from src.engines.guardrail.services import TextGuardrailService, ImageGuardrailService

logger = get_logger(__name__)


# =============================================================================
# Stage 0: Quality Check (Guardrail)
# =============================================================================

async def process_guardrail_stage(
    prompt: str,
    image_bytes: Optional[bytes],
    job_id: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate prompt and image using GuardrailService.
    
    Returns:
        Tuple of (is_passed, metadata)
    """
    set_job_context(job_id, "guardrail")
    start_time = datetime.utcnow()
    
    try:
        # Initialize repositories and services
        ml_repo = MLRepository()
        cache_repo = CacheRepository()
        log_repo = LogRepository()
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
        
        metadata = {
            "stage": "guardrail",
            "status": response.status.value,
            "reasons": response.reasons,
            "scores": response.scores,
            "duration_ms": duration_ms,
            "cache_hit": response.metadata.get("cache_hit", False)
        }
        
        logger.info(
            "guardrail_completed",
            status=response.status.value,
            duration_ms=duration_ms,
            passed=is_passed
        )
        
        return is_passed, metadata
        
    except Exception as e:
        logger.error("guardrail_failed", error=str(e))
        raise PipelineStageError(
            f"Guardrail validation failed: {str(e)}",
            stage="guardrail",
            job_id=job_id
        )



# =============================================================================
# Stage 1: Background Removal (Rembg)
# =============================================================================

def process_rembg_stage(
    image_bytes: bytes,
    job_id: str
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Remove background from image using rembg.
    
    Args:
        image_bytes: Input image as bytes
        job_id: Job ID for logging
        
    Returns:
        Tuple of (processed_image_bytes, metadata)
    """
    set_job_context(job_id, "rembg")
    circuit = get_circuit_breaker("rembg")
    
    if not circuit.can_execute():
        raise PipelineStageError(
            "Rembg service is temporarily unavailable",
            stage="rembg",
            job_id=job_id
        )
    
    start_time = datetime.utcnow()
    
    try:
        # Import rembg (lazy import for worker optimization)
        from rembg import remove
        
        logger.info("rembg_starting", input_size=len(image_bytes))
        
        # Load image
        input_image = Image.open(io.BytesIO(image_bytes))
        original_size = input_image.size
        
        # Process with rembg
        with track_stage_latency("rembg"):
            output_image = remove(input_image)
        
        # Convert to bytes
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_bytes = output_buffer.getvalue()
        
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        metadata = {
            "stage": "rembg",
            "input_size": len(image_bytes),
            "output_size": len(output_bytes),
            "original_dimensions": original_size,
            "duration_ms": duration_ms
        }
        
        logger.info(
            "rembg_completed",
            duration_ms=duration_ms,
            input_size=len(image_bytes),
            output_size=len(output_bytes)
        )
        
        circuit.record_success()
        update_gpu_metrics("rembg")
        
        return output_bytes, metadata
        
    except MemoryError:
        circuit.record_failure()
        raise GPUMemoryError(job_id=job_id)
        
    except Exception as e:
        circuit.record_failure(e)
        logger.error("rembg_failed", error=str(e))
        raise PipelineStageError(
            f"Background removal failed: {str(e)}",
            stage="rembg",
            job_id=job_id
        )


# =============================================================================
# Stage 2: 4K Tiled Upscaling (RealESRGAN)
# =============================================================================

def process_realesrgan_stage(
    image_bytes: bytes,
    job_id: str,
    scale: int = 4,
    tile_size: int = 512
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Upscale image using RealESRGAN with tiled processing for VRAM limits.
    
    Args:
        image_bytes: Input image as bytes
        job_id: Job ID for logging
        scale: Upscale factor (default 4x)
        tile_size: Tile size for VRAM management (default 512)
        
    Returns:
        Tuple of (upscaled_image_bytes, metadata)
    """
    set_job_context(job_id, "realesrgan")
    circuit = get_circuit_breaker("realesrgan")
    
    if not circuit.can_execute():
        raise PipelineStageError(
            "RealESRGAN service is temporarily unavailable",
            stage="realesrgan",
            job_id=job_id
        )
    
    start_time = datetime.utcnow()
    vram_before = None
    
    try:
        import torch
        import numpy as np
        
        logger.info("realesrgan_starting", input_size=len(image_bytes), scale=scale)
        
        # Check GPU availability and VRAM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            vram_before = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_vram_usage_gb_gauge.labels(device="cuda:0", stage="realesrgan_start").set(vram_before)
        
        # Load image
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = input_image.size
        
        with track_stage_latency("realesrgan"):
            try:
                # Try to import and use Real-ESRGAN
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                # Initialize model
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=scale
                )
                
                upsampler = RealESRGANer(
                    scale=scale,
                    model_path=None,  # Will use default weights
                    model=model,
                    tile=tile_size,
                    tile_pad=10,
                    pre_pad=0,
                    half=device == "cuda"  # Use FP16 on GPU
                )
                
                # Convert PIL to numpy
                input_array = np.array(input_image)
                
                # Process
                output_array, _ = upsampler.enhance(input_array, outscale=scale)
                
                # Convert back to PIL
                output_image = Image.fromarray(output_array)
                
            except ImportError:
                # Fallback: Simple PIL upscale if Real-ESRGAN not installed
                logger.warning(
                    "realesrgan_fallback",
                    message="Real-ESRGAN not installed, using PIL fallback"
                )
                new_size = (original_size[0] * scale, original_size[1] * scale)
                output_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to bytes
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG", quality=95)
        output_bytes = output_buffer.getvalue()
        
        # Get VRAM after processing
        vram_after = None
        if device == "cuda":
            vram_after = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_vram_usage_gb_gauge.labels(device="cuda:0", stage="realesrgan_end").set(vram_after)
            torch.cuda.empty_cache()
        
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        metadata = {
            "stage": "realesrgan",
            "input_size": len(image_bytes),
            "output_size": len(output_bytes),
            "original_dimensions": original_size,
            "output_dimensions": output_image.size,
            "scale": scale,
            "tile_size": tile_size,
            "duration_ms": duration_ms,
            "vram_used_gb": vram_after - vram_before if vram_before and vram_after else None,
            "device": device
        }
        
        logger.info(
            "realesrgan_completed",
            duration_ms=duration_ms,
            output_dimensions=output_image.size,
            vram_used_gb=metadata.get("vram_used_gb")
        )
        
        circuit.record_success()
        update_gpu_metrics("realesrgan")
        
        return output_bytes, metadata
        
    except (MemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            circuit.record_failure(e)
            
            # Try to free GPU memory
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            
            raise GPUMemoryError(job_id=job_id)
        raise
        
    except Exception as e:
        circuit.record_failure(e)
        logger.error("realesrgan_failed", error=str(e))
        raise PipelineStageError(
            f"Upscaling failed: {str(e)}",
            stage="realesrgan",
            job_id=job_id
        )


# =============================================================================
# Stage 3: Smart Placement (Nano Banana API)
# =============================================================================

async def process_nano_banana_stage(
    image_bytes: bytes,
    job_id: str,
    placement_options: Optional[Dict[str, Any]] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Apply smart placement using Nano Banana API ($0.08/image).
    
    Args:
        image_bytes: Input image as bytes
        job_id: Job ID for logging
        placement_options: Optional placement configuration
        
    Returns:
        Tuple of (processed_image_bytes, metadata)
    """
    set_job_context(job_id, "nano_banana")
    circuit = get_circuit_breaker("nano_banana")
    
    if not circuit.can_execute():
        raise ExternalAPIError(
            "Nano Banana API is temporarily unavailable",
            service="nano_banana",
            job_id=job_id
        )
    
    start_time = datetime.utcnow()
    
    # API configuration
    NANO_BANANA_URL = getattr(settings, "NANO_BANANA_API_URL", "https://api.nanobanana.ai/v1/placement")
    NANO_BANANA_API_KEY = getattr(settings, "NANO_BANANA_API_KEY", None)
    COST_PER_IMAGE = 0.08
    
    try:
        logger.info("nano_banana_starting", input_size=len(image_bytes))
        
        # Prepare request
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        payload = {
            "image": image_b64,
            "options": placement_options or {
                "background": "auto",
                "shadow": True,
                "reflection": False
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        if NANO_BANANA_API_KEY:
            headers["Authorization"] = f"Bearer {NANO_BANANA_API_KEY}"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                NANO_BANANA_URL,
                json=payload,
                headers=headers
            )
            
            # Record API call
            record_nano_banana_call(
                status="success" if response.status_code == 200 else "error",
                http_status=response.status_code
            )
            
            if response.status_code == 503:
                # Service unavailable - should retry
                circuit.record_failure()
                raise ExternalAPIError(
                    "Nano Banana API service unavailable",
                    service="nano_banana",
                    http_status=503,
                    job_id=job_id
                )
            
            if response.status_code != 200:
                circuit.record_failure()
                raise ExternalAPIError(
                    f"Nano Banana API error: {response.text}",
                    service="nano_banana",
                    http_status=response.status_code,
                    job_id=job_id
                )
            
            result = response.json()
        
        # Decode result image
        output_b64 = result.get("image", result.get("result", image_b64))
        output_bytes = base64.b64decode(output_b64)
        
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        metadata = {
            "stage": "nano_banana",
            "input_size": len(image_bytes),
            "output_size": len(output_bytes),
            "duration_ms": duration_ms,
            "cost_usd": COST_PER_IMAGE,
            "api_response_time_ms": result.get("processing_time_ms")
        }
        
        logger.info(
            "nano_banana_completed",
            duration_ms=duration_ms,
            cost_usd=COST_PER_IMAGE
        )
        
        circuit.record_success()
        
        return output_bytes, metadata
        
    except httpx.TimeoutException:
        circuit.record_failure()
        raise ExternalAPIError(
            "Nano Banana API timeout",
            service="nano_banana",
            job_id=job_id
        )
        
    except ExternalAPIError:
        raise
        
    except Exception as e:
        circuit.record_failure(e)
        logger.error("nano_banana_failed", error=str(e))
        raise ExternalAPIError(
            f"Nano Banana API call failed: {str(e)}",
            service="nano_banana",
            job_id=job_id
        )


# =============================================================================
# Stage 4: Hero Generation (Mocked)
# =============================================================================

def process_generation_stage(
    image_bytes: bytes,
    job_id: str,
    num_versions: int = 2
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Mocked Hero Generative Model to generate versions of the image.
    """
    set_job_context(job_id, "generation")
    start_time = datetime.utcnow()
    
    logger.info("generation_starting", num_versions=num_versions)
    
    # Simulate processing time
    import time
    time.sleep(2.0)
    
    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    metadata = {
        "stage": "generation",
        "mocked": True,
        "num_versions": num_versions,
        "duration_ms": duration_ms
    }
    
    logger.info("generation_completed", duration_ms=duration_ms)
    
    return image_bytes, metadata


# =============================================================================
# Simulated Stages for Development/Testing
# =============================================================================

async def simulate_guardrail_stage(
    prompt: str,
    image_bytes: Optional[bytes],
    job_id: str
) -> Tuple[bool, Dict[str, Any]]:
    """Simulated guardrail for development."""
    set_job_context(job_id, "guardrail")
    start_time = datetime.utcnow()
    
    await asyncio.sleep(0.5)
    
    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    metadata = {
        "stage": "guardrail",
        "simulated": True,
        "status": "PASS",
        "duration_ms": duration_ms
    }
    
    return True, metadata


def simulate_generation_stage(
    image_bytes: bytes,
    job_id: str,
    num_versions: int = 2
) -> Tuple[bytes, Dict[str, Any]]:
    """Simulated generation for development."""
    return process_generation_stage(image_bytes, job_id, num_versions)


def simulate_rembg_stage(
    image_bytes: bytes,
    job_id: str
) -> Tuple[bytes, Dict[str, Any]]:
    """Simulated rembg for development without GPU."""
    import time
    
    set_job_context(job_id, "rembg")
    start_time = datetime.utcnow()
    
    logger.info("rembg_simulated_starting", input_size=len(image_bytes))
    
    # Simulate processing time
    time.sleep(1.5)
    
    # Just return the same image (simulation)
    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    metadata = {
        "stage": "rembg",
        "simulated": True,
        "input_size": len(image_bytes),
        "output_size": len(image_bytes),
        "duration_ms": duration_ms
    }
    
    logger.info("rembg_simulated_completed", duration_ms=duration_ms)
    
    return image_bytes, metadata


def simulate_realesrgan_stage(
    image_bytes: bytes,
    job_id: str,
    scale: int = 4
) -> Tuple[bytes, Dict[str, Any]]:
    """Simulated RealESRGAN for development without GPU."""
    import time
    
    set_job_context(job_id, "realesrgan")
    start_time = datetime.utcnow()
    
    logger.info("realesrgan_simulated_starting", input_size=len(image_bytes))
    
    # Simulate processing time
    time.sleep(3.0)
    
    # Load and resize with PIL (simulation)
    input_image = Image.open(io.BytesIO(image_bytes))
    new_size = (input_image.width * scale, input_image.height * scale)
    output_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
    
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format="PNG")
    output_bytes = output_buffer.getvalue()
    
    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    metadata = {
        "stage": "realesrgan",
        "simulated": True,
        "input_size": len(image_bytes),
        "output_size": len(output_bytes),
        "original_dimensions": input_image.size,
        "output_dimensions": new_size,
        "scale": scale,
        "duration_ms": duration_ms
    }
    
    logger.info("realesrgan_simulated_completed", duration_ms=duration_ms)
    
    return output_bytes, metadata


async def simulate_nano_banana_stage(
    image_bytes: bytes,
    job_id: str,
    placement_options: Optional[Dict[str, Any]] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """Simulated Nano Banana for development."""
    set_job_context(job_id, "nano_banana")
    start_time = datetime.utcnow()
    
    logger.info("nano_banana_simulated_starting", input_size=len(image_bytes))
    
    # Simulate API call delay
    await asyncio.sleep(0.8)
    
    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    metadata = {
        "stage": "nano_banana",
        "simulated": True,
        "input_size": len(image_bytes),
        "output_size": len(image_bytes),
        "duration_ms": duration_ms,
        "cost_usd": 0.08
    }
    
    logger.info("nano_banana_simulated_completed", duration_ms=duration_ms, cost_usd=0.08)
    record_nano_banana_call(status="success", http_status=200)
    
    return image_bytes, metadata

