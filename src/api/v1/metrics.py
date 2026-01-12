"""
Metrics Endpoint

GET /api/v1/metrics - Prometheus metrics endpoint
"""

from fastapi import APIRouter, Response

from src.core.metrics import get_metrics, get_metrics_content_type

router = APIRouter()


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes:
    - pipeline_latency_seconds (per stage)
    - nano_banana_api_calls_total
    - gpu_vram_usage_gauge
    - guardrail_validations_total
    - imagery_jobs_total
    - http_requests_total
    """
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type()
    )

