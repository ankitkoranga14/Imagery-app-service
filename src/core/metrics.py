"""
Prometheus Metrics for Observability

Tracks pipeline performance, API calls, and GPU usage.
Exposes /metrics endpoint for Prometheus scraping.
"""

import time
import functools
from typing import Callable, Optional
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY
)

# =============================================================================
# Metrics Definitions
# =============================================================================

# Pipeline Latency - Per Stage
pipeline_latency_seconds = Histogram(
    "pipeline_latency_seconds",
    "Time spent in each pipeline stage",
    labelnames=["stage", "status"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

# Total Pipeline Duration
pipeline_total_duration = Histogram(
    "pipeline_total_duration_seconds",
    "Total time for complete pipeline execution",
    labelnames=["status"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
)

# Nano Banana API Calls
nano_banana_api_calls_total = Counter(
    "nano_banana_api_calls_total",
    "Total number of Nano Banana API calls",
    labelnames=["status", "http_status"]
)

# GPU VRAM Usage
gpu_vram_usage_gauge = Gauge(
    "gpu_vram_usage_bytes",
    "Current GPU VRAM usage in bytes",
    labelnames=["device", "stage"]
)

gpu_vram_usage_gb_gauge = Gauge(
    "gpu_vram_usage_gb",
    "Current GPU VRAM usage in gigabytes",
    labelnames=["device", "stage"]
)

# Jobs Counter
jobs_total = Counter(
    "imagery_jobs_total",
    "Total number of imagery jobs processed",
    labelnames=["status", "failure_stage"]
)

# Active Jobs
active_jobs_gauge = Gauge(
    "imagery_active_jobs",
    "Number of currently processing jobs"
)

# Guardrail Metrics
guardrail_validations_total = Counter(
    "guardrail_validations_total",
    "Total number of guardrail validations",
    labelnames=["status", "block_reason"]
)

guardrail_latency_seconds = Histogram(
    "guardrail_latency_seconds",
    "Time spent in guardrail validation",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# API Request Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    labelnames=["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Cache Metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    labelnames=["cache_type"]
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    labelnames=["cache_type"]
)

# Application Info
app_info = Info(
    "imagery_app",
    "Application information"
)


# =============================================================================
# Helper Functions
# =============================================================================

def set_app_info(version: str, environment: str):
    """Set application info metric."""
    app_info.info({
        "version": version,
        "environment": environment
    })


def get_gpu_vram_usage() -> Optional[float]:
    """Get current GPU VRAM usage in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            return allocated
    except ImportError:
        pass
    return None


def update_gpu_metrics(stage: str = "unknown"):
    """Update GPU VRAM metrics."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                allocated_gb = allocated / (1024 ** 3)
                device_name = f"cuda:{i}"
                
                gpu_vram_usage_gauge.labels(device=device_name, stage=stage).set(allocated)
                gpu_vram_usage_gb_gauge.labels(device=device_name, stage=stage).set(allocated_gb)
    except Exception:
        pass


@contextmanager
def track_stage_latency(stage: str):
    """
    Context manager to track stage latency.
    
    Usage:
        with track_stage_latency("rembg"):
            # do work
    """
    start = time.time()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start
        pipeline_latency_seconds.labels(stage=stage, status=status).observe(duration)
        update_gpu_metrics(stage)


def track_latency(stage: str):
    """
    Decorator to track function latency.
    
    Usage:
        @track_latency("rembg")
        async def remove_background(image):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with track_stage_latency(stage):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with track_stage_latency(stage):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def record_nano_banana_call(status: str, http_status: int = 200):
    """Record a Nano Banana API call."""
    nano_banana_api_calls_total.labels(
        status=status,
        http_status=str(http_status)
    ).inc()


def record_job_completion(status: str, failure_stage: str = "none"):
    """Record job completion."""
    jobs_total.labels(status=status, failure_stage=failure_stage).inc()
    if status in ["processing"]:
        active_jobs_gauge.inc()
    elif status in ["completed", "failed"]:
        active_jobs_gauge.dec()


def record_guardrail_validation(status: str, block_reason: str = "none"):
    """Record guardrail validation."""
    guardrail_validations_total.labels(
        status=status,
        block_reason=block_reason
    ).inc()


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest(REGISTRY)


def get_metrics_content_type() -> str:
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST


# Initialize app info on module load
set_app_info(version="1.0.0", environment="development")

