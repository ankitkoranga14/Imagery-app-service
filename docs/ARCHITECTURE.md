# Imagery Guardrail & Hybrid Pipeline - Architecture

## Overview

A production-ready, scalable, and observable full-stack application for image processing with AI guardrails.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (React + Tailwind)                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   Stepper   │ │   Metrics   │ │    Logs     │ │   Comparison Slider     │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTP/WebSocket
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (FastAPI)                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    /api/v1/ Router (Versioned)                          │ │
│  ├─────────────┬─────────────┬─────────────┬──────────────┬───────────────┤ │
│  │  /guardrail │  /process   │  /status    │   /metrics   │    /logs      │ │
│  │   (sync)    │  (dispatch) │  (polling)  │ (prometheus) │   (stream)    │ │
│  └─────────────┴─────────────┴─────────────┴──────────────┴───────────────┘ │
│                                     │                                        │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐  │
│  │                     Global Exception Handler                          │  │
│  │              Structured Logging (structlog → JSON)                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────┐
│   Redis Queue   │     │    PostgreSQL   │     │   Storage Abstraction   │
│   (Celery)      │     │    (SQLModel)   │     │  ┌─────────────────────┐│
│                 │     │                 │     │  │    IStorage         ││
│  - default      │     │  - ImageJob     │     │  ├─────────────────────┤│
│  - gpu_queue    │     │  - GuardrailLog │     │  │  LocalStorage  ────►││
│  - api_queue    │     │  - Users        │     │  │  AzureBlobStorage   ││
└────────┬────────┘     └─────────────────┘     │  └─────────────────────┘│
         │                                       └─────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CELERY WORKER PIPELINE                                │
│                                                                              │
│   ┌──────────┐      ┌──────────────┐      ┌──────────────┐                  │
│   │  Stage 1 │ ───► │   Stage 2    │ ───► │   Stage 3    │                  │
│   │  REMBG   │      │  REALESRGAN  │      │ NANO BANANA  │                  │
│   │  (GPU)   │      │  (GPU/Tiled) │      │  (HTTP API)  │                  │
│   └──────────┘      └──────────────┘      └──────────────┘                  │
│        │                   │                     │                           │
│        ▼                   ▼                     ▼                           │
│   Circuit Breaker    Circuit Breaker      Exponential Backoff               │
│   OOM Handling       VRAM Management      Retry on 503                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## API Versioning Strategy

All endpoints are versioned under `/api/v1/`:

```
app/api/v1/endpoints/process.py  →  POST /api/v1/process
app/api/v1/endpoints/status.py   →  GET  /api/v1/status/{job_id}
app/api/v1/endpoints/guardrail.py → POST /api/v1/guardrail/validate
app/api/v1/endpoints/metrics.py  →  GET  /api/v1/metrics
```

## Storage Abstraction (The Bridge Pattern)

```python
def get_storage():
    if settings.ENV == "PROD":
        return AzureBlobStorage(connection_string=settings.AZURE_STR)
    return LocalStorage(path="./data")
```

This means your code is **100% ready for Azure today**. When you're ready to move from local storage to the cloud, you simply change the `ENV` variable to `PROD` and provide the connection string. **No code changes required.**

## Pipeline Stages

### Stage 1: Background Removal (Rembg)
- GPU-accelerated background removal
- Circuit breaker for OOM protection
- Transparent PNG output

### Stage 2: 4K Tiled Upscaling (RealESRGAN)
- Tiled processing for VRAM management (512px tiles)
- 4x upscaling (512px → 2048px)
- Automatic FP16 on GPU

### Stage 3: Smart Placement (Nano Banana API)
- External API integration ($0.08/image)
- Exponential backoff on 503 errors
- Circuit breaker protection

## Structured Logging

Every log includes `job_id`, `version`, and `stage`:

```json
{
  "timestamp": "2024-05-20T10:00:00Z",
  "level": "info",
  "event": "stage_completed",
  "stage": "realesrgan",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "duration_ms": 4200,
  "vram_used_gb": 4.2
}
```

## Error Handling Flow

The system implements a **State Machine**. If the "Rembg" stage fails:
1. The task catches the exception
2. Updates the SQL database with the error message
3. Stops the $0.08 Nano Banana call from ever happening
4. Saves money on failed runs

```python
try:
    result = process_rembg(image)
except GPUMemoryError:
    update_job_status(job_id, "FAILED", stage="rembg")
    circuit_breaker.record_failure()
    # Nano Banana is never called - no cost incurred
```

## Prometheus Metrics

Available at `/api/v1/metrics`:

- `pipeline_latency_seconds` (per stage)
- `nano_banana_api_calls_total`
- `gpu_vram_usage_gauge`
- `guardrail_validations_total`
- `imagery_jobs_total`

## Docker Compose Services

| Service | Description | Port |
|---------|-------------|------|
| api | FastAPI application | 8000 |
| worker | Celery CPU worker | - |
| worker-gpu | Celery GPU worker | - |
| redis | Message broker | 6379 |
| frontend | React dashboard | 3000 |
| flower | Celery monitoring | 5555 |

## Quick Start

```bash
# Development
docker-compose up

# With GPU worker
docker-compose --profile gpu up

# With monitoring (Flower)
docker-compose --profile monitoring up
```

## Environment Variables

See `.env.example` for all configuration options.
