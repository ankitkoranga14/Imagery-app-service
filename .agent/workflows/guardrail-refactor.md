---
description: Guardrail Microservice Refactoring - Implementation Plan
---

# Guardrail Microservice Refactoring

## Overview
Refactor the complex imagery pipeline into a dedicated Guardrail Microservice. Remove all generative pipeline stages and create a lean, production-ready validation service.

## ✅ Phase 1: Project Cleanup (COMPLETED)

### 1.1 Removed directories
- ✅ `src/engines/u2net/` - Background removal engine removed
- ✅ `src/engines/gemini/` - Gemini integration removed

### 1.2 Cleaned up tasks.py
- ✅ Removed `run_full_pipeline` Celery chain
- ✅ Removed `process_rembg`, `process_realesrgan`, `process_nano_banana`, `process_generation` tasks
- ✅ Kept `process_guardrail` for backward compatibility
- ✅ Created new `validate_image_task` in `src/pipeline/validation_tasks.py`

### 1.3 Cleaned up stages.py
- ✅ Removed all stage implementations except guardrail
- ✅ Kept `process_guardrail_stage` as primary validation function

## ✅ Phase 2: Core Guardrail Logic (COMPLETED)

### 2.1 services.py (Preserved)
- ✅ Maintained fail-fast logic: L0 (Cache) -> L1 (Text) -> L2 (Physics)
- ✅ L3 (YOLO) and L4 (CLIP) run in parallel if L2 is "Borderline"
- ✅ `GUARDRAIL_PARALLEL_LOADING` feature maintained
- ✅ Lazy model loading preserved

### 2.2 New ValidationResult schema
Created in `src/modules/validation/models.py`:
- `ValidationLog`: Audit trail with status, scores, latency
- `ValidationJob`: Async job tracking

## ✅ Phase 3: API & Database Update (COMPLETED)

### 3.1 New API endpoint
Created `src/api/v1/validate.py`:
- ✅ `POST /api/v1/validate` - Primary validation endpoint
- ✅ Supports synchronous mode (wait for result)
- ✅ Supports asynchronous mode (return job ID)
- ✅ `GET /api/v1/validate/jobs/{job_id}` - Poll async results
- ✅ `GET /api/v1/validate/logs` - Validation history
- ✅ `GET /api/v1/validate/stats` - Statistics

### 3.2 Database Schema
Created `src/modules/validation/models.py`:
- ✅ `ValidationLog` table: image_hash, prompt, status, scores, latency
- ✅ `ValidationJob` table: async job tracking
- ✅ Migration: `alembic/versions/002_validation_tables.py`

## ✅ Phase 4: Infrastructure & Docker (COMPLETED)

### 4.1 Updated docker-compose.yml
- ✅ `api` service: FastAPI server
- ✅ `worker` service: Celery worker for validation
- ✅ `worker-gpu` service: GPU-enabled worker (optional profile)
- ✅ `redis` service: Celery broker + L0 cache

### 4.2 Created Dockerfile.worker
- ✅ Includes: OpenCV, YOLOv11, MobileCLIP2
- ✅ Removed: rembg, ESRGAN dependencies
- ✅ Pre-downloads models during build

### 4.3 Updated requirements.txt
- ✅ Removed generative dependencies (rembg, basicsr, realesrgan)
- ✅ Kept validation dependencies only

## ✅ Phase 5: Frontend Consistency (COMPLETED)

### 5.1 Updated React UI
- ✅ Created `ValidationDashboard.tsx` component
- ✅ Removed references to "Hero Generation", "Background Removal"
- ✅ Shows validation results with clear PASS/BLOCK indicators
- ✅ Displays failure reasons and scores
- ✅ Includes validation history and statistics

### 5.2 Updated API client
- ✅ `food-guard-ui/src/lib/api.ts` - New `validationApi` client
- ✅ `food-guard-ui/src/lib/types.ts` - New validation types

### 5.3 Updated App.tsx
- ✅ Changed branding to "Guardrail Service"
- ✅ Uses ValidationDashboard as main component

## Files Modified/Created

### New Files
- `src/modules/validation/models.py` - ValidationLog, ValidationJob models
- `src/modules/validation/__init__.py` - Module exports
- `src/pipeline/validation_tasks.py` - validate_image_task Celery task
- `src/api/v1/validate.py` - New /validate endpoint
- `Dockerfile.worker` - Optimized worker Dockerfile
- `alembic/versions/002_validation_tables.py` - Database migration
- `food-guard-ui/src/components/ValidationDashboard.tsx` - New UI component

### Modified Files
- `src/pipeline/tasks.py` - Removed generative tasks
- `src/pipeline/stages.py` - Removed generative stages
- `src/api/v1/__init__.py` - Added validate router
- `docker-compose.yml` - Simplified services
- `requirements.txt` - Removed generative dependencies
- `food-guard-ui/src/App.tsx` - Updated to use ValidationDashboard
- `food-guard-ui/src/lib/api.ts` - Added validation API
- `food-guard-ui/src/lib/types.ts` - Added validation types

### Deleted Files/Directories
- `src/engines/u2net/` - Entire directory
- `src/engines/gemini/` - Entire directory

## Running the Service

### Development
```bash
# Start services
docker compose up -d

# Run migrations
docker compose exec api alembic upgrade head

# View logs
docker compose logs -f api worker
```

### With GPU Support
```bash
docker compose --profile gpu up -d
```

### API Usage
```bash
# Synchronous validation
curl -X POST http://localhost:8000/api/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A delicious pasta dish", "image_bytes": "<base64>"}'

# Async validation
curl -X POST http://localhost:8000/api/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A delicious pasta dish", "image_bytes": "<base64>", "async_mode": true}'

# Poll async result
curl http://localhost:8000/api/v1/validate/jobs/{job_id}

# Get validation stats
curl http://localhost:8000/api/v1/validate/stats
```

## Key Technical Requirements Met

1. ✅ **GUARDRAIL_PARALLEL_LOADING** - Maintained via env variable
2. ✅ **Lazy ML model loading** - Prevents OOM during startup
3. ✅ **Modular code design** - Easy to add L5, L6 levels in future
4. ✅ **Fail-fast logic** - L0→L1→L2→L3/L4 (parallel if borderline)
5. ✅ **Audit logging** - All validations stored in ValidationLogs
6. ✅ **Sync/Async modes** - Both supported via single endpoint
