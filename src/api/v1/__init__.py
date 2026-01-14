"""
API v1 Router Module - Guardrail Microservice

All v1 endpoints are prefixed with /api/v1/

Primary endpoint: POST /api/v1/validate
- Synchronous and asynchronous validation modes
- Audit logging via ValidationLogs

Legacy endpoints (for backward compatibility):
- /api/v1/guardrail/* - Direct guardrail access
- /api/v1/process/* - Legacy pipeline (deprecated)
"""

from fastapi import APIRouter

from src.api.v1.validate import router as validate_router
from src.api.v1.guardrail import router as guardrail_router
from src.api.v1.process import router as process_router
from src.api.v1.metrics import router as metrics_router
from src.api.v1.status import router as status_router

# Main v1 router
api_v1_router = APIRouter(prefix="/api/v1")

# Primary endpoint - Guardrail Microservice
api_v1_router.include_router(validate_router, prefix="/validate", tags=["validation"])

# Legacy/supporting endpoints
api_v1_router.include_router(guardrail_router, prefix="/guardrail", tags=["guardrail"])
api_v1_router.include_router(process_router, prefix="/process", tags=["pipeline (deprecated)"])
api_v1_router.include_router(status_router, prefix="/status", tags=["status"])
api_v1_router.include_router(metrics_router, tags=["metrics"])

