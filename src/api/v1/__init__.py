"""
API v1 Router Module

All v1 endpoints are prefixed with /api/v1/
"""

from fastapi import APIRouter

from src.api.v1.guardrail import router as guardrail_router
from src.api.v1.process import router as process_router
from src.api.v1.metrics import router as metrics_router
from src.api.v1.status import router as status_router

# Main v1 router
api_v1_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_v1_router.include_router(guardrail_router, prefix="/guardrail", tags=["guardrail"])
api_v1_router.include_router(process_router, prefix="/process", tags=["pipeline"])
api_v1_router.include_router(status_router, prefix="/status", tags=["status"])
api_v1_router.include_router(metrics_router, tags=["metrics"])

