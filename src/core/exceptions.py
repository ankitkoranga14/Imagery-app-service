"""
Global Exception Handling

Provides structured error responses and circuit breaker pattern
for graceful failure handling.
"""

import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import get_logger, job_id_var

logger = get_logger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class ImageryBaseException(Exception):
    """Base exception for Imagery App."""
    
    def __init__(
        self,
        message: str,
        code: int = 500,
        job_id: Optional[str] = None,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.job_id = job_id or job_id_var.get()
        self.stage = stage
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ImageryBaseException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, code=400, **kwargs)


class GuardrailBlockedError(ImageryBaseException):
    """Raised when content is blocked by guardrail."""
    
    def __init__(self, message: str, reasons: list, **kwargs):
        super().__init__(message, code=422, **kwargs)
        self.details["reasons"] = reasons


class PipelineStageError(ImageryBaseException):
    """Raised when a pipeline stage fails."""
    
    def __init__(self, message: str, stage: str, **kwargs):
        super().__init__(message, code=500, stage=stage, **kwargs)


class ExternalAPIError(ImageryBaseException):
    """Raised when an external API call fails (e.g., Nano Banana)."""
    
    def __init__(self, message: str, service: str, http_status: Optional[int] = None, **kwargs):
        super().__init__(message, code=502, **kwargs)
        self.details["service"] = service
        self.details["http_status"] = http_status


class GPUMemoryError(ImageryBaseException):
    """Raised when GPU runs out of memory (OOM)."""
    
    def __init__(self, message: str = "GPU out of memory", **kwargs):
        super().__init__(message, code=507, **kwargs)


class StorageError(ImageryBaseException):
    """Raised when storage operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, code=500, **kwargs)


class CircuitBreakerOpenError(ImageryBaseException):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service: str, **kwargs):
        super().__init__(
            f"Service '{service}' is temporarily unavailable (circuit breaker open)",
            code=503,
            **kwargs
        )
        self.details["service"] = service


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreaker:
    """
    Circuit Breaker pattern for graceful failure handling.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail fast
    - HALF_OPEN: Testing if service is recovered
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "CLOSED"
        self._half_open_calls = 0
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        if self._state == "OPEN":
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._state = "HALF_OPEN"
                    self._half_open_calls = 0
        return self._state
    
    def can_execute(self) -> bool:
        """Check if request can proceed."""
        state = self.state
        
        if state == "CLOSED":
            return True
        elif state == "OPEN":
            return False
        elif state == "HALF_OPEN":
            return self._half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record a successful call."""
        if self._state == "HALF_OPEN":
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                # Recovery successful
                self._state = "CLOSED"
                self._failure_count = 0
                logger.info(
                    "circuit_breaker_closed",
                    circuit=self.name,
                    message="Service recovered"
                )
        elif self._state == "CLOSED":
            self._failure_count = 0
    
    def record_failure(self, error: Optional[Exception] = None):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        
        if self._state == "HALF_OPEN":
            # Recovery failed, go back to OPEN
            self._state = "OPEN"
            logger.warning(
                "circuit_breaker_reopened",
                circuit=self.name,
                error=str(error) if error else None
            )
        elif self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                "circuit_breaker_opened",
                circuit=self.name,
                failure_count=self._failure_count,
                error=str(error) if error else None
            )
    
    def reset(self):
        """Reset the circuit breaker."""
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


# Global circuit breakers for external services
circuit_breakers: Dict[str, CircuitBreaker] = {
    "nano_banana": CircuitBreaker("nano_banana", failure_threshold=3, recovery_timeout=120),
    "realesrgan": CircuitBreaker("realesrgan", failure_threshold=2, recovery_timeout=60),
    "rembg": CircuitBreaker("rembg", failure_threshold=2, recovery_timeout=60),
}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a service."""
    if name not in circuit_breakers:
        circuit_breakers[name] = CircuitBreaker(name)
    return circuit_breakers[name]


# =============================================================================
# Exception Handler Middleware
# =============================================================================

class GlobalExceptionMiddleware(BaseHTTPMiddleware):
    """
    Global exception handler middleware.
    
    Catches all exceptions and returns structured JSON responses.
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            return self._handle_exception(request, exc)
    
    def _handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Convert exception to structured JSON response."""
        
        # Get job_id from context if available
        job_id = job_id_var.get()
        
        # Determine error details based on exception type
        if isinstance(exc, ImageryBaseException):
            error_response = {
                "error": exc.message,
                "job_id": exc.job_id or job_id,
                "code": exc.code,
                "stage": exc.stage,
                "details": exc.details,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            status_code = exc.code
            
        elif isinstance(exc, HTTPException):
            error_response = {
                "error": exc.detail,
                "job_id": job_id,
                "code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            status_code = exc.status_code
            
        else:
            # Unexpected exception
            error_response = {
                "error": "Internal server error",
                "job_id": job_id,
                "code": 500,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            status_code = 500
            
            # Log full traceback for unexpected errors
            logger.error(
                "unhandled_exception",
                error=str(exc),
                error_type=type(exc).__name__,
                traceback=traceback.format_exc()
            )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )


def register_exception_handlers(app: FastAPI):
    """Register custom exception handlers with FastAPI app."""
    
    @app.exception_handler(ImageryBaseException)
    async def imagery_exception_handler(request: Request, exc: ImageryBaseException):
        job_id = exc.job_id or job_id_var.get()
        
        logger.error(
            "imagery_exception",
            error=exc.message,
            code=exc.code,
            stage=exc.stage,
            details=exc.details
        )
        
        return JSONResponse(
            status_code=exc.code,
            content={
                "error": exc.message,
                "job_id": job_id,
                "code": exc.code,
                "stage": exc.stage,
                "details": exc.details,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        job_id = job_id_var.get()
        
        logger.error(
            "unhandled_exception",
            error=str(exc),
            error_type=type(exc).__name__,
            path=str(request.url.path),
            traceback=traceback.format_exc()
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "job_id": job_id,
                "code": 500,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

