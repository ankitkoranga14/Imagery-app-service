"""
Structured Logging Configuration with structlog

Outputs JSON logs that are searchable in Azure Monitor, ELK, or CloudWatch.
Every log includes: job_id, version, stage, timestamp, and other context.
"""

import sys
import logging
import structlog
from typing import Optional, Any, Dict
from datetime import datetime
from contextvars import ContextVar
from functools import wraps

# Context variables for request-scoped logging
job_id_var: ContextVar[Optional[str]] = ContextVar("job_id", default=None)
stage_var: ContextVar[Optional[str]] = ContextVar("stage", default=None)

# Application version
APP_VERSION = "1.0.0"


def add_app_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add application context to every log entry."""
    event_dict["version"] = APP_VERSION
    
    # Add job_id if present
    job_id = job_id_var.get()
    if job_id:
        event_dict["job_id"] = job_id
    
    # Add stage if present
    stage = stage_var.get()
    if stage:
        event_dict["stage"] = stage
    
    return event_dict


def add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add ISO format timestamp."""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None
):
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, output JSON; if False, output colored console logs
        log_file: Optional file path for log output
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Choose processor based on format
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            add_timestamp,
            add_app_context,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.UnicodeDecoder(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for setting logging context.
    
    Usage:
        with LogContext(job_id="abc123", stage="rembg"):
            logger.info("Processing started")
    """
    
    def __init__(self, job_id: Optional[str] = None, stage: Optional[str] = None):
        self.job_id = job_id
        self.stage = stage
        self._job_id_token = None
        self._stage_token = None
    
    def __enter__(self):
        if self.job_id:
            self._job_id_token = job_id_var.set(self.job_id)
        if self.stage:
            self._stage_token = stage_var.set(self.stage)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._job_id_token:
            job_id_var.reset(self._job_id_token)
        if self._stage_token:
            stage_var.reset(self._stage_token)
        return False
    
    def set_stage(self, stage: str):
        """Update the current stage."""
        self._stage_token = stage_var.set(stage)


def set_job_context(job_id: str, stage: Optional[str] = None):
    """Set the current job context for logging."""
    job_id_var.set(job_id)
    if stage:
        stage_var.set(stage)


def clear_job_context():
    """Clear the current job context."""
    job_id_var.set(None)
    stage_var.set(None)


def with_logging(stage: str):
    """
    Decorator to wrap a function with logging context.
    
    Usage:
        @with_logging("rembg")
        async def remove_background(image: bytes) -> bytes:
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            stage_var.set(stage)
            
            logger.info(f"stage_started", stage=stage)
            start_time = datetime.utcnow()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                logger.info(
                    "stage_completed",
                    stage=stage,
                    duration_ms=duration_ms
                )
                return result
            except Exception as e:
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                logger.error(
                    "stage_failed",
                    stage=stage,
                    duration_ms=duration_ms,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            stage_var.set(stage)
            
            logger.info(f"stage_started", stage=stage)
            start_time = datetime.utcnow()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                logger.info(
                    "stage_completed",
                    stage=stage,
                    duration_ms=duration_ms
                )
                return result
            except Exception as e:
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                logger.error(
                    "stage_failed",
                    stage=stage,
                    duration_ms=duration_ms,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Example log output structure:
# {
#   "timestamp": "2024-05-20T10:00:00Z",
#   "level": "info",
#   "event": "stage_completed",
#   "stage": "realesrgan",
#   "job_id": "550e8400-e29b-41d4-a716-446655440000",
#   "version": "1.0.0",
#   "duration_ms": 4200,
#   "vram_used_gb": 4.2
# }

