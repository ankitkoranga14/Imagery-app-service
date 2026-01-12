"""
Celery Application Configuration

Configures Celery with:
- Task routing for GPU queue
- Exponential backoff for retries
- Result backend for task tracking
"""

from celery import Celery
from kombu import Queue

from src.core.config import settings

# Create Celery app
celery_app = Celery(
    "imagery_pipeline",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "src.pipeline.tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task tracking
    task_track_started=True,
    task_time_limit=600,  # 10 minute hard limit
    task_soft_time_limit=540,  # 9 minute soft limit
    
    # Result expiration
    result_expires=86400,  # 24 hours
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Process one task at a time for GPU
    worker_concurrency=1,  # Single worker for GPU tasks
    
    # Queue definitions
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("gpu_queue", routing_key="gpu.#"),
        Queue("api_queue", routing_key="api.#"),
    ),
    
    # Task routing
    task_routes={
        "src.pipeline.tasks.process_rembg": {"queue": "gpu_queue"},
        "src.pipeline.tasks.process_realesrgan": {"queue": "gpu_queue"},
        "src.pipeline.tasks.process_nano_banana": {"queue": "api_queue"},
        "src.pipeline.tasks.run_full_pipeline": {"queue": "default"},
    },
    
    # Retry settings with exponential backoff
    task_default_retry_delay=5,  # 5 seconds initial delay
    task_max_retries=5,
    
    # Late acknowledgment for reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Beat schedule for periodic tasks (optional)
celery_app.conf.beat_schedule = {
    # Example: cleanup old jobs every hour
    # "cleanup-old-jobs": {
    #     "task": "src.pipeline.tasks.cleanup_old_jobs",
    #     "schedule": 3600.0,
    # },
}

