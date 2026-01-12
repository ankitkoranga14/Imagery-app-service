"""
Application Configuration

Centralized settings management using Pydantic Settings.
Supports environment variables and .env files.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # ==========================================================================
    # App Settings
    # ==========================================================================
    APP_NAME: str = "Imagery Guardrail & Hybrid Pipeline"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"  # development, staging, PROD
    DEBUG: bool = True
    
    # ==========================================================================
    # Infrastructure
    # ==========================================================================
    REDIS_URL: str = "redis://localhost:6379/1"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/imagery.db"
    
    # PostgreSQL for production (optional)
    # DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/imagery"
    
    # ==========================================================================
    # Storage Settings (The Bridge Pattern)
    # ==========================================================================
    # Local storage path (development)
    LOCAL_STORAGE_PATH: str = "./data/storage"
    
    # Azure Blob Storage (production) - uncomment and set for cloud deployment
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None
    AZURE_CONTAINER_NAME: str = "imagery"
    
    # ==========================================================================
    # ML Settings
    # ==========================================================================
    ML_MODEL_CACHE_DIR: Path = Path("./ml_cache")
    CACHE_TTL_SECONDS: int = 3600
    MAX_IMAGE_SIZE_BYTES: int = 10485760  # 10MB
    MAX_PROMPT_LENGTH: int = 2000
    
    # ==========================================================================
    # Pipeline Settings
    # ==========================================================================
    # RealESRGAN
    UPSCALE_FACTOR: int = 4
    TILE_SIZE: int = 512  # For VRAM management
    
    # Nano Banana API
    NANO_BANANA_API_URL: str = "https://api.nanobanana.ai/v1/placement"
    NANO_BANANA_API_KEY: Optional[str] = None
    NANO_BANANA_COST_PER_IMAGE: float = 0.08
    
    # ==========================================================================
    # Celery Settings
    # ==========================================================================
    CELERY_BROKER_URL: Optional[str] = None  # Falls back to REDIS_URL
    CELERY_RESULT_BACKEND: Optional[str] = None  # Falls back to REDIS_URL
    
    # ==========================================================================
    # Logging Settings
    # ==========================================================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT_JSON: bool = True  # JSON for production, console for development
    
    # ==========================================================================
    # CORS Settings
    # ==========================================================================
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:8000"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure critical directories exist
settings.ML_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
Path(settings.LOCAL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(parents=True, exist_ok=True)
