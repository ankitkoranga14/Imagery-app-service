from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Food Guard Imagery Service"
    ENVIRONMENT: str = "production"
    
    # Infrastructure
    REDIS_URL: str = "redis://localhost:6379/1"
    DATABASE_URL: str = "sqlite+aiosqlite:///./food_guard.db"
    
    # Azure Settings (Placeholders for now)
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None
    AZURE_CONTAINER_NAME: str = "images"
    
    # ML Settings
    CACHE_TTL_SECONDS: int = 3600
    MAX_IMAGE_SIZE_BYTES: int = 10485760
    MAX_PROMPT_LENGTH: int = 2000
    ML_MODEL_CACHE_DIR: Path = Path("./ml_cache")

    class Config:
        env_file = ".env"

settings = Settings()
