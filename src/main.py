import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.core.database import create_db_and_tables
from src.api.v1.guardrail import router as guardrail_router
from src.api.dependencies import get_ml_repo
import redis.asyncio as redis
from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Imagery App Service...")
    
    # Initialize DB
    await create_db_and_tables()
    
    # Models will be loaded lazily on first request to avoid startup timeout
    logger.info("ML models will be loaded lazily on first request...")
    
    # Initialize cache
    # Initialize cache
    app.state.redis = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    
    logger.info("Service ready.")
    yield
    logger.info("Shutting down...")
    await app.state.redis.aclose()

app = FastAPI(
    title="Imagery App Service",
    description="Central Hub for AI Imagery Processing (Guardrail, U2Net, Gemini, RealESRGAN)",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Include routers
app.include_router(guardrail_router)

# Mount static files
# Ensure the directory exists or handle it gracefully if running locally without build
static_dir = os.path.join(os.path.dirname(__file__), "..", "food-guard-ui", "dist")
if os.path.exists(static_dir):
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

@app.get("/ui")
async def ui_redirect():
    if os.path.exists(os.path.join(static_dir, "index.html")):
        return FileResponse(os.path.join(static_dir, "index.html"))
    return {"message": "UI not built. Please run 'npm run build' in food-guard-ui directory."}

@app.get("/")
def root():
    return {"message": "Imagery App Service is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
