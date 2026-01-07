import torch
from sentence_transformers import SentenceTransformer
import open_clip
from pathlib import Path
from typing import Optional, Tuple, Any
import hashlib
import redis.asyncio as redis
import json
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from src.engines.guardrail.models import GuardrailLog

class MLRepository:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.text_model: Optional[SentenceTransformer] = None
        self.clip_model = None
        self.clip_preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_text_model(self) -> SentenceTransformer:
        if self.text_model is None:
            self.text_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=str(self.cache_dir / "sentence-transformers")
            )
        return self.text_model
    
    def get_clip_model(self) -> Tuple[Any, Any]:
        if self.clip_model is None:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k'
            )
            self.clip_model.eval()
            self.clip_model.to(self.device)
        return self.clip_model, self.clip_preprocess

class CacheRepository:
    def __init__(self, redis_client, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
    
    async def get(self, input_hash: str) -> Optional[Dict]:
        cached = await self.redis.get(f"guardrail:{input_hash}")
        return json.loads(cached) if cached else None
    
    async def set(self, input_hash: str, result: Dict):
        await self.redis.setex(f"guardrail:{input_hash}", self.ttl, json.dumps(result))
    
    async def compute_hash(self, prompt: str, image_bytes: Optional[str] = None) -> str:
        data = f"{prompt}:{image_bytes[:100] if image_bytes else ''}"
        return hashlib.sha256(data.encode()).hexdigest()

class LogRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, log: GuardrailLog) -> GuardrailLog:
        self.session.add(log)
        await self.session.commit()
        await self.session.refresh(log)
        return log
