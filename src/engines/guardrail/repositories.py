import torch
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import open_clip
from ultralytics import YOLO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from src.engines.guardrail.models import GuardrailLog, GuardrailFeedback, GuardrailConfigVariant

logger = logging.getLogger(__name__)


class MLRepository:
    """Repository for ML model management with singleton pattern.
    
    Handles lazy loading and caching of:
    - Sentence Transformers (text embeddings)
    - CLIP (image-text similarity)
    - YOLOv8 (object detection)
    """
    
    _instance: Optional['MLRepository'] = None
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.text_model: Optional[SentenceTransformer] = None
        self.clip_model = None
        self.clip_preprocess = None
        self.yolo_model: Optional[YOLO] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models_loaded = False
    
    @classmethod
    def get_instance(cls, cache_dir: Path = None) -> 'MLRepository':
        """Get singleton instance of MLRepository."""
        if cls._instance is None:
            if cache_dir is None:
                cache_dir = Path("./model_cache")
            cls._instance = cls(cache_dir)
        return cls._instance
    
    def get_text_model(self) -> SentenceTransformer:
        """Get sentence transformer model for text embeddings."""
        if self.text_model is None:
            logger.info("[MLRepository] Loading SentenceTransformer model...")
            self.text_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=str(self.cache_dir / "sentence-transformers")
            )
            logger.info("[MLRepository] SentenceTransformer loaded")
        return self.text_model
    
    def get_clip_model(self) -> Tuple[Any, Any]:
        """Get CLIP model and preprocessor."""
        if self.clip_model is None:
            logger.info("[MLRepository] Loading CLIP model...")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k'
            )
            self.clip_model.eval()
            self.clip_model.to(self.device)
            logger.info(f"[MLRepository] CLIP loaded on {self.device}")
        return self.clip_model, self.clip_preprocess
    
    def get_yolo_model(self) -> YOLO:
        """Get YOLOv8 nano model."""
        if self.yolo_model is None:
            logger.info("[MLRepository] Loading YOLOv8n model...")
            self.yolo_model = YOLO("yolov8n.pt")
            self.yolo_model.to(self.device)
            logger.info(f"[MLRepository] YOLOv8n loaded on {self.device}")
        return self.yolo_model
    
    def encode_text(self, text: str) -> Any:
        """Encode text using sentence transformer."""
        model = self.get_text_model()
        return model.encode(text, convert_to_tensor=True)
    
    def preload_all_models(self) -> bool:
        """Preload all models for faster cold start.
        
        Call this on application startup.
        Returns True if all models loaded successfully.
        """
        try:
            self.get_text_model()
            self.get_clip_model()
            self.get_yolo_model()
            self._models_loaded = True
            logger.info("[MLRepository] All models preloaded successfully")
            return True
        except Exception as e:
            logger.error(f"[MLRepository] Failed to preload models: {str(e)}")
            return False
    
    @property
    def models_loaded(self) -> bool:
        """Check if all models are loaded."""
        return self._models_loaded


class CacheRepository:
    """Repository for caching validation results in Redis."""
    
    def __init__(self, redis_client, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
        self.prefix = "guardrail"
    
    async def get(self, input_hash: str) -> Optional[Dict]:
        """Get cached validation result."""
        cached = await self.redis.get(f"{self.prefix}:{input_hash}")
        return json.loads(cached) if cached else None
    
    async def set(self, input_hash: str, result: Dict):
        """Cache validation result."""
        await self.redis.setex(
            f"{self.prefix}:{input_hash}", 
            self.ttl, 
            json.dumps(result, default=str)
        )
    
    async def delete(self, input_hash: str):
        """Delete cached result."""
        await self.redis.delete(f"{self.prefix}:{input_hash}")
    
    async def compute_hash(self, prompt: str, image_bytes: Optional[str] = None) -> str:
        """Compute hash for cache key.
        
        Uses prompt + first 100 chars of image for uniqueness.
        """
        data = f"{prompt}:{image_bytes[:100] if image_bytes else ''}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        try:
            keys = await self.redis.keys(f"{self.prefix}:*")
            return {
                "cached_entries": len(keys),
                "ttl_seconds": self.ttl
            }
        except Exception:
            return {"cached_entries": 0, "ttl_seconds": self.ttl}


class LogRepository:
    """Repository for guardrail validation logs."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, log: GuardrailLog) -> GuardrailLog:
        """Save a validation log entry."""
        self.session.add(log)
        await self.session.commit()
        await self.session.refresh(log)
        return log
    
    async def save_enhanced(
        self,
        prompt: str,
        status: str,
        reasons: Optional[str],
        processing_time_ms: int,
        input_hash: Optional[str] = None,
        levels_executed: Optional[List[str]] = None,
        levels_failed: Optional[List[str]] = None,
        scores: Optional[Dict[str, Any]] = None
    ) -> GuardrailLog:
        """Save an enhanced log entry with intermediate data."""
        log = GuardrailLog(
            prompt=prompt,
            status=status,
            reasons=reasons,
            processing_time_ms=processing_time_ms,
            input_hash=input_hash,
            levels_executed=",".join(levels_executed) if levels_executed else None,
            levels_failed=",".join(levels_failed) if levels_failed else None,
            darkness_score=scores.get("darkness_score") if scores else None,
            glare_percentage=scores.get("glare_percentage") if scores else None,
            food_score=scores.get("food_score") if scores else None,
            nsfw_score=scores.get("nsfw_score") if scores else None,
            dish_count=scores.get("distinct_dish_count") if scores else None,
        )
        self.session.add(log)
        await self.session.commit()
        await self.session.refresh(log)
        return log
    
    async def get_by_hash(self, input_hash: str) -> Optional[GuardrailLog]:
        """Get log entry by input hash."""
        result = await self.session.execute(
            select(GuardrailLog).where(GuardrailLog.input_hash == input_hash)
        )
        return result.scalar_one_or_none()
    
    async def get_recent(self, limit: int = 100) -> List[GuardrailLog]:
        """Get recent log entries."""
        result = await self.session.execute(
            select(GuardrailLog)
            .order_by(GuardrailLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = await self.session.execute(
            select(func.count(GuardrailLog.id))
        )
        pass_count = await self.session.execute(
            select(func.count(GuardrailLog.id)).where(GuardrailLog.status == "PASS")
        )
        block_count = await self.session.execute(
            select(func.count(GuardrailLog.id)).where(GuardrailLog.status == "BLOCK")
        )
        avg_time = await self.session.execute(
            select(func.avg(GuardrailLog.processing_time_ms))
        )
        
        return {
            "total_validations": total.scalar() or 0,
            "pass_count": pass_count.scalar() or 0,
            "block_count": block_count.scalar() or 0,
            "avg_processing_time_ms": avg_time.scalar() or 0
        }


class FeedbackRepository:
    """Repository for guardrail feedback management."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, feedback: GuardrailFeedback) -> GuardrailFeedback:
        """Save feedback entry."""
        self.session.add(feedback)
        await self.session.commit()
        await self.session.refresh(feedback)
        return feedback
    
    async def create(
        self,
        feedback_type: str,
        validation_id: Optional[int] = None,
        input_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        failed_level: Optional[str] = None,
        user_comment: Optional[str] = None,
        original_prompt: Optional[str] = None,
        original_status: Optional[str] = None
    ) -> GuardrailFeedback:
        """Create a new feedback entry."""
        feedback = GuardrailFeedback(
            validation_id=validation_id,
            input_hash=input_hash,
            user_id=user_id,
            feedback_type=feedback_type,
            failed_level=failed_level,
            user_comment=user_comment,
            original_prompt=original_prompt,
            original_status=original_status
        )
        return await self.save(feedback)
    
    async def get_by_validation(self, validation_id: int) -> List[GuardrailFeedback]:
        """Get feedback for a specific validation."""
        result = await self.session.execute(
            select(GuardrailFeedback).where(GuardrailFeedback.validation_id == validation_id)
        )
        return list(result.scalars().all())
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        total = await self.session.execute(
            select(func.count(GuardrailFeedback.id))
        )
        false_positive = await self.session.execute(
            select(func.count(GuardrailFeedback.id))
            .where(GuardrailFeedback.feedback_type == "false_positive")
        )
        false_negative = await self.session.execute(
            select(func.count(GuardrailFeedback.id))
            .where(GuardrailFeedback.feedback_type == "false_negative")
        )
        correct = await self.session.execute(
            select(func.count(GuardrailFeedback.id))
            .where(GuardrailFeedback.feedback_type == "correct")
        )
        
        total_val = total.scalar() or 0
        false_positive_val = false_positive.scalar() or 0
        false_negative_val = false_negative.scalar() or 0
        correct_val = correct.scalar() or 0
        
        # Calculate precision and recall if we have data
        precision = None
        recall = None
        if total_val > 0:
            # True positives = correct blocks
            # False positives = should have passed but blocked
            # False negatives = should have blocked but passed
            tp = correct_val  # Simplified: correct validations
            fp = false_positive_val
            fn = false_negative_val
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            if tp + fn > 0:
                recall = tp / (tp + fn)
        
        return {
            "total_feedback": total_val,
            "false_positives": false_positive_val,
            "false_negatives": false_negative_val,
            "correct": correct_val,
            "precision": precision,
            "recall": recall
        }
    
    async def get_by_level(self, level: str) -> List[GuardrailFeedback]:
        """Get feedback for a specific validation level."""
        result = await self.session.execute(
            select(GuardrailFeedback).where(GuardrailFeedback.failed_level == level)
        )
        return list(result.scalars().all())


class ConfigVariantRepository:
    """Repository for A/B testing configuration variants."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_active_variants(self) -> List[GuardrailConfigVariant]:
        """Get all active configuration variants."""
        result = await self.session.execute(
            select(GuardrailConfigVariant).where(GuardrailConfigVariant.is_active == True)
        )
        return list(result.scalars().all())
    
    async def get_variant_for_user(self, user_id: int) -> Optional[GuardrailConfigVariant]:
        """Get configuration variant for a user based on rollout percentage.
        
        Uses consistent hashing so users always get the same variant.
        """
        variants = await self.get_active_variants()
        if not variants:
            return None
        
        # Use user_id hash for consistent assignment
        user_hash = hash(str(user_id)) % 100
        
        cumulative = 0
        for variant in sorted(variants, key=lambda v: v.name):
            cumulative += variant.rollout_percentage
            if user_hash < cumulative:
                return variant
        
        return None
    
    async def save(self, variant: GuardrailConfigVariant) -> GuardrailConfigVariant:
        """Save a configuration variant."""
        self.session.add(variant)
        await self.session.commit()
        await self.session.refresh(variant)
        return variant
    
    async def increment_validation_count(self, variant_id: int):
        """Increment validation count for a variant."""
        variant = await self.session.get(GuardrailConfigVariant, variant_id)
        if variant:
            variant.total_validations += 1
            await self.session.commit()
    
    async def record_feedback(self, variant_id: int, is_false_positive: bool, is_false_negative: bool):
        """Record feedback metrics for a variant."""
        variant = await self.session.get(GuardrailConfigVariant, variant_id)
        if variant:
            if is_false_positive:
                variant.false_positives += 1
            if is_false_negative:
                variant.false_negatives += 1
            await self.session.commit()
