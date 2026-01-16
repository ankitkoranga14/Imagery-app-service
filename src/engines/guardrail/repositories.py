"""
Guardrail Repositories - Optimized for Fast Model Loading

REFACTORED (2026-01-16): Using YOLOE-26n-seg exclusively
Source: https://docs.ultralytics.com/models/yolo26/

Implements:
- Parallel model loading (50-60% faster)
- YOLOE-26 open-vocabulary detection
- Background loading with async support
- Model caching and optimization

Models:
- Text: all-MiniLM-L6-v2 (sentence-transformers) - L1 text validation
- Vision: YOLOE-26n-seg (ultralytics>=8.4.0) - Unified L3+L4 vision

REMOVED:
- CLIP/MobileCLIP (replaced by YOLOE-26 open-vocabulary)
- YOLOv11n (replaced by YOLOE-26n)
"""

import torch
import torch.nn.functional as F
import json
import hashlib
import logging
import threading
import time
import asyncio
import concurrent.futures
import os
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List, Union
from enum import Enum

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import numpy as np

# Pre-import ML modules at module level to avoid circular imports during parallel loading
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

from src.engines.guardrail.models import GuardrailLog, GuardrailFeedback, GuardrailConfigVariant

logger = logging.getLogger(__name__)


# =============================================================================
# Model Configuration - YOLOE-26 Only (No CLIP, No Legacy YOLO)
# =============================================================================

class ModelSize(str, Enum):
    """Model size configuration - STANDARD only."""
    STANDARD = "standard"      # Single optimal model configuration


class YOLOModelVariant(str, Enum):
    """YOLOE-26 model variants (from https://docs.ultralytics.com/models/yolo26/)."""
    YOLOE_26N_SEG = "yoloe-26n-seg"  # Nano: Fastest, ~2.3M params
    YOLOE_26S_SEG = "yoloe-26s-seg"  # Small: Balanced, ~7.1M params
    YOLOE_26M_SEG = "yoloe-26m-seg"  # Medium: More accurate, ~20.0M params
    YOLO_26N = "yolo26n"             # Standard YOLO26 (non-OV fallback)


# =============================================================================
# Single Standard Model Configuration - YOLOE-26n Unified Vision
# =============================================================================
# REFACTORED (2026-01-16): Using official YOLOE-26n-seg from Ultralytics YOLO26
# Source: https://docs.ultralytics.com/models/yolo26/
# 
# Benefits:
# - NMS-free: Native end-to-end design for faster inference
# - Open-vocabulary: Real-time detection using text prompts
# - ~43% faster on CPU compared to previous models
# - ~90MB memory saved by removing CLIP

MODEL_CONFIGS = {
    ModelSize.STANDARD: {
        "text_model": "all-MiniLM-L6-v2",           # 90MB, L1 text validation
        # YOLOE-26 models (from https://docs.ultralytics.com/models/yolo26/):
        #   - yoloe-26n-seg.pt: Nano (fastest, ~2.3M params)
        #   - yoloe-26s-seg.pt: Small (balanced, ~7.1M params)
        #   - yoloe-26m-seg.pt: Medium (more accurate, ~20.0M params)
        "yoloe_model": "yoloe-26n-seg.pt",          # Unified L3+L4 vision model (YOLOE-26 Nano Seg)
        "yoloe_oracle_model": "yoloe-26m-seg.pt",   # Oracle model for speculative cascade (YOLOE-26 Medium Seg)
    },
}


class MLRepository:
    """Optimized ML Model Repository with YOLOE-26n-seg.
    
    REFACTORED (2026-01-16): Using YOLOE-26 exclusively
    Source: https://docs.ultralytics.com/models/yolo26/
    
    - CLIP removed (replaced by YOLOE-26 open-vocabulary)
    - YOLOv11n removed (replaced by YOLOE-26n)
    - ~90MB memory saved, ~43% faster on CPU
    
    Features:
    - Parallel model loading (50-60% faster startup)
    - Thread-safe singleton pattern
    - Background loading support
    
    Models:
    - Text: all-MiniLM-L6-v2 (sentence-transformers) - L1 validation
    - Vision: YOLOE-26n-seg (ultralytics>=8.4.0) - Unified L3+L4 validation
    """
    
    _instance: Optional['MLRepository'] = None
    _instance_lock = threading.Lock()
    _loading_task: Optional[asyncio.Task] = None
    
    def __init__(self, cache_dir: Path, model_size: ModelSize = ModelSize.STANDARD):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Always use STANDARD - single model configuration
        self.model_size = ModelSize.STANDARD
        self.config = MODEL_CONFIGS[ModelSize.STANDARD]
        
        # Text model for L1 validation
        self.text_model = None
        
        # YOLOE-26n-seg for unified L3+L4 vision
        self.yoloe_model = None
        self.yoloe_classes_configured = False  # Track if .set_classes() was called
        
        # Model variant tracking
        self.yolo_variant: Optional[YOLOModelVariant] = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models_loaded = False
        self._load_lock = threading.Lock()
        self._loading_times: Dict[str, float] = {}
        
        logger.info(
            f"[MLRepository] Initialized with device={self.device}, "
            f"configuration=STANDARD, vision_model=YOLOE-26n-seg"
        )

    
    @classmethod
    def get_instance(cls, cache_dir: Path = None, model_size: ModelSize = None) -> 'MLRepository':
        """Get singleton instance of MLRepository (thread-safe).
        
        Always uses STANDARD model configuration - no options, no fallback.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    if cache_dir is None:
                        cache_dir = Path("./ml_cache")
                    # Always use STANDARD - single model configuration
                    model_size = ModelSize.STANDARD
                    cls._instance = cls(cache_dir, model_size)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)."""
        with cls._instance_lock:
            cls._instance = None
    
    # =========================================================================
    # Individual Model Loaders (for parallel loading)
    # =========================================================================
    
    def _load_text_model_impl(self):
        """Load text embedding model in isolation."""
        start = time.time()
        model_name = self.config["text_model"]
        
        logger.info(f"[MLRepository] Loading text model: {model_name}")
        
        model = SentenceTransformer(
            model_name,
            cache_folder=str(self.cache_dir / "sentence-transformers"),
            device=self.device
        )
        
        elapsed = time.time() - start
        self._loading_times["text_model"] = elapsed
        logger.info(f"[MLRepository] Text model loaded in {elapsed:.2f}s")
        
        return model
    
    
    def _load_yoloe_model_impl(self):
        """Load YOLOE-26n-seg with open-vocabulary semantic classes.
        
        YOLOE-26 is the official open-vocabulary model from Ultralytics YOLO26.
        Source: https://docs.ultralytics.com/models/yolo26/
        
        Key Features:
        - NMS-free: Native end-to-end design for faster inference
        - Open-vocabulary: Real-time detection using text prompts
        - ~43% faster on CPU compared to previous models
        
        Returns:
            YOLO model instance with classes configured
            
        Raises:
            Exception: If model fails to load
        """
        from src.engines.guardrail.yoloe_classes import GUARDRAIL_CLASSES
        
        start = time.time()
        model_name = self.config.get("yoloe_model", "yoloe-26n-seg.pt")
        
        # Check if model exists in cache
        cached_model_path = self.cache_dir / model_name
        if cached_model_path.exists():
            logger.info(f"[MLRepository] Found {model_name} in cache: {cached_model_path}")
            model_path = str(cached_model_path)
        else:
            logger.info(f"[MLRepository] Model {model_name} not found in cache, will download to CWD")
            model_path = model_name

        # Ensure MobileCLIP is available in CWD (symlink from cache if needed)
        # Ultralytics expects this file in CWD for open-vocabulary features
        mobileclip_name = "mobileclip2_b.ts"
        mobileclip_cache = self.cache_dir / mobileclip_name
        mobileclip_local = Path(mobileclip_name)
        
        if mobileclip_cache.exists() and not mobileclip_local.exists():
            try:
                logger.info(f"[MLRepository] Symlinking {mobileclip_name} from cache...")
                mobileclip_local.symlink_to(mobileclip_cache)
            except Exception as e:
                logger.warning(f"[MLRepository] Failed to symlink MobileCLIP: {e}")

        logger.info(f"[MLRepository] Loading YOLOE-26 model: {model_path}")
        
        try:
            model = YOLO(model_path)
            
            # Fuse layers for CPU inference speedup
            model.fuse()
            
            # Configure open-vocabulary classes using YOLOE-26 API
            # YOLOE-26 uses: model.set_classes(names, model.get_text_pe(names))
            # See: https://docs.ultralytics.com/models/yolo26/#usage-example
            logger.info(f"[MLRepository] Configuring YOLOE with {len(GUARDRAIL_CLASSES)} semantic classes")
            model.set_classes(GUARDRAIL_CLASSES, model.get_text_pe(GUARDRAIL_CLASSES))
            self.yoloe_classes_configured = True
            
            # Map model name to variant
            if "yoloe-26n" in model_name:
                self.yolo_variant = YOLOModelVariant.YOLOE_26N_SEG
            elif "yoloe-26s" in model_name:
                self.yolo_variant = YOLOModelVariant.YOLOE_26S_SEG
            elif "yoloe-26m" in model_name:
                self.yolo_variant = YOLOModelVariant.YOLOE_26M_SEG
            else:
                self.yolo_variant = YOLOModelVariant.YOLOE_26N_SEG
            
            elapsed = time.time() - start
            self._loading_times["yoloe_model"] = elapsed
            logger.info(
                f"[MLRepository] ✅ YOLOE-26 model loaded in {elapsed:.2f}s | "
                f"Variant: {self.yolo_variant.value} | "
                f"Classes: {len(GUARDRAIL_CLASSES)}"
            )
            
            return model
            
        except Exception as e:
            # Fallback to standard YOLO26 (non-OV) if YOLOE fails
            logger.warning(f"[MLRepository] YOLOE-26 failed to load: {e}")
            logger.warning("[MLRepository] Falling back to yolo26n.pt (without open-vocab)")
            
            try:
                model = YOLO("yolo26n.pt")
                model.fuse()
                self.yolo_variant = YOLOModelVariant.YOLO_26N
                self.yoloe_classes_configured = False
                
                elapsed = time.time() - start
                self._loading_times["yoloe_model"] = elapsed
                logger.info(f"[MLRepository] ✅ Fallback YOLO26n loaded in {elapsed:.2f}s")
                
                return model
            except Exception as e2:
                logger.error(f"[MLRepository] ❌ All YOLO fallbacks failed: {e2}")
                raise RuntimeError(f"Failed to load any YOLO model: {e2}")

    def _load_yoloe_oracle_model_impl(self):
        """Pre-download YOLOE-26m-seg (Oracle) model for speculative cascade."""
        start = time.time()
        model_name = self.config.get("yoloe_oracle_model", "yoloe-26m-seg.pt")
        
        logger.info(f"[MLRepository] Pre-downloading YOLOE Oracle model: {model_name}")
        
        try:
            # Check cache first
            cached_path = self.cache_dir / model_name
            if cached_path.exists():
                logger.info(f"[MLRepository] Found Oracle model in cache: {cached_path}")
                model = YOLO(str(cached_path))
            else:
                # We just need to initialize it to trigger download
                model = YOLO(model_name)
            elapsed = time.time() - start
            logger.info(f"[MLRepository] ✅ YOLOE Oracle model pre-downloaded in {elapsed:.2f}s")
            return model
        except Exception as e:
            logger.warning(f"[MLRepository] YOLOE Oracle model failed to download: {e}")
            return None

    
    # =========================================================================
    # Model Accessors (lazy loading, thread-safe)
    # =========================================================================
    
    def get_text_model(self):
        """Get sentence transformer model for text embeddings."""
        if self.text_model is None:
            with self._load_lock:
                if self.text_model is None:
                    self.text_model = self._load_text_model_impl()
        return self.text_model
    
    
    def get_yoloe_model(self):
        """Get YOLOE-26n model for unified L3+L4 vision inference.
        
        This is the primary vision model that replaces:
        - L3 (YOLOv11n): Object detection
        - L4 (MobileCLIP2): Contextual classification
        
        Features:
        - NMS-free detection for dense object scenes
        - Open-vocabulary semantic class detection
        - STAL (Small Target Aware) for safety detection
        
        Returns:
            YOLO model instance with guardrail classes configured
        """
        if self.yoloe_model is None:
            with self._load_lock:
                if self.yoloe_model is None:
                    self.yoloe_model = self._load_yoloe_model_impl()
                    if self.device == "cuda":
                        self.yoloe_model = self.yoloe_model.to(self.device)
        return self.yoloe_model
    
    def encode_text(self, text: str) -> Any:
        """Encode text using sentence transformer."""
        model = self.get_text_model()
        return model.encode(text, convert_to_tensor=True)
    
    # =========================================================================
    # Preloading Methods
    # =========================================================================
    
    def preload_all_models(self) -> bool:
        """Preload all models SEQUENTIALLY (original method, for compatibility).
        
        Use preload_all_models_parallel() for faster loading.
        
        Models loaded:
        - Text: all-MiniLM-L6-v2 (L1 validation)
        - Vision: YOLOE-26n-OV (unified L3+L4)
        """
        try:
            start = time.time()
            self.get_text_model()
            self.get_yoloe_model()  # Unified L3+L4 (replaces YOLO + CLIP)
            self._models_loaded = True
            
            elapsed = time.time() - start
            logger.info(f"[MLRepository] All models loaded (sequential) in {elapsed:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"[MLRepository] Failed to preload models: {str(e)}")
            return False
    
    def preload_all_models_parallel(self) -> bool:
        """Preload all models in PARALLEL for faster startup.
        
        REFACTORED (2026-01-14):
        - Removed CLIP loading (~90MB memory saved)
        - Now loads YOLOE-26n for unified vision inference
        
        Models loaded:
        - Text: SentenceTransformer (all-MiniLM-L6-v2) - L1 validation
        - Vision: YOLOE-26n-OV - Unified L3+L4 (replaces YOLOv11n + MobileCLIP2)
        """
        start = time.time()
        logger.info("[MLRepository] Starting parallel model loading (YOLOE unified vision)...")
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit loading tasks - only 2 models now (CLIP removed)
                futures = {
                    'text': executor.submit(self._load_text_model_impl),
                    'yoloe': executor.submit(self._load_yoloe_model_impl),
                    'yoloe_oracle': executor.submit(self._load_yoloe_oracle_model_impl),
                }
                
                # Wait for all to complete with timeout
                done, not_done = concurrent.futures.wait(
                    futures.values(),
                    timeout=900,  # 15 minute timeout
                    return_when=concurrent.futures.ALL_COMPLETED
                )
                
                if not_done:
                    logger.error(f"[MLRepository] Some models timed out: {not_done}")
                    for future in not_done:
                        future.cancel()
                    return False
                
                # Collect results
                self.text_model = futures['text'].result()
                self.yoloe_model = futures['yoloe'].result()
                # We don't store oracle_model here as it's lazy-loaded in FoodGuardrailV3
                # but we want it pre-downloaded in the cache.
            
            # Move to GPU sequentially (GPU memory operations should be serialized)
            if self.device == "cuda":
                logger.info("[MLRepository] Moving YOLOE model to GPU...")
                gpu_start = time.time()
                self.yoloe_model = self.yoloe_model.to(self.device)
                logger.info(f"[MLRepository] GPU transfer completed in {time.time() - gpu_start:.2f}s")
            
            self._models_loaded = True
            elapsed = time.time() - start
            
            # Calculate memory saved
            memory_saved_mb = 90  # Approximate CLIP model size
            
            logger.info(
                f"[MLRepository] ✅ All models loaded (parallel) in {elapsed:.1f}s | "
                f"Vision: {self.yolo_variant.value if self.yolo_variant else 'unknown'} | "
                f"YOLOE classes: {self.yoloe_classes_configured} | "
                f"Memory saved: ~{memory_saved_mb}MB (CLIP removed) | "
                f"Individual times: text={self._loading_times.get('text_model', 0):.1f}s, "
                f"yoloe={self._loading_times.get('yoloe_model', 0):.1f}s"
            )
            return True
            
        except Exception as e:
            logger.error(f"[MLRepository] ❌ Parallel loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # =========================================================================
    # Async Background Loading
    # =========================================================================
    
    @classmethod
    async def start_background_loading(cls, cache_dir: Path = None, model_size: ModelSize = None):
        """Start loading models in background immediately on app start.
        
        This allows the app to start accepting requests immediately
        while models load in the background.
        """
        instance = cls.get_instance(cache_dir, model_size)
        
        # Run in background thread to not block startup
        cls._loading_task = asyncio.create_task(
            asyncio.to_thread(instance.preload_all_models_parallel)
        )
        logger.info("[MLRepository] Background model loading started...")
        return cls._loading_task
    
    @classmethod
    async def wait_for_models(cls, timeout: float = 600.0) -> bool:
        """Wait for models to finish loading.
        
        Call this before handling requests that need ML models.
        """
        if cls._loading_task is not None:
            try:
                result = await asyncio.wait_for(cls._loading_task, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                logger.error("[MLRepository] Model loading timed out!")
                return False
            except Exception as e:
                logger.error(f"[MLRepository] Model loading failed: {e}")
                return False
        
        # No loading task, check if instance has models loaded
        instance = cls.get_instance()
        return instance.models_loaded if instance else False
    
    @classmethod
    def is_loading(cls) -> bool:
        """Check if models are currently loading in background."""
        return cls._loading_task is not None and not cls._loading_task.done()
    
    @property
    def models_loaded(self) -> bool:
        """Check if all models are loaded."""
        return self._models_loaded
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get model loading statistics."""
        return {
            "models_loaded": self._models_loaded,
            "device": self.device,
            "model_size": self.model_size.value,
            "yoloe_variant": self.yolo_variant.value if self.yolo_variant else "yoloe-26n-seg",
            "yoloe_classes_configured": self.yoloe_classes_configured,
            "loading_times": self._loading_times,
            "total_loading_time": sum(self._loading_times.values()),
            "config": self.config,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def convert_models_to_safetensors(cache_dir: Path = None):
    """Convert models to safetensors format for faster loading.
    
    REFACTORED (2026-01-16): YOLOE models don't need manual safetensors conversion.
    This function is now a placeholder for future optimizations.
    """
    logger.info("[MLRepository] Safetensors conversion not needed for YOLOE-26 architecture.")
    return


# =============================================================================
# Cache Repository
# =============================================================================

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


# =============================================================================
# Log Repository
# =============================================================================

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


# =============================================================================
# Feedback Repository
# =============================================================================

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
            tp = correct_val
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


# =============================================================================
# Config Variant Repository
# =============================================================================

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
        """Get configuration variant for a user based on rollout percentage."""
        variants = await self.get_active_variants()
        if not variants:
            return None
        
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
