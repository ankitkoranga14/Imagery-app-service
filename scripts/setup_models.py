#!/usr/bin/env python3
"""
Model Setup Script - Pre-download and Optimize ML Models

This script:
1. Downloads all required ML models (standard pip packages only)
2. Converts models to safetensors format for faster loading
3. Validates model integrity
4. Reports loading times

Models (standard pip packages only):
- Text: all-MiniLM-L6-v2 (sentence-transformers)
- Vision: YOLOE-26n-seg (ultralytics>=8.4.0) - Unified L3+L4 vision

Run this during Docker build to avoid download at runtime:
    python scripts/setup_models.py

Environment variables:
    ML_MODEL_CACHE_DIR: Directory to cache models (default: ./ml_cache)
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pre-import ML modules to avoid circular imports during parallel loading
# This must happen BEFORE any parallel threads try to import these
logger.info("Pre-importing ML modules...")
import torch
from sentence_transformers import SentenceTransformer
# import open_clip # Removed
from ultralytics import YOLO
logger.info("ML modules imported successfully")


def setup_models(
    cache_dir: str = "./ml_cache",
    convert_safetensors: bool = True,
    validate: bool = True,
    parallel: bool = True,
):
    """Download and setup all ML models (STANDARD configuration only).
    
    Args:
        cache_dir: Directory to cache models
        convert_safetensors: Whether to convert to safetensors format
        validate: Whether to validate models after download
        parallel: Whether to use parallel loading
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ML Model Setup Script")
    logger.info("=" * 60)
    logger.info(f"Cache directory: {cache_path.absolute()}")
    logger.info(f"Model configuration: STANDARD (single optimal model)")
    logger.info(f"Convert to safetensors: {convert_safetensors}")
    logger.info(f"Parallel loading: {parallel}")
    logger.info("=" * 60)
    
    total_start = time.time()
    
    # =========================================================================
    # Step 1: Download Models
    # =========================================================================
    logger.info("\n[Step 1/4] Downloading ML models...")
    
    # Set environment variables
    os.environ["ML_MODEL_CACHE_DIR"] = str(cache_path)
    
    try:
        # 1. Explicitly download MobileCLIP (Required for YOLOE Open Vocabulary)
        # We download this directly to cache to avoid runtime downloads
        mobileclip_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts"
        mobileclip_path = cache_path / "mobileclip2_b.ts"
        
        if not mobileclip_path.exists():
            logger.info(f"Downloading MobileCLIP (required for open-vocab) to {mobileclip_path}...")
            torch.hub.download_url_to_file(mobileclip_url, str(mobileclip_path))
        else:
            logger.info(f"MobileCLIP already in cache: {mobileclip_path}")

        # 2. Load other models via MLRepository
        from src.engines.guardrail.repositories import MLRepository, ModelSize
        
        # Always use STANDARD - single model configuration
        ml_repo = MLRepository(cache_path, ModelSize.STANDARD)
        
        download_start = time.time()
        
        # This will trigger YOLO downloads (likely to CWD)
        if parallel:
            success = ml_repo.preload_all_models_parallel()
        else:
            success = ml_repo.preload_all_models()
        
        download_time = time.time() - download_start
        
        if not success:
            logger.error("❌ Model download failed!")
            return False
        
        logger.info(f"✅ Models downloaded in {download_time:.1f}s")
        
        # 3. Move any models downloaded to CWD into cache
        logger.info("Ensuring all models are in cache...")
        for ext in ["*.pt", "*.ts"]:
            for f in Path(".").glob(ext):
                target = cache_path / f.name
                if not target.exists():
                    logger.info(f"Moving {f.name} to cache...")
                    f.rename(target)
                else:
                    logger.info(f"{f.name} already in cache, removing local copy...")
                    f.unlink()
        
        # Log individual times
        stats = ml_repo.get_loading_stats()
        for model, load_time in stats.get("loading_times", {}).items():
            logger.info(f"   - {model}: {load_time:.1f}s")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Step 2: Convert to Safetensors (Skipped for YOLOE)
    # =========================================================================
    # YOLOE models are already optimized or don't need safetensors conversion
    # in the same way CLIP did.
    logger.info("\n[Step 2/4] Skipping safetensors conversion (not needed for YOLOE)")
    
    # =========================================================================
    # Step 3: Validate Models
    # =========================================================================
    if validate:
        logger.info("\n[Step 3/4] Validating models...")
        
        try:
            # Test text model
            logger.info("Testing text model...")
            text_model = ml_repo.get_text_model()
            embedding = text_model.encode("test food prompt")
            assert embedding is not None and len(embedding) > 0
            logger.info("✅ Text model validated")
            
            # Test YOLOE model
            logger.info("Testing YOLOE-26n model...")
            
            # Ensure MobileCLIP is available in CWD (symlink from cache if needed)
            mobileclip_name = "mobileclip2_b.ts"
            mobileclip_cache = cache_path / mobileclip_name
            mobileclip_local = Path(mobileclip_name)
            
            if mobileclip_cache.exists() and not mobileclip_local.exists():
                logger.info(f"Symlinking {mobileclip_name} for validation...")
                mobileclip_local.symlink_to(mobileclip_cache)
            
            yolo_model = ml_repo.get_yoloe_model()
            
            # Pre-load text encoder to avoid runtime download
            from src.engines.guardrail.yoloe_classes import GUARDRAIL_CLASSES
            logger.info("Pre-loading YOLOE text encoder (CLIP)...")
            yolo_model.get_text_pe(GUARDRAIL_CLASSES)
            
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = yolo_model(dummy_img, verbose=False)
            assert results is not None
            logger.info("✅ YOLOE-26n model and text encoder validated")
            
            logger.info("✅ All models validated successfully")
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        logger.info("\n[Step 3/4] Skipping validation")
    
    # =========================================================================
    # Step 4: Report Summary
    # =========================================================================
    logger.info("\n[Step 4/4] Summary")
    
    total_time = time.time() - total_start
    
    # Calculate cache size
    total_size = 0
    file_count = 0
    for f in cache_path.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size
            file_count += 1
    
    # Get model variant info
    stats = ml_repo.get_loading_stats()
    
    logger.info("=" * 60)
    logger.info("✅ Model setup complete!")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Cache size: {total_size / (1024 * 1024):.1f}MB")
    logger.info(f"Files cached: {file_count}")
    logger.info(f"Device: {ml_repo.device}")
    logger.info(f"YOLOE variant: {stats.get('yoloe_variant', 'unknown')}")
    logger.info(f"YOLOE classes: {stats.get('yoloe_classes_configured', False)}")
    logger.info(f"Model config: {stats.get('config', {})}")
    logger.info("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and setup ML models for guardrail service"
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("ML_MODEL_CACHE_DIR", "./ml_cache"),
        help="Directory to cache models"
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Skip safetensors conversion"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip model validation"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential loading instead of parallel"
    )
    
    args = parser.parse_args()
    
    success = setup_models(
        cache_dir=args.cache_dir,
        convert_safetensors=not args.no_safetensors,
        validate=not args.no_validate,
        parallel=not args.sequential,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

