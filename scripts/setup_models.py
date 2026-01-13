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
- CLIP: ViT-B-32 with LAION weights (open-clip-torch)
- YOLO: YOLOv11n - 30% faster than v8, +2.2 mAP (ultralytics>=8.3.0)

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
import open_clip
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
        from src.engines.guardrail.repositories import MLRepository, ModelSize
        
        # Always use STANDARD - single model configuration
        ml_repo = MLRepository(cache_path, ModelSize.STANDARD)
        
        download_start = time.time()
        
        if parallel:
            success = ml_repo.preload_all_models_parallel()
        else:
            success = ml_repo.preload_all_models()
        
        download_time = time.time() - download_start
        
        if not success:
            logger.error("❌ Model download failed!")
            return False
        
        logger.info(f"✅ Models downloaded in {download_time:.1f}s")
        
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
    # Step 2: Convert to Safetensors
    # =========================================================================
    if convert_safetensors:
        logger.info("\n[Step 2/4] Converting models to safetensors format...")
        
        try:
            from safetensors.torch import save_file
            import torch
            
            # Convert CLIP model
            clip_safetensors_path = cache_path / "clip_vit_b_32.safetensors"
            
            if not clip_safetensors_path.exists():
                logger.info("Converting CLIP to safetensors...")
                convert_start = time.time()
                
                clip_model, _ = ml_repo.get_clip_model()
                state_dict = {k: v.cpu() for k, v in clip_model.state_dict().items()}
                save_file(state_dict, str(clip_safetensors_path))
                
                convert_time = time.time() - convert_start
                file_size = clip_safetensors_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ CLIP saved to safetensors ({file_size:.1f}MB) in {convert_time:.1f}s")
            else:
                logger.info(f"✅ CLIP safetensors already exists: {clip_safetensors_path}")
            
        except ImportError:
            logger.warning("⚠️ safetensors not installed, skipping conversion")
        except Exception as e:
            logger.warning(f"⚠️ Safetensors conversion failed: {e}")
    else:
        logger.info("\n[Step 2/4] Skipping safetensors conversion")
    
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
            
            # Test CLIP model
            logger.info("Testing CLIP model...")
            clip_model, preprocess = ml_repo.get_clip_model()
            import torch
            with torch.no_grad():
                # Create dummy image
                from PIL import Image
                import numpy as np
                dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                img_tensor = preprocess(dummy_img).unsqueeze(0)
                if ml_repo.device == "cuda":
                    img_tensor = img_tensor.cuda()
                features = clip_model.encode_image(img_tensor)
                assert features is not None
            logger.info("✅ CLIP model validated")
            
            # Test YOLO model
            logger.info("Testing YOLO model...")
            yolo_model = ml_repo.get_yolo_model()
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = yolo_model(dummy_img, verbose=False)
            assert results is not None
            logger.info("✅ YOLO model validated")
            
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
    logger.info(f"CLIP variant: {stats.get('clip_variant', 'unknown')}")
    logger.info(f"YOLO variant: {stats.get('yolo_variant', 'unknown')}")
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

