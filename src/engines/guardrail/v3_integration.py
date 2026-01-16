Food Guardrail V3.1 Integration Module

Bridges the V3.1 implementation with the existing services.py
for seamless integration into the validation pipeline.
"""

import os
import time
import base64
import asyncio
import logging
from typing import Dict, Any, Optional

import cv2
import numpy as np

from src.engines.guardrail.food_guardrail_v3 import (
    FoodGuardrailV3,
    PhysicsGatesV3,
    KitchenSceneOptimizerV3,
    GUARDRAIL_CLASSES_V3,
    convert_numpy_types,
    log_block_decision_to_redis,
)

from src.engines.guardrail.yoloe_classes import (
    V3_FOREIGN_OBJECT_THRESHOLD,
    V3_DARKNESS_LAB_L_THRESHOLD,
    V3_GLARE_VALUE_STD_THRESHOLD,
    V3_ANGLE_MAX_DEVIATION,
)

logger = logging.getLogger(__name__)


class ImageGuardrailServiceV3:
    """
    V3.1 Image Guardrail Service Integration.
    
    Drop-in replacement for ImageGuardrailService that uses
    the V3.1 advanced preprocessing and physics gates.
    """
    
    def __init__(self, ml_repo):
        """
        Initialize V3.0 Image Guardrail Service.
        
        Args:
            ml_repo: ML Repository instance for model loading
        """
        self.ml_repo = ml_repo
        self._guardrail_v3: Optional[FoodGuardrailV3] = None
    
    def _get_guardrail_v3(self) -> FoodGuardrailV3:
        """Lazy-load V3.0 guardrail."""
        if self._guardrail_v3 is None:
            logger.info("[V3.0] Initializing FoodGuardrailV3...")
            self._guardrail_v3 = FoodGuardrailV3("yoloe-26n-seg.pt")
        return self._guardrail_v3
    
    def _sync_check_physics_v3(self, image_base64: str) -> Dict[str, Any]:
        """
        V3.0 Physics Check using LAB/HSV/Hough.
        
        Replaces the old RGB-based physics checks.
        
        Args:
            image_base64: Base64-encoded image
            
        Returns:
            Physics check result with V3.0 gates
        """
        try:
            start_time = time.time()
            
            # Decode image
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                return {
                    "physics_passed": False,
                    "physics_error": "Failed to decode image",
                    "version": "v3.0"
                }
            
            # Run V3.0 physics gates
            result = PhysicsGatesV3.check_all_physics(img_bgr)
            
            # Convert to backward-compatible format
            l0_metrics = result["gates"].get("l0_darkness", {}).get("metrics", {})
            l1_metrics = result["gates"].get("l1_glare", {}).get("metrics", {})
            l2_metrics = result["gates"].get("l2_blur", {}).get("metrics", {})
            l25_metrics = result["gates"].get("l25_angle", {}).get("metrics", {})
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            backward_compat = {
                # Core metrics
                "darkness_score": l0_metrics.get("l_mean", 0.0),  # LAB L-channel
                "glare_percentage": l1_metrics.get("blown_ratio", 0.0),
                "blur_variance": l2_metrics.get("laplacian_variance", 0.0),
                "combined_blur_score": l2_metrics.get("combined_score", 0.0),
                
                # V3.0 specific metrics
                "l_channel_mean": l0_metrics.get("l_mean", 0.0),
                "v_channel_std": l1_metrics.get("v_std", 0.0),
                "high_sat_ratio": l1_metrics.get("high_sat_ratio", 0.0),
                "angle_deviation": l25_metrics.get("min_axis_deviation", 0.0),
                
                # Detailed flags
                "is_too_dark": not result["gates"].get("l0_darkness", {}).get("passed", True),
                "is_glary": not result["gates"].get("l1_glare", {}).get("passed", True),
                "is_blurry": not result["gates"].get("l2_blur", {}).get("passed", True),
                "has_invalid_angle": not result["gates"].get("l25_angle", {}).get("passed", True),
                "has_contrast": True,  # Contrast is bundled into other checks in V3.0
                
                # Reasons
                "darkness_reason": result["gates"].get("l0_darkness", {}).get("reason", ""),
                "glare_reason": result["gates"].get("l1_glare", {}).get("reason", ""),
                "blur_reason": result["gates"].get("l2_blur", {}).get("reason", ""),
                "angle_reason": result["gates"].get("l25_angle", {}).get("reason", ""),
                
                # Final decision
                "physics_passed": result["passed"],
                "failed_gate": result.get("failed_gate"),
                "fail_reason": result.get("fail_reason"),
                "physics_time_ms": elapsed_ms,
                
                # V3.1 full result
                "v3_gates": result["gates"],
                "version": "v3.1"
            }
            
            logger.info(
                f"[V3.0 Physics] Status: {'PASS' if result['passed'] else 'FAIL'} | "
                f"L0={result['gates'].get('l0_darkness', {}).get('passed', True)} | "
                f"L1={result['gates'].get('l1_glare', {}).get('passed', True)} | "
                f"L2={result['gates'].get('l2_blur', {}).get('passed', True)} | "
                f"L2.5={result['gates'].get('l25_angle', {}).get('passed', True)} | "
                f"Time={elapsed_ms:.1f}ms"
            )
            
            return convert_numpy_types(backward_compat)
            
        except Exception as e:
            logger.error(f"[V3.0 Physics] Error: {e}")
            return {
                "physics_passed": True,  # Don't block on error
                "physics_error": str(e),
                "physics_skipped": True,
                "version": "v3.0"
            }
    
    async def check_physics(self, image_base64: str) -> Dict[str, Any]:
        """Async wrapper for V3.0 physics check."""
        return await asyncio.to_thread(self._sync_check_physics_v3, image_base64)
    
    def _sync_process_unified_vision_v3(
        self, 
        image_base64: str, 
        prompt: str = ""
    ) -> Dict[str, Any]:
        """
        V3.0 Unified Vision Processing.
        
        Uses vessel-first hierarchy and zero-tolerance foreign object detection.
        
        Args:
            image_base64: Base64-encoded image
            prompt: Optional prompt for context
            
        Returns:
            Vision processing result
        """
        try:
            guardrail = self._get_guardrail_v3()
            result = guardrail.process_sync(image_base64, prompt)
            
            # Convert to backward-compatible format
            passed = result["status"] == "PASS"
            reasons = []
            
            if not passed:
                reasons.append(f"{result['reason']}: {result.get('details', '')}")
            
            backward_compat = {
                "passed": passed,
                "status": result["status"],
                "reasons": reasons,
                "detections": result.get("vision", {}).get("detections", []),
                "all_detections": result.get("vision", {}).get("detections", []),
                "scores": {
                    "cluster_count": result.get("logic", {}).get("cluster_count", 0),
                    "food_score": max(
                        result.get("vision", {}).get("class_confidences", {}).get(i, 0.0)
                        for i in [0, 1, 2, 3]
                    ) if result.get("vision", {}).get("class_confidences") else 0.0,
                    "foreign_object_score": max(
                        result.get("vision", {}).get("class_confidences", {}).get(i, 0.0)
                        for i in [9, 10]
                    ) if result.get("vision", {}).get("class_confidences") else 0.0,
                    "angle_score": 0.0 if result.get("physics", {}).get("gates", {}).get("l25_angle", {}).get("passed", True) else 1.0,
                },
                "metrics": {
                    "inference_time_ms": result.get("vision", {}).get("inference_time_ms", 0),
                    "total_time_ms": result.get("total_time_ms", 0)
                },
                "kitchen_metadata": result.get("logic", {}).get("cluster_metadata", {}),
                "context_state": result.get("logic", {}).get("context_state", "unknown"),
                "is_escalated": result.get("is_escalated", False),
                "sahi_count": result.get("vision", {}).get("sahi_count", 0),
                "version": "v3.1"
            }
            
            # Add foreign objects if detected
            if "foreign_objects" in result:
                backward_compat["foreign_objects"] = result["foreign_objects"]
            
            return convert_numpy_types(backward_compat)
            
        except Exception as e:
            logger.error(f"[V3.0 Vision] Error: {e}")
            return {
                "passed": False,
                "status": "DIVERGENCE_ERROR",
                "reasons": [f"System Error: {str(e)}"],
                "detections": [],
                "scores": {},
                "metrics": {},
                "version": "v3.0"
            }
    
    async def process_unified_vision(
        self, 
        image_base64: str, 
        prompt: str = ""
    ) -> Dict[str, Any]:
        """Async wrapper for V3.0 unified vision."""
        return await asyncio.to_thread(
            self._sync_process_unified_vision_v3, 
            image_base64, 
            prompt
        )
    
    async def process_full_v3(
        self, 
        image_base64: str, 
        prompt: str = ""
    ) -> Dict[str, Any]:
        """
        Full V3.0 processing pipeline.
        
        Runs physics + vision + logic in a single call.
        
        Args:
            image_base64: Base64-encoded image
            prompt: Optional prompt
            
        Returns:
            Complete V3.0 result
        """
        guardrail = self._get_guardrail_v3()
        return await guardrail.process(image_base64, prompt)


def create_v3_image_service(ml_repo) -> ImageGuardrailServiceV3:
    """
    Factory function to create V3.0 Image Guardrail Service.
    
    Args:
        ml_repo: ML Repository instance
        
    Returns:
        ImageGuardrailServiceV3 instance
    """
    return ImageGuardrailServiceV3(ml_repo)


# =============================================================================
# V3.0 SERVICE UPGRADE HELPER
# =============================================================================

def upgrade_image_service_to_v3(existing_service) -> ImageGuardrailServiceV3:
    """
    Upgrade an existing ImageGuardrailService to V3.0.
    
    Args:
        existing_service: Existing ImageGuardrailService instance
        
    Returns:
        ImageGuardrailServiceV3 instance with same ml_repo
    """
    return ImageGuardrailServiceV3(existing_service.ml_repo)
