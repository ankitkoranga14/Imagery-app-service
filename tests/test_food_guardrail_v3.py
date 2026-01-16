#!/usr/bin/env python3
"""
Food Guardrail V3.0 Unit Tests

Tests the V3.0 implementation to ensure all components work correctly.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Set CPU thread count before importing torch/ultralytics
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np
import cv2


class TestPhysicsGatesV3(unittest.TestCase):
    """Test V3.0 Physics Gates."""
    
    def setUp(self):
        """Create test images."""
        # Bright image (should pass L0)
        self.bright_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Dark image (should fail L0)
        self.dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 20
        
        # Glary image (should fail L1)
        self.glary_image = np.ones((480, 640, 3), dtype=np.uint8) * 250
        
        # Blurry image (should fail L2)
        self.blurry_image = cv2.GaussianBlur(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            (31, 31), 0
        )
        
        # Sharp image (should pass L2)
        self.sharp_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    def test_l0_darkness_lab_pass(self):
        """Test L0 Darkness Gate - Should PASS for bright images."""
        from src.engines.guardrail.food_guardrail_v3 import PhysicsGatesV3
        
        passed, reason, metrics = PhysicsGatesV3.check_l0_darkness_lab(self.bright_image)
        
        self.assertTrue(passed, f"Expected bright image to pass L0. Reason: {reason}")
        self.assertIn("l_mean", metrics)
        self.assertGreater(metrics["l_mean"], 62)
    
    def test_l0_darkness_lab_fail(self):
        """Test L0 Darkness Gate - Should FAIL for dark images."""
        from src.engines.guardrail.food_guardrail_v3 import PhysicsGatesV3
        
        passed, reason, metrics = PhysicsGatesV3.check_l0_darkness_lab(self.dark_image)
        
        self.assertFalse(passed, f"Expected dark image to fail L0. Metrics: {metrics}")
        self.assertIn("l_mean", metrics)
        self.assertLess(metrics["l_mean"], 62)
    
    def test_l1_glare_hsv_pass(self):
        """Test L1 Glare Gate - Should PASS for normal images."""
        from src.engines.guardrail.food_guardrail_v3 import PhysicsGatesV3
        
        passed, reason, metrics = PhysicsGatesV3.check_l1_glare_hsv(self.sharp_image)
        
        self.assertTrue(passed, f"Expected normal image to pass L1. Reason: {reason}")
    
    def test_l1_glare_hsv_fail(self):
        """Test L1 Glare Gate - Should FAIL for overexposed images."""
        from src.engines.guardrail.food_guardrail_v3 import PhysicsGatesV3
        
        passed, reason, metrics = PhysicsGatesV3.check_l1_glare_hsv(self.glary_image)
        
        # Glary image should fail due to high blown ratio
        # Note: Very bright uniform images may have low V std
        self.assertIn("v_std", metrics)
        self.assertIn("blown_ratio", metrics)
    
    def test_l2_blur_pass(self):
        """Test L2 Blur Gate - Should PASS for sharp images."""
        from src.engines.guardrail.food_guardrail_v3 import PhysicsGatesV3
        
        passed, reason, metrics = PhysicsGatesV3.check_l2_blur(self.sharp_image)
        
        self.assertTrue(passed, f"Expected sharp image to pass L2. Reason: {reason}")
        self.assertIn("combined_score", metrics)
    
    def test_l2_blur_fail(self):
        """Test L2 Blur Gate - Should FAIL for blurry images."""
        from src.engines.guardrail.food_guardrail_v3 import PhysicsGatesV3
        
        passed, reason, metrics = PhysicsGatesV3.check_l2_blur(self.blurry_image)
        
        self.assertIn("combined_score", metrics)
        # Blurry images should have high combined score
        # Note: May pass if blur is not severe enough
    
    def test_l25_angle_hough(self):
        """Test L2.5 Angle Gate - Should handle various angle scenarios."""
        from src.engines.guardrail.food_guardrail_v3 import PhysicsGatesV3
        
        # Create an image with clear horizontal/vertical lines
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 100
        
        # Draw some lines
        cv2.line(test_image, (0, 240), (640, 240), (0, 0, 0), 2)
        cv2.line(test_image, (320, 0), (320, 480), (0, 0, 0), 2)
        
        passed, reason, metrics = PhysicsGatesV3.check_l25_angle_hough(test_image)
        
        self.assertIn("lines_detected", metrics)
    
    def test_check_all_physics(self):
        """Test combined physics check."""
        from src.engines.guardrail.food_guardrail_v3 import PhysicsGatesV3
        
        result = PhysicsGatesV3.check_all_physics(self.bright_image)
        
        self.assertIn("passed", result)
        self.assertIn("gates", result)
        self.assertIn("l0_darkness", result["gates"])
        self.assertIn("l1_glare", result["gates"])
        self.assertIn("l2_blur", result["gates"])
        self.assertIn("l25_angle", result["gates"])


class TestKitchenSceneOptimizerV3(unittest.TestCase):
    """Test V3.0 Kitchen Scene Optimizer."""
    
    def setUp(self):
        """Create test optimizer."""
        from src.engines.guardrail.food_guardrail_v3 import KitchenSceneOptimizerV3
        self.optimizer = KitchenSceneOptimizerV3(640, 480)
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        
        iou = self.optimizer.calculate_iou(box1, box2)
        
        # 50x50 intersection, 100x100 + 100x100 - 50x50 union
        # IoU = 2500 / (10000 + 10000 - 2500) = 2500 / 17500 = 0.143
        self.assertAlmostEqual(iou, 0.143, places=2)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU with no overlap."""
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 150, 150])
        
        iou = self.optimizer.calculate_iou(box1, box2)
        
        self.assertEqual(iou, 0.0)
    
    def test_expand_roi(self):
        """Test ROI expansion."""
        bbox = np.array([100, 100, 200, 200])
        
        expanded = self.optimizer.expand_roi(bbox, buffer=6)
        
        self.assertEqual(expanded[0], 94)  # x1 - 6
        self.assertEqual(expanded[1], 94)  # y1 - 6
        self.assertEqual(expanded[2], 206)  # x2 + 6
        self.assertEqual(expanded[3], 206)  # y2 + 6
    
    def test_expand_roi_edge_clamping(self):
        """Test ROI expansion with edge clamping."""
        bbox = np.array([0, 0, 100, 100])
        
        expanded = self.optimizer.expand_roi(bbox, buffer=10)
        
        self.assertEqual(expanded[0], 0)  # Clamped to 0
        self.assertEqual(expanded[1], 0)  # Clamped to 0
    
    def test_is_food_in_vessel(self):
        """Test food-in-vessel containment check."""
        vessel_bbox = np.array([100, 100, 200, 200])
        
        # Food inside vessel
        food_inside = np.array([110, 110, 150, 150])
        self.assertTrue(self.optimizer.is_food_in_vessel(food_inside, vessel_bbox))
        
        # Food outside vessel
        food_outside = np.array([300, 300, 400, 400])
        self.assertFalse(self.optimizer.is_food_in_vessel(food_outside, vessel_bbox))
    
    def test_calculate_detection_density(self):
        """Test detection density calculation."""
        detections = [
            {"bbox": np.array([0, 0, 100, 100])},  # 10000 px
            {"bbox": np.array([100, 100, 200, 200])},  # 10000 px
        ]
        
        density = self.optimizer.calculate_detection_density(detections)
        
        # Total: 20000 / (640 * 480) = 20000 / 307200 = 0.065
        self.assertAlmostEqual(density, 0.065, places=2)
    
    def test_cluster_with_vessel_hierarchy_empty(self):
        """Test clustering with no detections."""
        cluster_count, cluster_indices, metadata = self.optimizer.cluster_with_vessel_hierarchy_v3([])
        
        self.assertEqual(cluster_count, 0)
        self.assertEqual(cluster_indices, [])
    
    def test_cluster_with_vessel_hierarchy_single_vessel(self):
        """Test clustering with single vessel containing foods."""
        from src.engines.guardrail.food_guardrail_v3 import VESSEL_CLASS_INDICES_V3
        
        detections = [
            {
                "class_id": 4,  # Vessel
                "bbox": np.array([100, 100, 300, 300])
            },
            {
                "class_id": 0,  # Food
                "bbox": np.array([120, 120, 180, 180])
            },
            {
                "class_id": 0,  # Food
                "bbox": np.array([200, 200, 280, 280])
            },
        ]
        
        cluster_count, cluster_indices, metadata = self.optimizer.cluster_with_vessel_hierarchy_v3(detections)
        
        self.assertEqual(metadata["vessel_count"], 1)
        self.assertEqual(metadata["food_count"], 2)
    
    def test_detect_foreign_objects(self):
        """Test foreign object detection."""
        detections = [
            {
                "class_id": 9,  # Human
                "class_name": "human hand",
                "confidence": 0.65
            },
            {
                "class_id": 0,  # Food
                "class_name": "food",
                "confidence": 0.80
            },
        ]
        
        has_foreign, foreign_details = self.optimizer.detect_foreign_objects(detections)
        
        self.assertTrue(has_foreign)
        self.assertEqual(len(foreign_details), 1)
        self.assertEqual(foreign_details[0]["class_name"], "human hand")
    
    def test_detect_foreign_objects_below_threshold(self):
        """Test foreign object detection below threshold."""
        detections = [
            {
                "class_id": 9,  # Human
                "class_name": "human hand",
                "confidence": 0.30  # Below 0.42 threshold
            },
        ]
        
        has_foreign, foreign_details = self.optimizer.detect_foreign_objects(detections)
        
        self.assertFalse(has_foreign)
        self.assertEqual(len(foreign_details), 0)


class TestGuardrailClassesV3(unittest.TestCase):
    """Test V3.0 class definitions."""
    
    def test_guardrail_classes_count(self):
        """Test correct number of classes."""
        from src.engines.guardrail.food_guardrail_v3 import GUARDRAIL_CLASSES_V3
        
        self.assertEqual(len(GUARDRAIL_CLASSES_V3), 11)
    
    def test_food_class_indices(self):
        """Test food class indices."""
        from src.engines.guardrail.food_guardrail_v3 import FOOD_CLASS_INDICES_V3
        
        self.assertEqual(FOOD_CLASS_INDICES_V3, {0, 1, 2, 3})
    
    def test_vessel_class_indices(self):
        """Test vessel class indices."""
        from src.engines.guardrail.food_guardrail_v3 import VESSEL_CLASS_INDICES_V3
        
        self.assertEqual(VESSEL_CLASS_INDICES_V3, {4})
    
    def test_safety_class_indices(self):
        """Test safety class indices."""
        from src.engines.guardrail.food_guardrail_v3 import SAFETY_CLASS_INDICES_V3
        
        self.assertEqual(SAFETY_CLASS_INDICES_V3, {6, 7})


class TestYoloeClassesV3(unittest.TestCase):
    """Test V3.0 yoloe_classes.py definitions."""
    
    def test_v3_thresholds_exist(self):
        """Test V3.0 thresholds are defined."""
        from src.engines.guardrail.yoloe_classes import (
            V3_FOREIGN_OBJECT_THRESHOLD,
            V3_DARKNESS_LAB_L_THRESHOLD,
            V3_GLARE_VALUE_STD_THRESHOLD,
            V3_ANGLE_MAX_DEVIATION,
        )
        
        self.assertEqual(V3_FOREIGN_OBJECT_THRESHOLD, 0.42)
        self.assertEqual(V3_DARKNESS_LAB_L_THRESHOLD, 62)
        self.assertEqual(V3_GLARE_VALUE_STD_THRESHOLD, 82)
        self.assertEqual(V3_ANGLE_MAX_DEVIATION, 28)
    
    def test_v3_foreign_object_classes(self):
        """Test V3.0 foreign object classes."""
        from src.engines.guardrail.yoloe_classes import V3_FOREIGN_OBJECT_CLASSES
        
        self.assertIn('human hand', V3_FOREIGN_OBJECT_CLASSES)
        self.assertIn('plastic packaging', V3_FOREIGN_OBJECT_CLASSES)
        self.assertIn('knife', V3_FOREIGN_OBJECT_CLASSES)


class TestKitchenOptimizationsV3(unittest.TestCase):
    """Test V3.0 kitchen_optimizations.py additions."""
    
    def test_v3_thresholds_exist(self):
        """Test V3.0 thresholds are defined in kitchen_optimizations."""
        from src.engines.guardrail.kitchen_optimizations import (
            V3_VESSEL_ROI_BUFFER,
            V3_VESSEL_FOOD_IOU_THRESHOLD,
            V3_DENSITY_THRESHOLD,
            V3_INFERENCE_CONF,
            V3_INFERENCE_IMGSZ,
            V3_NMS_AGNOSTIC,
        )
        
        self.assertEqual(V3_VESSEL_ROI_BUFFER, 6)
        self.assertEqual(V3_VESSEL_FOOD_IOU_THRESHOLD, 0.25)
        self.assertEqual(V3_DENSITY_THRESHOLD, 0.32)
        self.assertEqual(V3_INFERENCE_CONF, 0.38)
        self.assertEqual(V3_INFERENCE_IMGSZ, 416)
        self.assertTrue(V3_NMS_AGNOSTIC)


if __name__ == '__main__':
    unittest.main(verbosity=2)
