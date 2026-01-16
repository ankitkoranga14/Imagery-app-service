#!/usr/bin/env python3
"""
Food Guardrail V3.0 Calibration Script

Tests the V3.0 implementation against ground truth data
and generates a comprehensive calibration report.

Usage:
    python food_guardrail_calibrator_v3.py [--images-dir ./sample_images] [--ground-truth ./ground_truth.json]
"""

import os
import sys
import json
import time
import base64
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

# Set CPU thread count before importing torch/ultralytics
os.environ['OMP_NUM_THREADS'] = str(max(1, (os.cpu_count() or 4) // 2))
os.environ['MKL_NUM_THREADS'] = str(max(1, (os.cpu_count() or 4) // 2))

from src.engines.guardrail.food_guardrail_v3 import (
    FoodGuardrailV3,
    PhysicsGatesV3,
    KitchenSceneOptimizerV3,
    GUARDRAIL_CLASSES_V3,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_image_as_base64(image_path: str) -> str:
    """Load image and convert to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_ground_truth(gt_path: str) -> Dict[str, str]:
    """Load ground truth labels from JSON file."""
    if not os.path.exists(gt_path):
        logger.warning(f"Ground truth file not found: {gt_path}")
        return {}
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    # Normalize to {filename: PASS/BLOCK}
    labels = {}
    
    # Handle both list and dict formats
    items = gt_data.get('ground_truth', gt_data) if isinstance(gt_data, dict) else gt_data
    
    for item in items:
        if isinstance(item, dict):
            filename = item.get('image_id', item.get('filename', item.get('image', '')))
            expected = item.get('ground_truth_decision', item.get('expected', item.get('expected_output', ''))).upper()
            if filename:
                labels[filename] = expected
    
    return labels


def calculate_metrics(
    results: List[Tuple[str, str, str, bool]]
) -> Dict[str, Any]:
    """
    Calculate precision, recall, F1-score, and accuracy.
    
    Args:
        results: List of (filename, expected, predicted, is_correct) tuples
        
    Returns:
        Dictionary with metrics
    """
    if not results:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Count true positives, false positives, etc.
    # PASS = Positive (food detected correctly)
    # BLOCK = Negative (non-food or quality issue)
    tp = sum(1 for _, exp, pred, _ in results if exp == "PASS" and pred == "PASS")
    fp = sum(1 for _, exp, pred, _ in results if exp == "BLOCK" and pred == "PASS")
    fn = sum(1 for _, exp, pred, _ in results if exp == "PASS" and pred == "BLOCK")
    tn = sum(1 for _, exp, pred, _ in results if exp == "BLOCK" and pred == "BLOCK")
    
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total": total
    }


def run_physics_only_test(img_bgr: np.ndarray, filename: str) -> Dict[str, Any]:
    """Run only physics gates on an image."""
    result = PhysicsGatesV3.check_all_physics(img_bgr)
    
    return {
        "filename": filename,
        "physics_passed": result["passed"],
        "failed_gate": result.get("failed_gate"),
        "gates": {
            gate: {
                "passed": data["passed"],
                "reason": data["reason"]
            }
            for gate, data in result["gates"].items()
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Food Guardrail V3.0 Calibration Script"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./sample_images",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="./ground_truth.json",
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./calibration_reports",
        help="Directory to save calibration reports"
    )
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="Run only physics gates (faster, no model loading)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each image"
    )
    
    args = parser.parse_args()
    
    # Setup
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return 1
    
    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth)
    logger.info(f"Loaded {len(ground_truth)} ground truth labels")
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    logger.info(f"Found {len(image_files)} images in {images_dir}")
    
    # Initialize guardrail
    if not args.physics_only:
        logger.info("Initializing FoodGuardrailV3...")
        guardrail = FoodGuardrailV3("yolov8s-world.pt")
        logger.info("FoodGuardrailV3 initialized")
    else:
        guardrail = None
        logger.info("Running physics-only test (no model loading)")
    
    # Run calibration
    results = []
    physics_results = []
    detailed_results = []
    
    logger.info("\n" + "="*70)
    logger.info("Starting V3.0 Calibration Run")
    logger.info("="*70 + "\n")
    
    total_time = 0
    
    for i, image_file in enumerate(sorted(image_files)):
        filename = image_file.name
        expected = ground_truth.get(filename, "UNKNOWN")
        
        logger.info(f"[{i+1}/{len(image_files)}] Processing: {filename}")
        
        # Load image
        img_bgr = cv2.imread(str(image_file))
        if img_bgr is None:
            logger.warning(f"  Failed to load image: {filename}")
            continue
        
        # Run physics test
        physics_result = run_physics_only_test(img_bgr, filename)
        physics_results.append(physics_result)
        
        if args.physics_only:
            predicted = "BLOCK" if not physics_result["physics_passed"] else "PASS"
            is_correct = (predicted == expected) if expected != "UNKNOWN" else None
            
            results.append((filename, expected, predicted, is_correct))
            
            if args.verbose or not is_correct:
                symbol = "✓" if is_correct else "✗" if is_correct is not None else "?"
                logger.info(f"  {symbol} Expected: {expected}, Predicted: {predicted}")
                if not physics_result["physics_passed"]:
                    logger.info(f"    Failed gate: {physics_result['failed_gate']}")
        else:
            # Run full V3.0 pipeline
            start_time = time.time()
            
            image_base64 = load_image_as_base64(str(image_file))
            result = guardrail.process_sync(image_base64)
            
            elapsed = (time.time() - start_time) * 1000
            total_time += elapsed
            
            predicted = result["status"]
            # Normalize to PASS/BLOCK
            if predicted == "REVIEW":
                predicted = "BLOCK"  # Conservative approach for REVIEW
            
            is_correct = (predicted == expected) if expected != "UNKNOWN" else None
            
            results.append((filename, expected, predicted, is_correct))
            
            detailed = {
                "filename": filename,
                "expected": expected,
                "predicted": predicted,
                "is_correct": is_correct,
                "reason": result.get("reason", ""),
                "details": result.get("details", ""),
                "physics": physics_result,
                "total_time_ms": result.get("total_time_ms", 0),
            }
            detailed_results.append(detailed)
            
            if args.verbose or (is_correct is False):
                symbol = "✓" if is_correct else "✗" if is_correct is not None else "?"
                logger.info(f"  {symbol} Expected: {expected}, Predicted: {predicted} ({elapsed:.1f}ms)")
                if predicted == "BLOCK":
                    logger.info(f"    Reason: {result.get('reason', 'N/A')}")
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("V3.0 CALIBRATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total Images: {metrics['total']}")
    logger.info(f"Accuracy:     {metrics['accuracy']:.2%}")
    logger.info(f"Precision:    {metrics['precision']:.2%}")
    logger.info(f"Recall:       {metrics['recall']:.2%}")
    logger.info(f"F1-Score:     {metrics['f1']:.2%}")
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info(f"  True Positives:  {metrics['true_positives']}")
    logger.info(f"  True Negatives:  {metrics['true_negatives']}")
    logger.info(f"  False Positives: {metrics['false_positives']}")
    logger.info(f"  False Negatives: {metrics['false_negatives']}")
    
    if not args.physics_only and len(image_files) > 0:
        avg_time = total_time / len(image_files)
        logger.info(f"\nAverage Processing Time: {avg_time:.1f}ms")
    
    # List mismatches
    mismatches = [r for r in results if r[3] is False]
    if mismatches:
        logger.info("\nMISMATCHES:")
        for filename, expected, predicted, _ in mismatches:
            logger.info(f"  {filename}: Expected={expected}, Got={predicted}")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"v3_calibration_{timestamp}.json"
    
    report = {
        "version": "v3.0",
        "timestamp": timestamp,
        "config": {
            "images_dir": str(images_dir),
            "ground_truth": args.ground_truth,
            "physics_only": args.physics_only,
        },
        "metrics": metrics,
        "results": [
            {
                "filename": filename,
                "expected": expected,
                "predicted": predicted,
                "is_correct": is_correct
            }
            for filename, expected, predicted, is_correct in results
        ],
        "detailed_results": detailed_results if not args.physics_only else physics_results,
        "mismatches": [
            {"filename": f, "expected": e, "predicted": p}
            for f, e, p, c in mismatches
        ]
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\nReport saved to: {report_file}")
    logger.info("="*70)
    
    # Return exit code based on target F1
    target_f1 = 0.998  # 99.8% target
    if metrics['f1'] >= target_f1:
        logger.info(f"✓ TARGET ACHIEVED: F1-Score {metrics['f1']:.3%} >= {target_f1:.1%}")
        return 0
    else:
        logger.warning(f"✗ TARGET NOT MET: F1-Score {metrics['f1']:.3%} < {target_f1:.1%}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
