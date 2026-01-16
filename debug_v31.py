import os
import sys
import json
import logging
import base64
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.engines.guardrail.food_guardrail_v3 import FoodGuardrailV3
from src.engines.guardrail.yoloe_classes import GUARDRAIL_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_images():
    # Initialize V3.1 Guardrail
    # Use the same model as in integration
    guardrail = FoodGuardrailV3("yoloe-26n-seg.pt")
    
    sample_dir = Path("sample_images")
    results = {}
    
    # Load ground truth
    with open("ground_truth.json", "r") as f:
        ground_truth = json.load(f)["ground_truth"]
    
    gt_map = {item["image_id"]: item for item in ground_truth}
    
    test_files = [
        "170203151709-jambalaya.jpg",
        "Chicken-Rice-Bowl_EXPS_TOHAM25_25774_P2_MD_04_24_10b.jpg",
        "burger.jpeg",
        "carrot-rice-plate-marble-surface.jpg",
        "thali.jpeg",
        "blur.jpeg",
        "multiple.jpg",
        "fires.jpg"
    ]
    
    print(f"{'Image ID':<60} | {'GT':<6} | {'V3.1':<6} | {'Reason'}")
    print("-" * 120)
    
    for file_name in test_files:
        file_path = sample_dir / file_name
        if not file_path.exists():
            print(f"File not found: {file_name}")
            continue
            
        # Run guardrail
        res = guardrail.process_sync(str(file_path))
        
        gt = gt_map.get(file_name, {}).get("ground_truth_decision", "N/A")
        v31 = res["status"]
        reason = res.get("reason", "")
        details = res.get("details", "")
        
        print(f"{file_name:<60} | {gt:<6} | {v31:<6} | {reason}: {details}")
        
        # Print vision details if blocked
        vision = res.get("vision", {})
        logic = res.get("logic", {})
        detections = vision.get("detections", [])
        if detections:
            print(f"  > Detections: {[(d['class_name'], round(d['confidence'], 2)) for d in detections[:5]]}")
        
        if v31 == "BLOCK" or v31 == "REVIEW":
            print(f"  > Vision: {vision.get('sahi_count', 0)} SAHI dets")
            print(f"  > Logic: {logic.get('cluster_count', 0)} clusters, {logic.get('cluster_metadata', {})}")

if __name__ == "__main__":
    test_images()
