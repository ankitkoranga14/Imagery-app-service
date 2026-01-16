
import json
import base64
import requests
import os
import time
import re
from pathlib import Path
from typing import Dict, List, Any

# Configuration
API_URL = "http://localhost:8000/api/v1/guardrail/validate"
GROUND_TRUTH_PATH = "/home/ankit.koranga/fast/imagery-app-service/ground_truth.json"
SERVICES_PATH = "/home/ankit.koranga/fast/imagery-app-service/src/engines/guardrail/services.py"
SAMPLE_IMAGES_DIR = "/home/ankit.koranga/fast/imagery-app-service/sample_images"

class GuardrailCalibrator:
    def __init__(self, ground_truth_path: str, services_path: str):
        with open(ground_truth_path, "r") as f:
            self.ground_truth = json.load(f)["ground_truth"]
        self.services_path = services_path
        self.iteration = 1
        self.initial_step = 5.0
        
    def get_current_thresholds(self) -> Dict[str, float]:
        with open(self.services_path, "r") as f:
            content = f.read()
        
        thresholds = {}
        patterns = {
            "darkness_min": r"PHYSICS_DARKNESS_THRESHOLD\s*=\s*(\d+\.?\d*)",
            "glare_max": r"PHYSICS_GLARE_THRESHOLD\s*=\s*(\d+\.?\d*)",
            "cluster_count_max": r"GEOMETRY_MAX_CLUSTERS_READY_TO_EAT\s*=\s*(\d+\.?\d*)",
            "food_score_min": r"CONTEXT_FOOD_THRESHOLD\s*=\s*(\d+\.?\d*)",
            "foreign_object_max": r"CONTEXT_FOREIGN_OBJECT_THRESHOLD\s*=\s*(\d+\.?\d*)",
            "angle_max": r"CONTEXT_POOR_ANGLE_THRESHOLD\s*=\s*(\d+\.?\d*)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                thresholds[key] = float(match.group(1))
        
        return thresholds

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        with open(self.services_path, "r") as f:
            content = f.read()
        
        patterns = {
            "darkness_min": (r"PHYSICS_DARKNESS_THRESHOLD\s*=\s*\d+\.?\d*", "PHYSICS_DARKNESS_THRESHOLD = {}"),
            "glare_max": (r"PHYSICS_GLARE_THRESHOLD\s*=\s*\d+\.?\d*", "PHYSICS_GLARE_THRESHOLD = {}"),
            "cluster_count_max": (r"GEOMETRY_MAX_CLUSTERS_READY_TO_EAT\s*=\s*\d+\.?\d*", "GEOMETRY_MAX_CLUSTERS_READY_TO_EAT = {}"),
            "food_score_min": (r"CONTEXT_FOOD_THRESHOLD\s*=\s*\d+\.?\d*", "CONTEXT_FOOD_THRESHOLD = {}"),
            "foreign_object_max": (r"CONTEXT_FOREIGN_OBJECT_THRESHOLD\s*=\s*\d+\.?\d*", "CONTEXT_FOREIGN_OBJECT_THRESHOLD = {}"),
            "angle_max": (r"CONTEXT_POOR_ANGLE_THRESHOLD\s*=\s*\d+\.?\d*", "CONTEXT_POOR_ANGLE_THRESHOLD = {}")
        }
        
        for key, value in new_thresholds.items():
            if key in patterns:
                pattern, template = patterns[key]
                # Format value to 2 decimal places if it's a float
                val_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                content = re.sub(pattern, template.format(val_str), content)
        
        with open(self.services_path, "w") as f:
            f.write(content)

    def run_test(self) -> List[Dict]:
        results = []
        for item in self.ground_truth:
            image_id = item["image_id"]
            image_path = os.path.join(SAMPLE_IMAGES_DIR, image_id)
            
            with open(image_path, "rb") as f:
                img_bytes = base64.b64encode(f.read()).decode("utf-8")
            
            payload = {
                "prompt": "A photo of food",
                "image_bytes": img_bytes
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    res_data = response.json()
                    results.append({
                        "image_id": image_id,
                        "ground_truth": item["ground_truth_decision"],
                        "guardrail_decision": res_data["status"],
                        "scores": res_data["scores"],
                        "reasons": res_data["reasons"]
                    })
                else:
                    print(f"Error testing {image_id}: {response.status_code}")
            except Exception as e:
                print(f"Exception testing {image_id}: {str(e)}")
        
        return results

    def calibrate(self, max_iterations=10):
        for i in range(max_iterations):
            self.iteration = i + 1
            print(f"\n--- Iteration {self.iteration} ---")
            
            # Clear Redis cache before each test run
            os.system("docker exec guardrail-redis redis-cli FLUSHALL > /dev/null 2>&1")
            
            results = self.run_test()
            matches = 0
            mismatches = []
            
            for res in results:
                print(f"  {res['image_id']}: GT={res['ground_truth']}, GD={res['guardrail_decision']}, Reasons={res['reasons']}")
                if res["ground_truth"] == res["guardrail_decision"]:
                    matches += 1
                else:
                    mismatches.append(res)
            
            accuracy = (matches / len(results)) * 100
            print(f"Accuracy: {accuracy:.2f}% ({matches}/{len(results)})")
            
            if accuracy == 100:
                print("Success! 100% accuracy achieved.")
                break
            
            # Adjust thresholds
            current_thresholds = self.get_current_thresholds()
            new_thresholds = current_thresholds.copy()
            
            # Dynamic step size
            adjustment_step = self.initial_step / (self.iteration)
            
            for mismatch in mismatches:
                gt = mismatch["ground_truth"]
                gd = mismatch["guardrail_decision"]
                scores = mismatch["scores"]
                reasons = mismatch["reasons"]
                
                # FALSE NEGATIVE: GT is BLOCK, Guardrail is PASS (Too lenient)
                if gt == "BLOCK" and gd == "PASS":
                    gt_item = next(item for item in self.ground_truth if item["image_id"] == mismatch["image_id"])
                    violation = gt_item.get("primary_violation")
                    
                    if violation == "multiple_food_items":
                        # Lower the cluster count threshold
                        # If we have 2 clusters and we want to block, threshold should be < 2
                        current_clusters = scores.get("cluster_count", 0)
                        if current_clusters >= 2:
                            new_thresholds["cluster_count_max"] = min(new_thresholds["cluster_count_max"], current_clusters - 0.5)
                    
                    elif violation == "darkness":
                        new_thresholds["darkness_min"] = max(new_thresholds["darkness_min"], scores.get("darkness_score", 35) + adjustment_step)
                    
                    elif violation == "glare":
                        new_thresholds["glare_max"] = min(new_thresholds["glare_max"], scores.get("glare_percentage", 0.12) - 0.01)
                    
                    elif violation == "foreign_objects":
                        # If it's not food, food_score should be low.
                        # But YOLOE uses class counts.
                        # If "unnamed.jpg" (not food) passed, it means it didn't trigger ContextualState.NOT_READY
                        # We might need to adjust classify_contextual_state logic, but for now let's try food_score if available
                        pass

                # FALSE POSITIVE: GT is PASS, Guardrail is BLOCK (Too strict)
                elif gt == "PASS" and gd == "BLOCK":
                    reasons_str = "".join(reasons).lower()
                    if "geometry" in reasons_str or "multiple" in reasons_str:
                        new_thresholds["cluster_count_max"] = max(new_thresholds["cluster_count_max"], scores.get("cluster_count", 0) + 0.5)
                    if "dark" in reasons_str:
                        new_thresholds["darkness_min"] -= adjustment_step
                    if "glare" in reasons_str:
                        new_thresholds["glare_max"] += 0.02

            # Clamp thresholds to sane values
            new_thresholds["darkness_min"] = max(10, min(80, new_thresholds["darkness_min"]))
            new_thresholds["glare_max"] = max(0.01, min(0.5, new_thresholds["glare_max"]))
            new_thresholds["cluster_count_max"] = max(1.0, min(10.0, new_thresholds["cluster_count_max"]))
            
            print(f"Adjusting thresholds: {new_thresholds}")
            self.update_thresholds(new_thresholds)
            
            # Wait for service to reload
            time.sleep(3) 

if __name__ == "__main__":
    calibrator = GuardrailCalibrator(GROUND_TRUTH_PATH, SERVICES_PATH)
    calibrator.calibrate()
