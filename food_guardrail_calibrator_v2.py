import os
import json
import requests
import base64
import time
import re
from typing import List, Dict, Any

# Configuration
API_URL = "http://localhost:8000/api/v1/guardrail/validate"
SAMPLE_IMAGES_DIR = "./sample_images"
GROUND_TRUTH_FILE = "ground_truth.json"
SERVICES_FILE = "src/engines/guardrail/services.py"
REPORT_DIR = "./calibration_reports"

class GuardrailCalibrator:
    def __init__(self, ground_truth_path: str, services_path: str):
        self.ground_truth_path = ground_truth_path
        self.services_path = services_path
        with open(ground_truth_path, "r") as f:
            self.ground_truth = json.load(f)["ground_truth"]
        
        if not os.path.exists(REPORT_DIR):
            os.makedirs(REPORT_DIR)
            
        self.iteration = 0
        self.initial_step = 5.0
        self.adjustment_step = self.initial_step

    def get_current_thresholds(self) -> Dict[str, float]:
        with open(self.services_path, "r") as f:
            content = f.read()
        
        thresholds = {}
        patterns = {
            "darkness": r"PHYSICS_DARKNESS_THRESHOLD\s*=\s*(\d+\.?\d*)",
            "glare": r"PHYSICS_GLARE_THRESHOLD\s*=\s*(\d+\.?\d*)",
            "multiple_food_items": r"GEOMETRY_MAX_CLUSTERS_READY_TO_EAT\s*=\s*(\d+\.?\d*)",
            "food_score": r"CONTEXT_FOOD_THRESHOLD\s*=\s*(\d+\.?\d*)",
            "foreign_objects": r"CONTEXT_FOREIGN_OBJECT_THRESHOLD\s*=\s*(\d+\.?\d*)",
            "angle": r"CONTEXT_POOR_ANGLE_THRESHOLD\s*=\s*(\d+\.?\d*)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                thresholds[key] = float(match.group(1))
            else:
                defaults = {
                    "darkness": 35.0, "glare": 0.2, "multiple_food_items": 1.9,
                    "food_score": 0.10, "foreign_objects": 0.1, "angle": 0.6
                }
                thresholds[key] = defaults.get(key, 0.5)
        
        return thresholds

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        with open(self.services_path, "r") as f:
            content = f.read()
        
        patterns = {
            "darkness": (r"PHYSICS_DARKNESS_THRESHOLD\s*=\s*\d+\.?\d*", "PHYSICS_DARKNESS_THRESHOLD = {}"),
            "glare": (r"PHYSICS_GLARE_THRESHOLD\s*=\s*\d+\.?\d*", "PHYSICS_GLARE_THRESHOLD = {}"),
            "multiple_food_items": (r"GEOMETRY_MAX_CLUSTERS_READY_TO_EAT\s*=\s*\d+\.?\d*", "GEOMETRY_MAX_CLUSTERS_READY_TO_EAT = {}"),
            "food_score": (r"CONTEXT_FOOD_THRESHOLD\s*=\s*\d+\.?\d*", "CONTEXT_FOOD_THRESHOLD = {}"),
            "foreign_objects": (r"CONTEXT_FOREIGN_OBJECT_THRESHOLD\s*=\s*\d+\.?\d*", "CONTEXT_FOREIGN_OBJECT_THRESHOLD = {}"),
            "angle": (r"CONTEXT_POOR_ANGLE_THRESHOLD\s*=\s*\d+\.?\d*", "CONTEXT_POOR_ANGLE_THRESHOLD = {}")
        }
        
        for key, value in new_thresholds.items():
            if key in patterns:
                pattern, template = patterns[key]
                val_str = f"{value:.4f}"
                content = re.sub(pattern, template.format(val_str), content)
        
        with open(self.services_path, "w") as f:
            f.write(content)

    def flush_cache(self):
        print("Flushing Redis cache and restarting guardrail-api...")
        os.system("docker exec guardrail-redis redis-cli FLUSHALL > /dev/null 2>&1")
        os.system("docker restart guardrail-api > /dev/null 2>&1")
        time.sleep(15)

    def run_test_batch(self) -> List[Dict]:
        results = []
        for item in self.ground_truth:
            image_id = item["image_id"]
            image_path = os.path.join(SAMPLE_IMAGES_DIR, image_id)
            
            if not os.path.exists(image_path):
                continue

            with open(image_path, "rb") as f:
                img_bytes = base64.b64encode(f.read()).decode("utf-8")
            
            payload = {"prompt": "A photo of food", "image_bytes": img_bytes}
            
            for _ in range(5):
                try:
                    response = requests.post(API_URL, json=payload, timeout=30)
                    if response.status_code == 200:
                        res_data = response.json()
                        results.append({
                            "image_id": image_id,
                            "ground_truth": item["ground_truth_decision"],
                            "guardrail_decision": res_data["status"],
                            "scores": res_data["scores"],
                            "reasons": res_data["reasons"],
                            "gt_analysis": item["your_analysis"]
                        })
                        break
                    else:
                        time.sleep(5)
                except Exception:
                    time.sleep(5)
        return results

    def adjust_threshold(self, parameter, mismatch_type, current_threshold, detected_score):
        new_threshold = current_threshold
        if mismatch_type == "FALSE_NEGATIVE":
            if parameter == "multiple_food_items":
                new_threshold = detected_score - 0.1
                new_threshold = max(new_threshold, 1.0)
            elif parameter == "darkness":
                new_threshold = detected_score + self.adjustment_step
            elif parameter == "glare":
                new_threshold = detected_score - 0.01
                new_threshold = max(new_threshold, 0.05)
            elif parameter == "angle":
                new_threshold = detected_score - 0.05
            elif parameter == "foreign_objects":
                new_threshold = detected_score - 0.05
                new_threshold = max(new_threshold, 0.05)
            elif parameter == "food_score":
                new_threshold = detected_score + 0.05
        elif mismatch_type == "FALSE_POSITIVE":
            if parameter == "multiple_food_items":
                new_threshold = detected_score + 0.1
            elif parameter == "darkness":
                new_threshold = detected_score - self.adjustment_step
            elif parameter == "glare":
                new_threshold = detected_score + 0.02
            elif parameter == "angle":
                new_threshold = detected_score + 0.05
            elif parameter == "foreign_objects":
                new_threshold = detected_score + 0.05
                new_threshold = min(new_threshold, 0.85)
            elif parameter == "food_score":
                new_threshold = detected_score - 0.05
                new_threshold = max(new_threshold, 0.01)
        return new_threshold

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        metrics = {
            "overall": {"accuracy": 0, "total": len(results), "matches": 0},
            "parameters": {
                "multiple_food_items": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "darkness": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "glare": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "angle": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "foreign_objects": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "food_score": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            }
        }
        matches = 0
        for res in results:
            if res["ground_truth"] == res["guardrail_decision"]:
                matches += 1
            gt_analysis = res["gt_analysis"]
            reasons = "".join(res["reasons"]).lower()
            for param in metrics["parameters"]:
                if param == "food_score": continue
                gt_block = gt_analysis[param]["should_block"]
                gd_block = False
                if param == "multiple_food_items" and ("geometry" in reasons or "multiple" in reasons or "cluster" in reasons):
                    gd_block = True
                elif param == "darkness" and "dark" in reasons:
                    gd_block = True
                elif param == "glare" and "glare" in reasons:
                    gd_block = True
                elif param == "angle" and "angle" in reasons:
                    gd_block = True
                elif param == "foreign_objects" and ("foreign" in reasons or "object" in reasons or "hand" in reasons or "packaging" in reasons or "utensil" in reasons):
                    gd_block = True
                if gt_block and gd_block: metrics["parameters"][param]["tp"] += 1
                elif not gt_block and not gd_block: metrics["parameters"][param]["tn"] += 1
                elif not gt_block and gd_block: metrics["parameters"][param]["fp"] += 1
                elif gt_block and not gd_block: metrics["parameters"][param]["fn"] += 1
            gd_context_block = "context" in reasons
            gt_not_food = res["ground_truth"] == "BLOCK" and not any(gt_analysis[p]["should_block"] for p in gt_analysis if p != "food_score")
            if gt_not_food and gd_context_block: metrics["parameters"]["food_score"]["tp"] += 1
            elif not gt_not_food and not gd_context_block: metrics["parameters"]["food_score"]["tn"] += 1
            elif not gt_not_food and gd_context_block: metrics["parameters"]["food_score"]["fp"] += 1
            elif gt_not_food and not gd_context_block: metrics["parameters"]["food_score"]["fn"] += 1
        metrics["overall"]["matches"] = matches
        metrics["overall"]["accuracy"] = (matches / len(results)) * 100 if results else 0
        for param, m in metrics["parameters"].items():
            total = m["tp"] + m["tn"] + m["fp"] + m["fn"]
            m["accuracy"] = (m["tp"] + m["tn"]) / total * 100 if total > 0 else 100
            m["precision"] = m["tp"] / (m["tp"] + m["fp"]) * 100 if (m["tp"] + m["fp"]) > 0 else 100
            m["recall"] = m["tp"] / (m["tp"] + m["fn"]) * 100 if (m["tp"] + m["fn"]) > 0 else 100
        return metrics

    def generate_iteration_report(self, metrics: Dict, results: List[Dict]) -> str:
        report = f"# Iteration {self.iteration} Report\n\n"
        report += f"## Overall Performance\n- **Overall Accuracy**: {metrics['overall']['accuracy']:.1f}%\n- **Matches**: {metrics['overall']['matches']}/{metrics['overall']['total']}\n\n"
        report += "## Parameter-Specific Accuracy\n\n| Parameter | Accuracy | Precision | Recall | TP | TN | FP | FN |\n|-----------|----------|-----------|--------|----|----|----|----|\n"
        for param, m in metrics["parameters"].items():
            report += f"| {param} | {m['accuracy']:.1f}% | {m['precision']:.1f}% | {m['recall']:.1f}% | {m['tp']} | {m['tn']} | {m['fp']} | {m['fn']} |\n"
        report += "\n## Sample Mismatches\n\n"
        for res in results:
            if res["ground_truth"] != res["guardrail_decision"]:
                report += f"- **{res['image_id']}**: GT={res['ground_truth']}, GD={res['guardrail_decision']}\n"
                report += f"  - Reasons: {res['reasons']}\n"
                report += f"  - Scores: {json.dumps(res['scores'])}\n"
        return report

    def calibrate(self, max_iterations=20, target_accuracy=100.0):
        prev_accuracy = 0.0
        for i in range(max_iterations):
            self.iteration = i + 1
            print(f"\n=== Iteration {self.iteration} ===")
            self.flush_cache()
            results = self.run_test_batch()
            metrics = self.calculate_metrics(results)
            accuracy = metrics["overall"]["accuracy"]
            print(f"Overall Accuracy: {accuracy:.2f}% ({metrics['overall']['matches']}/{metrics['overall']['total']})")
            report = self.generate_iteration_report(metrics, results)
            with open(os.path.join(REPORT_DIR, f"report_iter_{self.iteration}.md"), "w") as f:
                f.write(report)
            if accuracy >= target_accuracy:
                print(f"Target accuracy {target_accuracy}% achieved!")
                break
            if accuracy < prev_accuracy - 3.0:
                print(f"Accuracy degraded. Reducing step size.")
                self.adjustment_step *= 0.5
            current_thresholds = self.get_current_thresholds()
            new_thresholds = current_thresholds.copy()
            self.adjustment_step = self.initial_step / (2 ** (self.iteration - 1))
            self.adjustment_step = max(self.adjustment_step, 0.5)
            mismatches = [r for r in results if r["ground_truth"] != r["guardrail_decision"]]
            for res in mismatches:
                gt, gd, scores, gt_analysis, reasons = res["ground_truth"], res["guardrail_decision"], res["scores"], res["gt_analysis"], "".join(res["reasons"]).lower()
                mismatch_type = "FALSE_NEGATIVE" if gt == "BLOCK" and gd == "PASS" else "FALSE_POSITIVE"
                if mismatch_type == "FALSE_NEGATIVE":
                    for param, analysis in gt_analysis.items():
                        if analysis["should_block"]:
                            score_map = {"multiple_food_items": "cluster_count", "darkness": "darkness_score", "glare": "glare_percentage", "angle": "angle_score", "foreign_objects": "foreign_object_score"}
                            score_key = score_map.get(param)
                            detected_score = scores.get(score_key, current_thresholds[param])
                            new_val = self.adjust_threshold(param, mismatch_type, new_thresholds[param], detected_score)
                            print(f"  Adjusting {param}: {new_thresholds[param]:.4f} -> {new_val:.4f} (FN on {res['image_id']})")
                            new_thresholds[param] = new_val
                else:
                    triggered_params = []
                    if "geometry" in reasons or "multiple" in reasons: triggered_params.append("multiple_food_items")
                    if "dark" in reasons: triggered_params.append("darkness")
                    if "glare" in reasons: triggered_params.append("glare")
                    if "angle" in reasons: triggered_params.append("angle")
                    if "foreign" in reasons or "object" in reasons or "hand" in reasons or "packaging" in reasons or "utensil" in reasons: triggered_params.append("foreign_objects")
                    if "context" in reasons: triggered_params.append("food_score")
                    for param in triggered_params:
                        score_map = {"multiple_food_items": "cluster_count", "darkness": "darkness_score", "glare": "glare_percentage", "angle": "angle_score", "foreign_objects": "foreign_object_score", "food_score": "food_score"}
                        score_key = score_map.get(param)
                        detected_score = scores.get(score_key, current_thresholds[param])
                        new_val = self.adjust_threshold(param, mismatch_type, new_thresholds[param], detected_score)
                        print(f"  Adjusting {param}: {new_thresholds[param]:.4f} -> {new_val:.4f} (FP on {res['image_id']})")
                        new_thresholds[param] = new_val
            self.update_thresholds(new_thresholds)
            prev_accuracy = accuracy

if __name__ == "__main__":
    calibrator = GuardrailCalibrator(GROUND_TRUTH_FILE, SERVICES_FILE)
    calibrator.calibrate(max_iterations=10)
