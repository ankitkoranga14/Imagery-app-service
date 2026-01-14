
import os
import base64
import json
import requests
from pathlib import Path

def run_benchmark():
    api_url = "http://localhost:8000/api/v1/guardrail/validate"
    image_dir = Path("./sample_images")
    images = list(image_dir.glob("*"))
    
    # Primary Vision Model (Me) - Manual labels for comparison
    # Format: {filename: expected_status}
    primary_model_labels = {
        "burger.jpeg": "PASS",
        "blur.jpeg": "BLOCK",
        "multiple.jpg": "BLOCK",
        "thali.jpeg": "PASS",
        "schema.png": "BLOCK",
        "170203151709-jambalaya.jpg": "PASS",
        "fires.jpg": "PASS",
        "multiple.webp": "BLOCK",
        "mult.webp": "BLOCK",
        "blur2.jpeg": "BLOCK",
        "blur.jpeg": "BLOCK",
        "blurr.jpg": "BLOCK",
        "blurrrr.jpeg": "BLOCK",
        "blu.webp": "BLOCK",
        "unnamed.jpg": "BLOCK", # Guardrail says not food (score ~0.0)
    }
    
    results = []
    
    for img_path in images:
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp', '.avif']:
            continue
            
        print(f"Processing {img_path.name}...")
        
        try:
            with open(img_path, "rb") as f:
                img_bytes = base64.b64encode(f.read()).decode("utf-8")
            
            payload = {
                "prompt": "A delicious food item",
                "image_bytes": img_bytes
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                guardrail_result = response.json()
                status = guardrail_result.get("status")
                reasons = guardrail_result.get("reasons", [])
                scores = guardrail_result.get("scores", {})
                
                expected = primary_model_labels.get(img_path.name, "UNKNOWN")
                
                results.append({
                    "filename": img_path.name,
                    "primary_model_status": expected,
                    "guardrail_status": status,
                    "reasons": reasons,
                    "scores": scores,
                    "divergence": expected != status if expected != "UNKNOWN" else False
                })
            else:
                print(f"Error from API for {img_path.name}: {response.status_code} - {response.text}")
                results.append({
                    "filename": img_path.name,
                    "primary_model_status": primary_model_labels.get(img_path.name, "UNKNOWN"),
                    "guardrail_status": "ERROR",
                    "error": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            results.append({
                "filename": img_path.name,
                "primary_model_status": primary_model_labels.get(img_path.name, "UNKNOWN"),
                "guardrail_status": "ERROR",
                "error": str(e)
            })
            
    # Save results
    with open("audit_log.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Benchmark complete. Results saved to audit_log.json")

if __name__ == "__main__":
    run_benchmark()
