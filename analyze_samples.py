
import json
import base64
import requests
import os
from pathlib import Path
import time

def analyze_samples():
    json_path = "/home/ankit.koranga/fast/imagery-app-service/sample_images/extract-data-2026-01-14.json"
    api_url = "http://localhost:8000/api/v1/guardrail/validate"
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    samples = data.get("food_images", [])
    results = []
    
    print(f"Starting analysis of {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        url = sample.get("image_url")
        description = sample.get("image_description", "")
        name = sample.get("food_item_name", "Unknown")
        
        print(f"[{i+1}/{len(samples)}] Processing: {name}...")
        
        try:
            # Download image
            img_response = requests.get(url, timeout=10)
            if img_response.status_code != 200:
                print(f"  Failed to download image from {url}")
                continue
            
            img_bytes = base64.b64encode(img_response.content).decode("utf-8")
            
            # Prepare payload
            payload = {
                "prompt": f"A photo of {name}. {description}",
                "image_bytes": img_bytes
            }
            
            # Call Guardrail API
            start_time = time.time()
            response = requests.post(api_url, json=payload, timeout=30)
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                res_data = response.json()
                status = res_data.get("status")
                reasons = res_data.get("reasons", [])
                scores = res_data.get("scores", {})
                
                # Heuristic for "Expected" status based on description
                expected = "PASS"
                if any(word in description.lower() for word in ["raw", "spoiled", "rotten", "moldy"]):
                    expected = "BLOCK"
                if "two" in description.lower() or "multiple" in description.lower() or "side of" in description.lower():
                    # This is tricky, "side of fries" might be okay if clustered
                    pass
                
                results.append({
                    "url": url,
                    "name": name,
                    "description": description,
                    "guardrail_status": status,
                    "guardrail_reasons": reasons,
                    "guardrail_scores": scores,
                    "latency_ms": latency,
                    "divergence_hint": expected != status if expected == "BLOCK" else False # Only flag if we expected a block and got a pass for now
                })
            else:
                print(f"  API Error: {response.status_code}")
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            
        # Limit to first 20 for initial analysis to save time/resources
        if i >= 19:
            break
            
    with open("sample_analysis_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Analysis complete. Results saved to sample_analysis_report.json")

if __name__ == "__main__":
    analyze_samples()
