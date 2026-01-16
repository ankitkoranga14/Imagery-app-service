import torch
import os

weight = "mobileclip2_b.ts"
print(f"Testing torch.jit.load on {weight}...")
try:
    model = torch.jit.load(weight, map_location="cpu")
    print("✅ Successfully loaded with torch.jit.load")
except Exception as e:
    print(f"❌ Failed to load: {e}")
    import traceback
    traceback.print_exc()
