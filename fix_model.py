from ultralytics import YOLO
import os

model_name = 'yoloe-26n-seg.pt'
print(f"Loading {model_name}...")
model = YOLO(model_name)

classes = ["food", "vessel", "hand"]
print(f"Triggering text encoder download for {classes}...")
try:
    pe = model.get_text_pe(classes)
    print("✅ Text encoder loaded successfully")
except Exception as e:
    print(f"❌ Text encoder failed: {e}")
    import traceback
    traceback.print_exc()
