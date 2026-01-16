from ultralytics import YOLO
import cv2
import numpy as np
import os

model_path = 'yoloe-26n-seg.pt'
model = YOLO(model_path)

# Test image
img_path = 'sample_images/170203151709-jambalaya.jpg'
if not os.path.exists(img_path):
    print(f"File not found: {img_path}")
    exit()

img = cv2.imread(img_path)

# Try different prompts
prompts_sets = [
    ["food", "vessel", "hand"],
    ["plated meal", "bowl", "hand"],
    ["jambalaya", "rice", "bowl"],
    [
        "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, sandwich",
        "snack, dessert, cookie, cake, pastry, muffin, donut",
        "beverage, drink, coffee, tea, juice, water, smoothie",
        "food, meal, dish, generic food item",
        "bowl, plate, tray, food container, cup, mug",
        "raw ingredients, vegetables, fruits, meat, fish, uncooked",
        "pest, insect, fly, cockroach, bug, ant",
        "spoilage, mold, rotten food, decay, contamination",
        "knife, fork, spoon, cutlery, utensil, chopsticks",
        "human hand, human arm, fingers, person, body part",
        "plastic packaging, wrapper, bag, foil, plastic wrap",
    ]
]

for i, prompts in enumerate(prompts_sets):
    print(f"\n--- Test {i+1}: {prompts[:3]}... ---")
    model.set_classes(prompts, model.get_text_pe(prompts))
    results = model.predict(img, conf=0.1, verbose=False)
    for r in results:
        for box in r.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Detected: {prompts[cid]} (conf={conf:.2f})")
