# Iteration 1 Report

## Overall Performance
- **Overall Accuracy**: 91.7%
- **Matches**: 11/12

## Parameter-Specific Accuracy

| Parameter | Accuracy | Precision | Recall | TP | TN | FP | FN |
|-----------|----------|-----------|--------|----|----|----|----|
| multiple_food_items | 58.3% | 44.4% | 100.0% | 4 | 3 | 5 | 0 |
| darkness | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| glare | 83.3% | 100.0% | 0.0% | 0 | 10 | 0 | 2 |
| angle | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| foreign_objects | 66.7% | 75.0% | 50.0% | 3 | 5 | 1 | 3 |
| food_score | 66.7% | 0.0% | 100.0% | 0 | 8 | 4 | 0 |

## Sample Mismatches

- **Native-Foods-3.jpg**: GT=PASS, GD=BLOCK
  - Reasons: ['Geometry: Multiple distinct dishes detected (2)']
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 169.876340230306, "glare_percentage": 0.16197586059570312, "blur_variance": 558.9201757472774, "combined_blur_score": 0.2784137703399783, "contrast_std_dev": 79.50972920260479, "dynamic_range": 242.0, "cluster_count": 2.0, "food_score": 0.27595213055610657, "foreign_object_score": 0.0, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 0.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 1.0, "beverage, drink, coffee, tea, soda, juice, water": 1.0, "food, meal, dish": 0.0, "bowl, plate, tray, food container, cup, mug": 1.0, "raw ingredients, vegetables, fruits, meat, fish": 0.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 0.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
