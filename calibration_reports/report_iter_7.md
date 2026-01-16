# Iteration 7 Report

## Overall Performance
- **Overall Accuracy**: 83.3%
- **Matches**: 10/12

## Parameter-Specific Accuracy

| Parameter | Accuracy | Precision | Recall | TP | TN | FP | FN |
|-----------|----------|-----------|--------|----|----|----|----|
| multiple_food_items | 83.3% | 75.0% | 75.0% | 3 | 7 | 1 | 1 |
| darkness | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| glare | 83.3% | 100.0% | 0.0% | 0 | 10 | 0 | 2 |
| angle | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| foreign_objects | 75.0% | 80.0% | 66.7% | 4 | 5 | 1 | 2 |
| food_score | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |

## Sample Mismatches

- **unnamed.jpg**: GT=BLOCK, GD=PASS
  - Reasons: []
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 72.35273539936625, "glare_percentage": 0.08442485249125874, "blur_variance": 3123.757596920195, "combined_blur_score": 0.0, "contrast_std_dev": 80.7185220708208, "dynamic_range": 253.0, "cluster_count": 1.0, "food_score": 0.016694948077201843, "foreign_object_score": 0.0, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 0.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 0.0, "beverage, drink, coffee, tea, soda, juice, water": 0.0, "food, meal, dish": 1.0, "bowl, plate, tray, food container, cup, mug": 2.0, "raw ingredients, vegetables, fruits, meat, fish": 0.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 0.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
- **salad-bowl-slices-tomato-spinach-600nw-2344224543.webp**: GT=BLOCK, GD=PASS
  - Reasons: []
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 137.97841666666667, "glare_percentage": 0.00013333333333333334, "blur_variance": 750.7734653282043, "combined_blur_score": 0.20583549210408741, "contrast_std_dev": 57.27545082749725, "dynamic_range": 155.0, "cluster_count": 3.0, "food_score": 0.0, "foreign_object_score": 0.01897241733968258, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 0.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 0.0, "beverage, drink, coffee, tea, soda, juice, water": 0.0, "food, meal, dish": 0.0, "bowl, plate, tray, food container, cup, mug": 2.0, "raw ingredients, vegetables, fruits, meat, fish": 2.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 1.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
