# Iteration 6 Report

## Overall Performance
- **Overall Accuracy**: 75.0%
- **Matches**: 9/12

## Parameter-Specific Accuracy

| Parameter | Accuracy | Precision | Recall | TP | TN | FP | FN |
|-----------|----------|-----------|--------|----|----|----|----|
| multiple_food_items | 75.0% | 66.7% | 50.0% | 2 | 7 | 1 | 2 |
| darkness | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| glare | 83.3% | 100.0% | 0.0% | 0 | 10 | 0 | 2 |
| angle | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| foreign_objects | 58.3% | 100.0% | 16.7% | 1 | 6 | 0 | 5 |
| food_score | 58.3% | 0.0% | 100.0% | 0 | 7 | 5 | 0 |

## Sample Mismatches

- **burger.jpeg**: GT=BLOCK, GD=PASS
  - Reasons: []
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 95.41320066666667, "glare_percentage": 0.0015213333333333333, "blur_variance": 788.9110942929631, "combined_blur_score": 0.10892424157266374, "contrast_std_dev": 56.34794594785509, "dynamic_range": 175.0, "cluster_count": 5.0, "food_score": 0.5677117109298706, "foreign_object_score": 0.03908828645944595, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 1.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 1.0, "beverage, drink, coffee, tea, soda, juice, water": 0.0, "food, meal, dish": 2.0, "bowl, plate, tray, food container, cup, mug": 3.0, "raw ingredients, vegetables, fruits, meat, fish": 0.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 1.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
- **170203151709-jambalaya.jpg**: GT=PASS, GD=BLOCK
  - Reasons: ['Context: no clear food pattern detected']
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 147.16137170087228, "glare_percentage": 0.0017465819962982278, "blur_variance": 222.61437464620454, "combined_blur_score": 0.5592451878940944, "contrast_std_dev": 46.704917534854964, "dynamic_range": 136.0, "cluster_count": 1.0, "food_score": 0.0, "foreign_object_score": 0.0, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 0.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 0.0, "beverage, drink, coffee, tea, soda, juice, water": 0.0, "food, meal, dish": 0.0, "bowl, plate, tray, food container, cup, mug": 0.0, "raw ingredients, vegetables, fruits, meat, fish": 0.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 0.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
- **unnamed.jpg**: GT=BLOCK, GD=PASS
  - Reasons: []
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 72.35273539936625, "glare_percentage": 0.08442485249125874, "blur_variance": 3123.757596920195, "combined_blur_score": 0.0, "contrast_std_dev": 80.7185220708208, "dynamic_range": 253.0, "cluster_count": 2.0, "food_score": 0.016694948077201843, "foreign_object_score": 0.0, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 0.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 0.0, "beverage, drink, coffee, tea, soda, juice, water": 0.0, "food, meal, dish": 1.0, "bowl, plate, tray, food container, cup, mug": 2.0, "raw ingredients, vegetables, fruits, meat, fish": 0.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 0.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
