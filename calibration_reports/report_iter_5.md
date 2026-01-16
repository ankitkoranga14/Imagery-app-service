# Iteration 5 Report

## Overall Performance
- **Overall Accuracy**: 66.7%
- **Matches**: 8/12

## Parameter-Specific Accuracy

| Parameter | Accuracy | Precision | Recall | TP | TN | FP | FN |
|-----------|----------|-----------|--------|----|----|----|----|
| multiple_food_items | 75.0% | 60.0% | 75.0% | 3 | 6 | 2 | 1 |
| darkness | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| glare | 83.3% | 100.0% | 0.0% | 0 | 10 | 0 | 2 |
| angle | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| foreign_objects | 75.0% | 100.0% | 50.0% | 3 | 6 | 0 | 3 |
| food_score | 41.7% | 0.0% | 100.0% | 0 | 5 | 7 | 0 |

## Sample Mismatches

- **170203151709-jambalaya.jpg**: GT=PASS, GD=BLOCK
  - Reasons: ['Context: no clear food pattern detected']
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 147.16137170087228, "glare_percentage": 0.0017465819962982278, "blur_variance": 222.61437464620454, "combined_blur_score": 0.5592451878940944, "contrast_std_dev": 46.704917534854964, "dynamic_range": 136.0, "cluster_count": 1.0, "food_score": 0.0, "foreign_object_score": 0.0, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 0.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 0.0, "beverage, drink, coffee, tea, soda, juice, water": 0.0, "food, meal, dish": 0.0, "bowl, plate, tray, food container, cup, mug": 0.0, "raw ingredients, vegetables, fruits, meat, fish": 0.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 0.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
- **unnamed.jpg**: GT=BLOCK, GD=PASS
  - Reasons: []
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 72.35273539936625, "glare_percentage": 0.08442485249125874, "blur_variance": 3123.757596920195, "combined_blur_score": 0.0, "contrast_std_dev": 80.7185220708208, "dynamic_range": 253.0, "cluster_count": 2.0, "food_score": 0.09695834666490555, "foreign_object_score": 0.0, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 2.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 0.0, "beverage, drink, coffee, tea, soda, juice, water": 0.0, "food, meal, dish": 0.0, "bowl, plate, tray, food container, cup, mug": 0.0, "raw ingredients, vegetables, fruits, meat, fish": 0.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 0.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
- **Native-Foods-3.jpg**: GT=PASS, GD=BLOCK
  - Reasons: ['Geometry: Multiple distinct dishes detected (5)']
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 169.876340230306, "glare_percentage": 0.16197586059570312, "blur_variance": 558.9201757472774, "combined_blur_score": 0.2784137703399783, "contrast_std_dev": 79.50972920260479, "dynamic_range": 242.0, "cluster_count": 5.0, "food_score": 0.2759517431259155, "foreign_object_score": 0.035845573991537094, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 0.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 1.0, "beverage, drink, coffee, tea, soda, juice, water": 1.0, "food, meal, dish": 0.0, "bowl, plate, tray, food container, cup, mug": 4.0, "raw ingredients, vegetables, fruits, meat, fish": 0.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 1.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
- **istockphoto-542180758-612x612.jpg**: GT=PASS, GD=BLOCK
  - Reasons: ['Context: no clear food pattern detected']
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 137.99509403434575, "glare_percentage": 0.1628500256311675, "blur_variance": 579.3563651609587, "combined_blur_score": 0.2980175096923323, "contrast_std_dev": 97.89114355221159, "dynamic_range": 243.0, "cluster_count": 1.0, "food_score": 0.0, "foreign_object_score": 0.0, "angle_score": 0.0, "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, taco, sandwich, jambalaya": 0.0, "snack, dessert, cookie, cake, pastry, muffin, donut": 0.0, "beverage, drink, coffee, tea, soda, juice, water": 0.0, "food, meal, dish": 0.0, "bowl, plate, tray, food container, cup, mug": 0.0, "raw ingredients, vegetables, fruits, meat, fish": 0.0, "pest, insect, fly, cockroach, bug": 0.0, "spoilage, mold, rotten food, decay": 0.0, "knife, fork, spoon, cutlery, utensil, chopsticks": 0.0, "human hand, arm, fingers, person": 0.0, "plastic packaging, wrapper, bag, foil": 0.0}
