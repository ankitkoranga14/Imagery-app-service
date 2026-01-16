# Iteration 20 Report

## Overall Performance
- **Overall Accuracy**: 83.3%
- **Matches**: 10/12

## Parameter-Specific Accuracy

| Parameter | Accuracy | Precision | Recall | TP | TN | FP | FN |
|-----------|----------|-----------|--------|----|----|----|----|
| multiple_food_items | 75.0% | 66.7% | 50.0% | 2 | 7 | 1 | 2 |
| darkness | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| glare | 83.3% | 100.0% | 0.0% | 0 | 10 | 0 | 2 |
| angle | 100.0% | 100.0% | 100.0% | 0 | 12 | 0 | 0 |
| foreign_objects | 50.0% | 50.0% | 83.3% | 5 | 1 | 5 | 1 |

## Sample Mismatches

- **170203151709-jambalaya.jpg**: GT=PASS, GD=BLOCK
  - Reasons: ['Context: no clear food pattern detected']
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 147.16137170087228, "glare_percentage": 0.0017465819962982278, "blur_variance": 222.61437464620454, "combined_blur_score": 0.5592451878940944, "contrast_std_dev": 46.704917534854964, "dynamic_range": 136.0, "cluster_count": 1.0, "food_score": 0.0, "foreign_object_score": 0.0, "angle_score": 0.0, "a plated meal, ready to eat food, cooked dish, gourmet meal, restaurant food, rice dish, pasta, salad, soup, burger, pizza, taco, sandwich": 0.0, "a snack, dessert, cookie, cake, pastry, sweet treat, muffin, donut": 0.0, "a beverage, drink, coffee, tea, soda, juice, smoothie, cocktail, water": 0.0, "food, delicious food, meal, edible item, dish, appetizer, main course": 0.0, "a bowl, plate, tray, food container, dishware, serving platter, cup, mug": 0.0, "raw ingredients, uncooked food, vegetables, fruits, produce, meat, fish": 0.0, "a pest, insect, fly, cockroach, ant, bug, infestation": 0.0, "spoilage, mold, rotten food, decay, expired food, fuzzy growth": 0.0, "a knife, fork, spoon, cutlery, silverware, utensil, chopsticks": 0.0, "a human hand, arm, fingers, person holding food": 0.0, "plastic packaging, wrapper, bag, container lid, foil, cling wrap": 0.0}
- **istockphoto-542180758-612x612.jpg**: GT=PASS, GD=BLOCK
  - Reasons: ['Context: no clear food pattern detected']
  - Scores: {"injection_score": 0.0, "semantic_similarity": 0.002888282760977745, "policy_score": 0.0, "food_domain_score": 0.5928181409835815, "darkness_score": 137.99509403434575, "glare_percentage": 0.1628500256311675, "blur_variance": 579.3563651609587, "combined_blur_score": 0.2980175096923323, "contrast_std_dev": 97.89114355221159, "dynamic_range": 243.0, "cluster_count": 1.0, "food_score": 0.0, "foreign_object_score": 0.0, "angle_score": 0.0, "a plated meal, ready to eat food, cooked dish, gourmet meal, restaurant food, rice dish, pasta, salad, soup, burger, pizza, taco, sandwich": 0.0, "a snack, dessert, cookie, cake, pastry, sweet treat, muffin, donut": 0.0, "a beverage, drink, coffee, tea, soda, juice, smoothie, cocktail, water": 0.0, "food, delicious food, meal, edible item, dish, appetizer, main course": 0.0, "a bowl, plate, tray, food container, dishware, serving platter, cup, mug": 0.0, "raw ingredients, uncooked food, vegetables, fruits, produce, meat, fish": 0.0, "a pest, insect, fly, cockroach, ant, bug, infestation": 0.0, "spoilage, mold, rotten food, decay, expired food, fuzzy growth": 0.0, "a knife, fork, spoon, cutlery, silverware, utensil, chopsticks": 0.0, "a human hand, arm, fingers, person holding food": 0.0, "plastic packaging, wrapper, bag, container lid, foil, cling wrap": 0.0}
