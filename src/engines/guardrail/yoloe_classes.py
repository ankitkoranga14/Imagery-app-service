"""
YOLOE-26n Open-Vocabulary Classes for Food Guardrail

This module defines the semantic classes for the unified vision stage
that replaces both L3 (YOLOv11n) and L4 (MobileCLIP2).

V3.0 Upgrades:
- Enhanced semantic class descriptions for better zero-shot detection
- Zero-tolerance foreign object detection thresholds
- Vessel-first hierarchy support
"""

from typing import Dict, List, Set, Tuple
from enum import Enum


class GuardrailClassCategory(str, Enum):
    """Categories for guardrail semantic classes."""
    MEAL = "meal"
    VESSEL = "vessel"
    RAW = "raw"
    SAFETY = "safety"
    UTENSIL = "utensil"
    HUMAN = "human"
    PACKAGING = "packaging"


# =============================================================================
# V3.0 ZERO-TOLERANCE FOREIGN OBJECT DETECTION
# =============================================================================

# Zero-tolerance threshold - any detection above this triggers BLOCK
V3_FOREIGN_OBJECT_THRESHOLD = 0.42

# Zero-tolerance foreign object classes (immediate BLOCK if detected)
V3_FOREIGN_OBJECT_CLASSES: Set[str] = {
    'human hand',
    'human arm',
    'human hands/arms',
    'plastic packaging',
    'pest',
    'insect',
    'knife',
    'person',
    'wrapper',
    'bag',
    'foil',
}

# =============================================================================
# V3.0 PHYSICS THRESHOLDS
# =============================================================================

# L0: Darkness (LAB L-Channel) - Target: 99.2% Precision
V3_DARKNESS_LAB_L_THRESHOLD = 62  # L_mean > 62
V3_DARKNESS_HARD_LIMIT = 25       # Absolute minimum

# L1: Glare (HSV Saturation/Value) - Target: 98.1% Recall
V3_GLARE_VALUE_STD_THRESHOLD = 82       # Std(V) > 82
V3_GLARE_SATURATION_HIGH_RATIO = 0.07   # Saturation ratio (>245) > 7%

# L2.5: Angle Detection (Hough Transform) - Target: 96.8% Accuracy
V3_ANGLE_MAX_DEVIATION = 38  # Relaxed from 28 to 38

# =============================================================================
# V3.0 VISION LOGIC THRESHOLDS
# =============================================================================

# Vessel-Based Clustering
V3_VESSEL_ROI_BUFFER = 6          # 6px buffer around vessel bbox
V3_VESSEL_FOOD_IOU_THRESHOLD = 0.25  # Food must have IoU > 0.25 with vessel
V3_DENSITY_THRESHOLD = 0.32       # Complex scene threshold

# Inference Settings
V3_INFERENCE_CONF = 0.38
V3_INFERENCE_IMGSZ = 416


# =============================================================================
# YOLOE-26n GUARDRAIL SEMANTIC CLASSES
# =============================================================================

GUARDRAIL_CLASSES: List[str] = [
    "plated meal, cooked food, rice, pasta, salad, soup, burger, pizza, sandwich",  # 0: MEAL
    "snack, dessert, cookie, cake, pastry, muffin, donut",                           # 1: SNACK
    "beverage, drink, coffee, tea, juice, water, smoothie",                          # 2: BEVERAGE
    "food, meal, dish, generic food item",                                           # 3: GENERIC_FOOD
    "bowl, plate, tray, food container, cup, mug",                                   # 4: VESSEL
    "raw ingredients, vegetables, fruits, meat, fish, uncooked",                     # 5: RAW
    "pest, insect, fly, cockroach, bug, ant",                                        # 6: PEST
    "spoilage, mold, rotten food, decay, contamination",                             # 7: SPOILAGE
    "knife, fork, spoon, cutlery, utensil, chopsticks",                              # 8: UTENSIL
    "human hand, human arm, fingers, person, body part",                             # 9: HUMAN
    "plastic packaging, wrapper, bag, foil, plastic wrap",                           # 10: PACKAGING
]

# Map class names to indices for fast lookup
GUARDRAIL_CLASS_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(GUARDRAIL_CLASSES)
}

# Category mappings for semantic reasoning
CLASS_CATEGORIES: Dict[str, GuardrailClassCategory] = {
    GUARDRAIL_CLASSES[0]: GuardrailClassCategory.MEAL,
    GUARDRAIL_CLASSES[1]: GuardrailClassCategory.MEAL,
    GUARDRAIL_CLASSES[2]: GuardrailClassCategory.MEAL,
    GUARDRAIL_CLASSES[3]: GuardrailClassCategory.MEAL,
    GUARDRAIL_CLASSES[4]: GuardrailClassCategory.VESSEL,
    GUARDRAIL_CLASSES[5]: GuardrailClassCategory.RAW,
    GUARDRAIL_CLASSES[6]: GuardrailClassCategory.SAFETY,
    GUARDRAIL_CLASSES[7]: GuardrailClassCategory.SAFETY,
    GUARDRAIL_CLASSES[8]: GuardrailClassCategory.UTENSIL,
    GUARDRAIL_CLASSES[9]: GuardrailClassCategory.HUMAN,
    GUARDRAIL_CLASSES[10]: GuardrailClassCategory.PACKAGING,
}


# =============================================================================
# SAFETY BLOCKING THRESHOLDS
# =============================================================================

SAFETY_BLOCK_THRESHOLD = 0.45
DEFAULT_CONF_THRESHOLD = 0.05


# =============================================================================
# CONTEXTUAL CLASSIFICATION RULES
# =============================================================================

class ContextualState(str, Enum):
    """Contextual states for food safety classification."""
    READY_TO_EAT = "ready_to_eat"
    PREP_MODE = "prep_mode"
    NOT_READY = "not_ready_to_eat"
    PACKAGED = "packaged_food"
    SAFETY_RISK = "safety_risk"
    UNKNOWN = "unknown"


def classify_contextual_state(
    class_counts: Dict[str, int],
    class_confidences: Dict[str, float]
) -> tuple[ContextualState, str]:
    """
    Classify the contextual state based on YOLOE-26n detections.
    """
    plated_meal_count = class_counts.get(GUARDRAIL_CLASSES[0], 0)
    snack_count = class_counts.get(GUARDRAIL_CLASSES[1], 0)
    beverage_count = class_counts.get(GUARDRAIL_CLASSES[2], 0)
    generic_food_count = class_counts.get(GUARDRAIL_CLASSES[3], 0)
    raw_ingredient_count = class_counts.get(GUARDRAIL_CLASSES[5], 0)
    vessel_count = class_counts.get(GUARDRAIL_CLASSES[4], 0)
    pest_count = class_counts.get(GUARDRAIL_CLASSES[6], 0)
    spoilage_count = class_counts.get(GUARDRAIL_CLASSES[7], 0)
    knife_count = class_counts.get(GUARDRAIL_CLASSES[8], 0)
    packaging_count = class_counts.get(GUARDRAIL_CLASSES[10], 0)
    
    # Get max confidences for safety classes
    pest_conf = class_confidences.get(GUARDRAIL_CLASSES[6], 0.0)
    spoilage_conf = class_confidences.get(GUARDRAIL_CLASSES[7], 0.0)
    
    # Rule 1: Safety Risk - High priority
    if pest_count > 0 and pest_conf > SAFETY_BLOCK_THRESHOLD:
        return ContextualState.SAFETY_RISK, f"pest_insect detected (conf={pest_conf:.2f})"
    
    if spoilage_count > 0 and spoilage_conf > SAFETY_BLOCK_THRESHOLD:
        return ContextualState.SAFETY_RISK, f"spoilage_mold detected (conf={spoilage_conf:.2f})"
    
    # Rule 2: Prep Mode - Knife near raw ingredients
    if knife_count > 0 and raw_ingredient_count > 0:
        return ContextualState.PREP_MODE, "cutlery_knife + raw_ingredient = prep_mode"
    
    # Rule 3: Not Ready to Eat - Raw without plated meal
    if raw_ingredient_count > 0 and plated_meal_count == 0 and snack_count == 0 and generic_food_count == 0:
        return ContextualState.NOT_READY, "raw_ingredient present, no plated_meal/snack"
    
    # Rule 4: Packaged Food
    if packaging_count > 0 and plated_meal_count == 0 and snack_count == 0 and generic_food_count == 0:
        return ContextualState.PACKAGED, "packaging_plastic without plated_meal/snack"
    
    # Rule 5: Ready to Eat - Plated meal, snack, beverage, or generic food
    ready_to_eat_count = plated_meal_count + snack_count + beverage_count + generic_food_count
    if ready_to_eat_count > 0:
        if vessel_count > 0:
            return ContextualState.READY_TO_EAT, "food/drink within vessel_bowl_plate cluster"
        else:
            return ContextualState.READY_TO_EAT, "food/drink detected"
    
    # Default: Unknown state
    return ContextualState.UNKNOWN, "no clear food pattern detected"


# =============================================================================
# VESSEL-FIRST HIERARCHY FOR YOLOE
# =============================================================================

YOLOE_VESSEL_KEYWORDS: Set[str] = {
    GUARDRAIL_CLASSES[4],
}

YOLOE_FOOD_KEYWORDS: Set[str] = {
    GUARDRAIL_CLASSES[0],
    GUARDRAIL_CLASSES[1],
    GUARDRAIL_CLASSES[2],
    GUARDRAIL_CLASSES[3],
    GUARDRAIL_CLASSES[5],
}

YOLOE_SAFETY_KEYWORDS: Set[str] = {
    GUARDRAIL_CLASSES[6],
    GUARDRAIL_CLASSES[7],
}

YOLOE_UTENSIL_KEYWORDS: Set[str] = {
    GUARDRAIL_CLASSES[8],
}


def is_vessel_class(class_name: str) -> bool:
    """Check if class name is a vessel/container."""
    return class_name in YOLOE_VESSEL_KEYWORDS


def is_food_class(class_name: str) -> bool:
    """Check if class name is a food item."""
    return class_name in YOLOE_FOOD_KEYWORDS


def is_safety_class(class_name: str) -> bool:
    """Check if class name is a safety concern."""
    return class_name in YOLOE_SAFETY_KEYWORDS


def is_prep_indicator(class_name: str) -> bool:
    """Check if class indicates prep mode."""
    return class_name in YOLOE_UTENSIL_KEYWORDS
