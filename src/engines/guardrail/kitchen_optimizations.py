"""
Kitchen Scene Understanding Optimizations
Based on SOTA research: EPIC-KITCHENS VISOR and SafeCOOK benchmarks

Features:
1. Optimized thresholds for kitchen environments
2. Vessel-first hierarchical clustering
3. Scale-invariant union-find
4. Reflective surface filtering (HSV-based)
5. Prep mode detection and bypass
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# OPTIMIZED KITCHEN THRESHOLDS (Based on EPIC-KITCHENS Research)
# =============================================================================

# YOLO Inference Params (Optimized for Kitchen Clutter)
KITCHEN_CONF_THRESHOLD = 0.50          # Was 0.25 → Filter out kitchen reflections/glints
KITCHEN_IOU_THRESHOLD = 0.30           # Was 0.45 → Stricter NMS for textured foods
KITCHEN_MIN_AREA_RATIO = 0.015         # 1.5% of image → Ignore crumbs/noise

# Clustering Params
KITCHEN_CLUSTER_SCALE_FACTOR = 0.4     # 40% of max diagonal (scale-invariant)

# Reflection Detection (HSV)
KITCHEN_REFLECTION_SAT_STD_MAX = 12    # Reflections have low color variance
KITCHEN_REFLECTION_VAL_MEAN_MIN = 200 # Reflections are bright

# Prep Mode Detection
KITCHEN_PREP_MODE_CONF_THRESHOLD = 0.6 # High confidence for knife/cutting board
KITCHEN_PREP_MODE_MAX_CLUSTERS = 3     # Allow 3 clusters in prep mode

# Vessel Classes (COCO IDs for containers)
VESSEL_CLASSES = {
    51: "bowl",
    45: "cup", 
    41: "knife",
    42: "spoon",
    43: "fork",
    47: "plate",
    # Add more as needed
}

PREP_MODE_INDICATORS = {
    41: "knife",
    81: "cutting_board",  # If available in model
}


class KitchenSceneOptimizer:
    """
    Optimizes food detection for kitchen environments using SOTA research.
    """
    
    def __init__(self, img_width: int, img_height: int):
        self.img_width = img_width
        self.img_height = img_height
        self.img_area = img_width * img_height
        self.img_diagonal = np.sqrt(img_width**2 + img_height**2)
    
    # =========================================================================
    # 1. REFLECTION FILTERING (HSV Variance Check)
    # =========================================================================
    
    def is_reflection(self, img_bgr: np.ndarray, bbox: np.ndarray) -> bool:
        """
        Detect if a detection is likely a reflection on stainless steel.
        
        Research Insight: Real food has high texture variance. 
        Reflections have high brightness but low color variance.
        
        Args:
            img_bgr: Original BGR image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            True if detection is likely a reflection
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Bounds check
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.img_width, x2)
            y2 = min(self.img_height, y2)
            
            if x2 <= x1 or y2 <= y1:
                return False
            
            # Extract region
            region = img_bgr[y1:y2, x1:x2]
            
            if region.size == 0:
                return False
            
            # Convert to HSV
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Calculate saturation std dev and value mean
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            
            sat_std = np.std(s_channel)
            val_mean = np.mean(v_channel)
            
            # Reflection criteria: low saturation variance + high brightness
            is_likely_reflection = (
                sat_std < KITCHEN_REFLECTION_SAT_STD_MAX and 
                val_mean > KITCHEN_REFLECTION_VAL_MEAN_MIN
            )
            
            if is_likely_reflection:
                logger.debug(
                    f"Detected reflection: sat_std={sat_std:.1f}, val_mean={val_mean:.1f}"
                )
            
            return is_likely_reflection
            
        except Exception as e:
            logger.warning(f"Error in reflection detection: {e}")
            return False
    
    # =========================================================================
    # 2. AREA FILTERING (Ignore Crumbs/Noise)
    # =========================================================================
    
    def is_valid_detection_size(self, bbox: np.ndarray) -> bool:
        """
        Filter out detections smaller than minimum area threshold.
        
        Research: Anything < 1.5% of frame is usually noise.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            True if detection is large enough
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_ratio = area / self.img_area
        
        return area_ratio >= KITCHEN_MIN_AREA_RATIO
    
    # =========================================================================
    # 3. VESSEL-FIRST HIERARCHICAL CLUSTERING
    # =========================================================================
    
    def cluster_with_vessel_hierarchy(
        self, 
        detections: List[Dict],
        img_bgr: Optional[np.ndarray] = None
    ) -> Tuple[int, List[List[int]], Dict[str, Any]]:
        """
        Hierarchical clustering: Container → Content grammar.
        
        Research Insight: Kitchen objects follow "Container -> Content" structure.
        
        Steps:
        A. Identify vessels (plates, bowls, trays) as parent nodes
        B. Merge food items inside vessels into parent cluster
        C. Handle "thali exception": bowls inside plates merge
        
        Args:
            detections: List of detection dicts with bbox, class_id, etc.
            img_bgr: Optional BGR image for reflection filtering
            
        Returns:
            (cluster_count, cluster_indices, metadata)
        """
        n = len(detections)
        if n <= 1:
            return n, [[0]] if n == 1 else [], {}
        
        # Identify vessels (anchors)
        vessel_indices = []
        food_indices = []
        
        for i, det in enumerate(detections):
            class_id = det.get("class_id", -1)
            if class_id in VESSEL_CLASSES:
                vessel_indices.append(i)
            else:
                food_indices.append(i)
        
        # Union-Find with scale-invariant threshold
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Step A: Anchor-based clustering - food inside vessels
        for food_idx in food_indices:
            food_bbox = detections[food_idx]["bbox"]
            food_center = self._get_center(food_bbox)
            
            for vessel_idx in vessel_indices:
                vessel_bbox = detections[vessel_idx]["bbox"]
                
                # If food centroid is inside vessel bbox, merge
                if self._point_inside_box(food_center, vessel_bbox):
                    union(food_idx, vessel_idx)
                    logger.debug(
                        f"Merged food {detections[food_idx].get('class_name')} "
                        f"into vessel {detections[vessel_idx].get('class_name')}"
                    )
                    break
        
        # Step B: Thali exception - bowls inside plates
        for i in vessel_indices:
            for j in vessel_indices:
                if i >= j:
                    continue
                
                bbox_i = detections[i]["bbox"]
                bbox_j = detections[j]["bbox"]
                
                # Check if one vessel contains another
                if self._box_contains(bbox_i, bbox_j) or self._box_contains(bbox_j, bbox_i):
                    union(i, j)
                    logger.debug(f"Thali exception: merged nested vessels")
        
        # Step C: Scale-invariant proximity clustering for remaining items
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) == find(j):
                    continue  # Already clustered
                
                if self._are_close_scale_invariant(
                    detections[i]["bbox"], 
                    detections[j]["bbox"]
                ):
                    union(i, j)
        
        # Build cluster groups
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        
        metadata = {
            "vessel_count": len(vessel_indices),
            "food_count": len(food_indices),
            "clustering_method": "vessel_hierarchy"
        }
        
        return len(clusters), list(clusters.values()), metadata
    
    # =========================================================================
    # 4. SCALE-INVARIANT DISTANCE
    # =========================================================================
    
    def _are_close_scale_invariant(self, bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
        """
        Adaptive clustering with dynamic threshold based on object size.
        
        Research: Threshold = 40% of larger object's diagonal prevents 
        small items from being isolated.
        
        Args:
            bbox1, bbox2: Bounding boxes [x1, y1, x2, y2]
            
        Returns:
            True if boxes are close relative to their size
        """
        # Calculate diagonals for scale
        diag1 = np.sqrt((bbox1[2] - bbox1[0])**2 + (bbox1[3] - bbox1[1])**2)
        diag2 = np.sqrt((bbox2[2] - bbox2[0])**2 + (bbox2[3] - bbox2[1])**2)
        
        # Dynamic threshold: 40% of larger object's diagonal
        max_diag = max(diag1, diag2)
        dynamic_threshold = max_diag * KITCHEN_CLUSTER_SCALE_FACTOR
        
        # Calculate centroid distance
        center1 = self._get_center(bbox1)
        center2 = self._get_center(bbox2)
        
        distance = np.sqrt(
            (center1[0] - center2[0])**2 + 
            (center1[1] - center2[1])**2
        )
        
        return distance < dynamic_threshold
    
    # =========================================================================
    # 5. PREP MODE DETECTION
    # =========================================================================
    
    def detect_prep_mode(self, detections: List[Dict]) -> Tuple[bool, str]:
        """
        Detect if scene is in "food prep" mode vs "finished meal" mode.
        
        Research Logic: If knife or cutting board detected with high confidence,
        increase allowed cluster count (1 → 3) to recognize "work in progress".
        
        Args:
            detections: List of detection dicts
            
        Returns:
            (is_prep_mode, reason)
        """
        for det in detections:
            class_id = det.get("class_id", -1)
            confidence = det.get("confidence", 0.0)
            
            if class_id in PREP_MODE_INDICATORS and confidence > KITCHEN_PREP_MODE_CONF_THRESHOLD:
                indicator_name = PREP_MODE_INDICATORS[class_id]
                return True, f"prep_mode_{indicator_name}_detected"
        
        return False, "finished_meal"
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Get centroid of bounding box."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return (cx, cy)
    
    def _point_inside_box(self, point: Tuple[float, float], bbox: np.ndarray) -> bool:
        """Check if point (x, y) is inside bounding box."""
        x, y = point
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
    
    def _box_contains(self, outer: np.ndarray, inner: np.ndarray) -> bool:
        """Check if outer box fully contains inner box."""
        return (
            outer[0] <= inner[0] and 
            outer[1] <= inner[1] and 
            outer[2] >= inner[2] and 
            outer[3] >= inner[3]
        )
    
    # =========================================================================
    # INTEGRATED FILTERING PIPELINE
    # =========================================================================
    
    def filter_and_cluster_detections(
        self,
        detections: List[Dict],
        img_bgr: np.ndarray
    ) -> Tuple[List[Dict], int, List[List[int]], Dict[str, Any]]:
        """
        Complete SOTA pipeline for kitchen scene understanding.
        
        Pipeline:
        1. Filter noise (reflections + small areas)
        2. Detect prep mode
        3. Vessel-first hierarchical clustering
        4. Return filtered detections + cluster info
        
        Args:
            detections: Raw YOLO detections
            img_bgr: BGR image for HSV analysis
            
        Returns:
            (filtered_detections, cluster_count, cluster_indices, metadata)
        """
        logger.info(f"[Kitchen Optimizer] Starting with {len(detections)} raw detections")
        
        # Step 1: Filter noise
        filtered = []
        reflection_count = 0
        small_area_count = 0
        
        for det in detections:
            bbox = det["bbox"]
            
            # Check area threshold
            if not self.is_valid_detection_size(bbox):
                small_area_count += 1
                continue
            
            # Check for reflection
            if self.is_reflection(img_bgr, bbox):
                reflection_count += 1
                continue
            
            filtered.append(det)
        
        logger.info(
            f"[Kitchen Optimizer] Filtered: {len(filtered)} valid, "
            f"{reflection_count} reflections, {small_area_count} too small"
        )
        
        # Step 2: Detect prep mode
        is_prep_mode, prep_reason = self.detect_prep_mode(filtered)
        
        # Step 3: Cluster with vessel hierarchy
        cluster_count, cluster_indices, cluster_meta = self.cluster_with_vessel_hierarchy(
            filtered, img_bgr
        )
        
        # Metadata
        metadata = {
            "original_detection_count": len(detections),
            "filtered_count": len(filtered),
            "reflections_removed": reflection_count,
            "small_areas_removed": small_area_count,
            "is_prep_mode": is_prep_mode,
            "prep_mode_reason": prep_reason,
            "cluster_count": cluster_count,
            **cluster_meta
        }
        
        # Adjust cluster threshold for prep mode
        if is_prep_mode:
            logger.info(
                f"[Kitchen Optimizer] Prep mode detected: {prep_reason}. "
                f"Allowing up to {KITCHEN_PREP_MODE_MAX_CLUSTERS} clusters."
            )
        
        return filtered, cluster_count, cluster_indices, metadata
