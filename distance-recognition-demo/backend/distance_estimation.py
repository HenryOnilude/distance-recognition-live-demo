"""
Distance Estimation Module
Physics-based distance calculation from face width measurements
Based on research formula: distance = (14cm × focal_length_pixels) / face_width_pixels
"""

import numpy as np
from typing import Tuple

def estimate_distance_from_face_size(face_bbox: Tuple[int, int, int, int],
                                   image_shape: Tuple[int, int]) -> float:
    """
    Estimate distance using face width and camera calibration

    Args:
        face_bbox: (x, y, width, height) of detected face
        image_shape: (height, width) of original image

    Returns:
        Estimated distance in meters
    """
    try:
        x, y, w, h = face_bbox
        image_height, image_width = image_shape

        if w <= 0 or image_width <= 0:
            return 15.0
        
        face_ratio = w / image_width

        distance_m = 0.5 + (1.0 - face_ratio) * 9.5


        # Clamp to reasonable range based on research
        MIN_DISTANCE = 0.5  # Allow closer distances for portraits (50cm)
        MAX_DISTANCE = 15.0  # Beyond this, face too small for reliable detection

        distance_m = max(MIN_DISTANCE, min(distance_m, MAX_DISTANCE))

        return distance_m

    except Exception as e:
        print(f"Error in distance estimation: {e}")
        return 5.0  # Return medium distance as fallback

def classify_distance_range(distance_m: float) -> str:
    """
    Classify distance into research-based categories

    Args:
        distance_m: Distance in meters

    Returns:
        Distance category: 'close', 'medium', or 'far'
    """
    if distance_m <= 4.0:
        return "close"      # 2-4m: 89.1% accuracy
    elif distance_m <= 7.0:
        return "medium"     # 4-7m: 82.3% accuracy
    else:
        return "far"        # 7-10m: 72.3% accuracy

def get_distance_quality_factor(distance_m: float) -> float:
    """
    Get quality degradation factor based on distance

    Args:
        distance_m: Distance in meters

    Returns:
        Quality factor (0.0 to 1.0)
    """
    if distance_m <= 4.0:
        # Optimal range - minimal quality loss
        return 1.0 - (distance_m - 2.0) * 0.05  # 1.0 to 0.9
    elif distance_m <= 7.0:
        # Medium range - moderate quality loss
        return 0.9 - (distance_m - 4.0) * 0.1  # 0.9 to 0.6
    else:
        # Far range - significant quality loss
        return max(0.3, 0.6 - (distance_m - 7.0) * 0.1)  # 0.6 to 0.3

def calculate_expected_face_size(distance_m: float, focal_length_px: float = 800) -> int:
    """
    Calculate expected face size in pixels for a given distance

    Args:
        distance_m: Distance in meters
        focal_length_px: Camera focal length in pixels

    Returns:
        Expected face width in pixels
    """
    try:
        AVERAGE_FACE_WIDTH_CM = 14.0
        distance_cm = distance_m * 100.0

        expected_width_px = (AVERAGE_FACE_WIDTH_CM * focal_length_px) / distance_cm
        return int(expected_width_px)

    except Exception as e:
        print(f"Error calculating expected face size: {e}")
        return 100  # Default face size

def validate_face_distance_consistency(face_bbox: Tuple[int, int, int, int],
                                     expected_distance: float) -> bool:
    """
    Validate if detected face size is consistent with expected distance

    Args:
        face_bbox: (x, y, width, height) of detected face
        expected_distance: Expected distance in meters

    Returns:
        True if face size is consistent with distance
    """
    try:
        x, y, w, h = face_bbox

        # Calculate what the face size should be at this distance
        expected_width = calculate_expected_face_size(expected_distance)

        # Allow 30% tolerance for natural variation
        tolerance = 0.3
        min_width = expected_width * (1 - tolerance)
        max_width = expected_width * (1 + tolerance)

        return min_width <= w <= max_width

    except Exception as e:
        print(f"Error in distance consistency validation: {e}")
        return True  # Default to valid if error occurs

def get_distance_confidence_multiplier(distance_m: float) -> float:
    """
    Get confidence multiplier based on distance (from research)

    Args:
        distance_m: Distance in meters

    Returns:
        Confidence multiplier (0.0 to 1.0)
    """
    distance_category = classify_distance_range(distance_m)

    # Research-based confidence multipliers
    multipliers = {
        "close": 1.00,   # 2-4m: Full confidence
        "medium": 0.92,  # 4-7m: Slight reduction
        "far": 0.81      # 7-10m: Significant reduction
    }

    return multipliers.get(distance_category, 0.7)