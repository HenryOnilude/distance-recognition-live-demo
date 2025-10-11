"""
Distance-Adaptive CLAHE Preprocessing for Face Recognition
Adjusts contrast enhancement based on distance to optimize recognition accuracy
Research shows different distance ranges require different preprocessing strategies
"""

import cv2
import numpy as np
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistanceAdaptiveCLAHE:
    """
    Distance-adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization)

    Research insights:
    - Close distances (0.5-1m): Minimal processing needed, high detail preservation
    - Medium distances (1-4m): Moderate enhancement, balance detail vs noise
    - Far distances (4-10m): Aggressive enhancement, noise acceptable for visibility
    """

    def __init__(self):
        # CLAHE parameters for different distance ranges
        self.distance_configs = {
            'portrait': {  # 0.5-1.0m
                'clip_limit': 1.0,
                'tile_grid_size': (8, 8),
                'description': 'Minimal enhancement for high-quality close shots'
            },
            'close': {  # 1.0-4.0m
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8),
                'description': 'Balanced enhancement for standard distances'
            },
            'medium': {  # 4.0-7.0m
                'clip_limit': 3.0,
                'tile_grid_size': (12, 12),
                'description': 'Moderate enhancement for medium distances'
            },
            'far': {  # 7.0-10.0m
                'clip_limit': 4.0,
                'tile_grid_size': (16, 16),
                'description': 'Aggressive enhancement for far distances'
            }
        }

        logger.info("✅ Distance-adaptive CLAHE preprocessor initialized")

    def classify_distance_category(self, distance_m: float) -> str:
        """Classify distance into preprocessing categories"""
        if distance_m <= 1.0:
            return 'portrait'
        elif distance_m <= 4.0:
            return 'close'
        elif distance_m <= 7.0:
            return 'medium'
        else:
            return 'far'

    def apply_distance_adaptive_clahe(self, image: np.ndarray, distance_m: float) -> np.ndarray:
        """
        Apply distance-adaptive CLAHE preprocessing

        Args:
            image: Input image (BGR or RGB)
            distance_m: Estimated distance in meters

        Returns:
            Enhanced image with distance-appropriate CLAHE
        """
        # Determine distance category
        category = self.classify_distance_category(distance_m)
        config = self.distance_configs[category]

        logger.debug(f"Distance {distance_m:.1f}m → {category} → {config['description']}")

        # Convert to LAB color space for better CLAHE results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Create CLAHE object with distance-specific parameters
        clahe = cv2.createCLAHE(
            clipLimit=config['clip_limit'],
            tileGridSize=config['tile_grid_size']
        )

        # Apply CLAHE to L channel only
        l_channel_clahe = clahe.apply(l_channel)

        # Merge channels back
        lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])

        # Convert back to BGR
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return result

    def apply_distance_adaptive_preprocessing(self, image: np.ndarray, distance_m: float, quality_score: float = 0.5) -> np.ndarray:
        """
        Complete distance-adaptive preprocessing pipeline

        Args:
            image: Input image
            distance_m: Estimated distance in meters
            quality_score: Quality score (0-1) for adaptive processing

        Returns:
            Preprocessed image optimized for the distance
        """
        # Step 1: Distance-adaptive CLAHE
        enhanced = self.apply_distance_adaptive_clahe(image, distance_m)

        # Step 2: Quality-based additional processing
        if quality_score < 0.4:  # Low quality images need more processing
            # Apply additional noise reduction for poor quality
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
            logger.debug("Applied additional noise reduction for low quality image")

        # Step 3: Distance-specific sharpening
        category = self.classify_distance_category(distance_m)

        if category in ['medium', 'far']:
            # Apply unsharp masking for medium/far distances
            gaussian = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            logger.debug(f"Applied sharpening for {category} distance")

        # Step 4: Brightness normalization for far distances
        if category == 'far':
            # Normalize brightness to improve visibility at far distances
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)

            if mean_brightness < 100:  # Dark image
                brightness_boost = 1.2
                enhanced = cv2.convertScaleAbs(enhanced, alpha=brightness_boost, beta=10)
                logger.debug("Applied brightness boost for dark far-distance image")

        return enhanced

    def get_preprocessing_info(self, distance_m: float) -> Dict:
        """Get information about preprocessing applied for a given distance"""
        category = self.classify_distance_category(distance_m)
        config = self.distance_configs[category]

        return {
            'distance_m': distance_m,
            'category': category,
            'clip_limit': config['clip_limit'],
            'tile_grid_size': config['tile_grid_size'],
            'description': config['description']
        }

    def compare_preprocessing_modes(self, image: np.ndarray, distance_m: float) -> Dict[str, np.ndarray]:
        """
        Compare different preprocessing modes for analysis/debugging

        Returns:
            Dictionary with original and processed versions
        """
        original = image.copy()
        adaptive = self.apply_distance_adaptive_preprocessing(image, distance_m)

        # Standard CLAHE for comparison
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_standard = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_standard = clahe_standard.apply(l)
        lab_standard = cv2.merge([l_standard, a, b])
        standard_clahe = cv2.cvtColor(lab_standard, cv2.COLOR_LAB2BGR)

        return {
            'original': original,
            'standard_clahe': standard_clahe,
            'adaptive_clahe': adaptive
        }


# Global preprocessor instance
adaptive_preprocessor = DistanceAdaptiveCLAHE()

def preprocess_for_distance(image: np.ndarray, distance_m: float, quality_score: float = 0.5) -> np.ndarray:
    """
    Convenience function for distance-adaptive preprocessing

    Args:
        image: Input image
        distance_m: Estimated distance in meters
        quality_score: Quality score (0-1)

    Returns:
        Preprocessed image optimized for recognition at the given distance
    """
    return adaptive_preprocessor.apply_distance_adaptive_preprocessing(image, distance_m, quality_score)

def get_preprocessing_info(distance_m: float) -> Dict:
    """Get preprocessing parameters for a given distance"""
    return adaptive_preprocessor.get_preprocessing_info(distance_m)


# ===================
# USAGE EXAMPLE
# ===================
if __name__ == "__main__":
    # Test the adaptive preprocessing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("Testing distance-adaptive CLAHE preprocessing...")

    # Test different distances
    test_distances = [0.7, 2.5, 5.5, 8.5]
    categories = ['portrait', 'close', 'medium', 'far']

    for dist, cat in zip(test_distances, categories):
        processed = preprocess_for_distance(test_image, dist, quality_score=0.6)
        info = get_preprocessing_info(dist)
        print(f"Distance {dist}m ({cat}): clip_limit={info['clip_limit']}, grid={info['tile_grid_size']}")

    print("✅ Distance-adaptive CLAHE preprocessing working!")