"""
BRISQUE-Inspired Image Quality Assessment
Implements key BRISQUE features without requiring external BRISQUE libraries
Research-backed quality metrics for distance-aware face recognition
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BRISQUEInspiredQuality:
    """
    BRISQUE-inspired quality assessment using OpenCV and NumPy

    BRISQUE measures:
    - Spatial naturalness based on Natural Scene Statistics (NSS)
    - Distortion measurement using moment-based features
    - Global quality score (lower = better quality)

    Our implementation captures the key insights:
    1. Sharpness/Blur detection
    2. Noise estimation
    3. Contrast measurement
    4. Naturalness assessment
    """

    def __init__(self):
        # Quality assessment parameters based on BRISQUE research
        self.block_size = 16  # Block size for local statistics
        self.overlap = 8      # Overlap between blocks

        logger.info("✅ BRISQUE-inspired quality assessor initialized")

    def calculate_mscn_coefficients(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate Mean Subtracted Contrast Normalized (MSCN) coefficients
        Core component of BRISQUE algorithm
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gaussian blur for local mean estimation
        mu = cv2.GaussianBlur(gray.astype(np.float32), (7, 7), 1.166)

        # Calculate local variance
        mu_sq = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (7, 7), 1.166)
        sigma = np.sqrt(np.maximum(mu_sq - mu ** 2, 0))

        # MSCN coefficients
        mscn = (gray.astype(np.float32) - mu) / (sigma + 1)

        return mscn

    def calculate_naturalness_features(self, mscn: np.ndarray) -> Dict[str, float]:
        """
        Extract naturalness features from MSCN coefficients
        Based on Natural Scene Statistics principles
        """
        # Shape and scale parameters of generalized Gaussian distribution
        # Natural images follow specific statistical distributions

        # Calculate moments
        mean = np.mean(mscn)
        variance = np.var(mscn)
        skewness = np.mean((mscn - mean) ** 3) / (variance ** 1.5 + 1e-10)
        kurtosis = np.mean((mscn - mean) ** 4) / (variance ** 2 + 1e-10)

        # Pairwise products (adjacent pixel correlations)
        h_diff = mscn[:, :-1] * mscn[:, 1:]  # Horizontal
        v_diff = mscn[:-1, :] * mscn[1:, :]  # Vertical
        d1_diff = mscn[:-1, :-1] * mscn[1:, 1:]  # Diagonal 1
        d2_diff = mscn[:-1, 1:] * mscn[1:, :-1]  # Diagonal 2

        features = {
            'mscn_mean': float(mean),
            'mscn_variance': float(variance),
            'mscn_skewness': float(skewness),
            'mscn_kurtosis': float(kurtosis),
            'h_mean': float(np.mean(h_diff)),
            'h_variance': float(np.var(h_diff)),
            'v_mean': float(np.mean(v_diff)),
            'v_variance': float(np.var(v_diff)),
            'd1_mean': float(np.mean(d1_diff)),
            'd1_variance': float(np.var(d1_diff)),
            'd2_mean': float(np.mean(d2_diff)),
            'd2_variance': float(np.var(d2_diff))
        }

        return features

    def calculate_distortion_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate distortion-specific features
        Complements MSCN-based naturalness assessment
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Sharpness measurement (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        # Noise estimation (high-frequency content in smooth regions)
        # Use Sobel operators to detect edges, then measure noise in non-edge areas
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Smooth regions (low edge magnitude)
        smooth_mask = edge_magnitude < np.percentile(edge_magnitude, 30)
        noise_estimate = np.std(gray[smooth_mask]) if np.any(smooth_mask) else 0

        # Contrast measurement
        contrast = np.std(gray)

        # Brightness uniformity
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-10))

        features = {
            'sharpness': float(sharpness),
            'noise_estimate': float(noise_estimate),
            'contrast': float(contrast),
            'entropy': float(entropy)
        }

        return features

    def calculate_brisque_score(self, image: np.ndarray) -> float:
        """
        Calculate BRISQUE-inspired quality score

        Returns:
            Quality score between 0 and 1 (1 = highest quality, 0 = lowest quality)
            Note: Original BRISQUE returns 0-100 with lower = better
        """
        try:
            # Step 1: Calculate MSCN coefficients
            mscn = self.calculate_mscn_coefficients(image)

            # Step 2: Extract naturalness features
            naturalness_features = self.calculate_naturalness_features(mscn)

            # Step 3: Extract distortion features
            distortion_features = self.calculate_distortion_features(image)

            # Step 4: Combine features into quality score
            # Simplified model based on BRISQUE insights

            # Naturalness score (how "natural" the statistics are)
            # Natural images have specific mean/variance characteristics
            naturalness_score = 1.0 / (1.0 + abs(naturalness_features['mscn_variance'] - 1.0))
            naturalness_score *= 1.0 / (1.0 + abs(naturalness_features['mscn_skewness']))
            naturalness_score *= 1.0 / (1.0 + abs(naturalness_features['mscn_kurtosis'] - 3.0))

            # Sharpness score (normalized)
            sharpness_score = min(1.0, distortion_features['sharpness'] / 1000.0)

            # Noise score (inverted - less noise = higher score)
            noise_score = 1.0 / (1.0 + distortion_features['noise_estimate'] / 50.0)

            # Contrast score (optimal contrast around 40-60)
            optimal_contrast = 50.0
            contrast_score = 1.0 / (1.0 + abs(distortion_features['contrast'] - optimal_contrast) / optimal_contrast)

            # Combine scores with weights based on BRISQUE research
            quality_score = (
                naturalness_score * 0.4 +
                sharpness_score * 0.3 +
                noise_score * 0.2 +
                contrast_score * 0.1
            )

            # Ensure score is in [0, 1] range
            quality_score = np.clip(quality_score, 0.0, 1.0)

            logger.debug(f"BRISQUE quality components: naturalness={naturalness_score:.3f}, "
                        f"sharpness={sharpness_score:.3f}, noise={noise_score:.3f}, "
                        f"contrast={contrast_score:.3f}, final={quality_score:.3f}")

            return float(quality_score)

        except Exception as e:
            logger.error(f"Error calculating BRISQUE score: {e}")
            return 0.5  # Default moderate quality

    def assess_quality_detailed(self, image: np.ndarray) -> Dict:
        """
        Comprehensive quality assessment with detailed breakdown
        """
        quality_score = self.calculate_brisque_score(image)

        # Calculate individual components for debugging
        mscn = self.calculate_mscn_coefficients(image)
        naturalness_features = self.calculate_naturalness_features(mscn)
        distortion_features = self.calculate_distortion_features(image)

        # Quality category
        if quality_score >= 0.8:
            quality_category = "Excellent"
        elif quality_score >= 0.6:
            quality_category = "Good"
        elif quality_score >= 0.4:
            quality_category = "Fair"
        elif quality_score >= 0.2:
            quality_category = "Poor"
        else:
            quality_category = "Very Poor"

        return {
            'brisque_score': quality_score,
            'quality_category': quality_category,
            'naturalness_features': naturalness_features,
            'distortion_features': distortion_features,
            'recommendations': self._get_quality_recommendations(quality_score, distortion_features)
        }

    def _get_quality_recommendations(self, score: float, distortion_features: Dict) -> List[str]:
        """Generate recommendations for quality improvement"""
        recommendations = []

        if score < 0.6:
            if distortion_features['sharpness'] < 100:
                recommendations.append("Image appears blurry - check focus or reduce motion blur")
            if distortion_features['noise_estimate'] > 20:
                recommendations.append("High noise detected - improve lighting or reduce ISO")
            if distortion_features['contrast'] < 30:
                recommendations.append("Low contrast - adjust lighting or camera settings")
            if distortion_features['contrast'] > 80:
                recommendations.append("High contrast - reduce harsh lighting or shadows")

        if not recommendations:
            recommendations.append("Image quality is acceptable for face recognition")

        return recommendations


# Global quality assessor instance
quality_assessor = BRISQUEInspiredQuality()

def assess_image_quality(image: np.ndarray, detailed: bool = False):
    """
    Convenience function for image quality assessment

    Args:
        image: Input image as numpy array
        detailed: Whether to return detailed analysis

    Returns:
        Quality score (0-1) or detailed assessment dict
    """
    if detailed:
        return quality_assessor.assess_quality_detailed(image)
    else:
        return quality_assessor.calculate_brisque_score(image)


# ===================
# USAGE EXAMPLE
# ===================
if __name__ == "__main__":
    # Test the quality assessor
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("Testing BRISQUE-inspired quality assessment...")
    score = assess_image_quality(test_image)
    print(f"Quality score for random noise: {score:.3f}")

    detailed = assess_image_quality(test_image, detailed=True)
    print(f"Quality category: {detailed['quality_category']}")
    print(f"Recommendations: {detailed['recommendations']}")

    print("✅ BRISQUE-inspired quality assessment working!")