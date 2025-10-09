import cv2
import numpy as np

def calculate_quality_score(face_image):
    """
    Calculate quality score using multiple factors:
    - Laplacian variance (sharpness)
    - Brightness analysis
    - Contrast analysis
    - Face size scoring
    - Edge density

    Returns score between 0.0 and 1.0
    """
    try:
        if face_image is None or face_image.size == 0:
            return 0.0

        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()

        height, width = gray.shape

        # 1. SHARPNESS SCORING (Laplacian Variance)
        # Higher variance = sharper image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normalize to 0-1

        # 2. BRIGHTNESS SCORING
        # Optimal brightness around 0.4-0.7 range
        mean_brightness = np.mean(gray) / 255.0
        if 0.3 <= mean_brightness <= 0.7:
            brightness_score = 1.0
        elif 0.2 <= mean_brightness < 0.3 or 0.7 < mean_brightness <= 0.8:
            brightness_score = 0.7
        elif 0.1 <= mean_brightness < 0.2 or 0.8 < mean_brightness <= 0.9:
            brightness_score = 0.4
        else:
            brightness_score = 0.1  # Too dark or too bright

        # 3. CONTRAST SCORING
        # Good contrast has standard deviation > 40
        std_contrast = np.std(gray)
        contrast_score = min(1.0, std_contrast / 80.0)  # Normalize to 0-1

        # 4. SIZE SCORING
        # Larger faces generally have better quality
        face_area = width * height
        if face_area >= 10000:  # 100x100 or larger
            size_score = 1.0
        elif face_area >= 6400:  # 80x80 to 100x100
            size_score = 0.8
        elif face_area >= 3600:  # 60x60 to 80x80
            size_score = 0.6
        else:
            size_score = 0.3  # Very small faces

        # 5. EDGE DENSITY SCORING
        # Good quality images have clear edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(1.0, edge_density * 10)  # Normalize

        # WEIGHTED COMBINATION
        # Based on research: sharpness and brightness most important
        quality_score = (
            sharpness_score * 0.30 +    # Sharpness most important
            brightness_score * 0.25 +   # Brightness critical
            contrast_score * 0.20 +     # Contrast important
            size_score * 0.15 +         # Size matters for recognition
            edge_score * 0.10           # Edge definition
        )

        return min(1.0, max(0.0, quality_score))

    except Exception as e:
        print(f"Error calculating quality score: {e}")
        return 0.0

def assess_image_quality_factors(face_image):
    """
    Detailed quality assessment returning individual factor scores
    """
    try:
        if face_image is None or face_image.size == 0:
            return {
                'overall_quality': 0.0,
                'sharpness': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'size': 0.0,
                'edge_density': 0.0
            }

        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()

        height, width = gray.shape

        # Calculate individual factors
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 1000.0)

        mean_brightness = np.mean(gray) / 255.0
        brightness = 1.0 - abs(mean_brightness - 0.5) * 2.0  # Peak at 0.5

        contrast = min(1.0, np.std(gray) / 80.0)

        face_area = width * height
        size = min(1.0, face_area / 10000.0)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = min(1.0, (np.sum(edges > 0) / edges.size) * 10)

        overall = calculate_quality_score(face_image)

        return {
            'overall_quality': overall,
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'size': size,
            'edge_density': edge_density
        }

    except Exception as e:
        print(f"Error in detailed quality assessment: {e}")
        return {
            'overall_quality': 0.0,
            'sharpness': 0.0,
            'brightness': 0.0,
            'contrast': 0.0,
            'size': 0.0,
            'edge_density': 0.0
        }