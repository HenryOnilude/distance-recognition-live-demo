import cv2
import numpy as np

def apply_clahe_enhancement(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing

    Args:
        image: Input image (BGR or grayscale)
        clip_limit: Threshold for contrast limiting (default: 3.0)
        tile_grid_size: Size of the neighborhood area (default: (8,8))

    Returns:
        Enhanced image
    """
    try:
        if image is None or image.size == 0:
            return image

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Apply CLAHE based on image type
        if len(image.shape) == 3:
            # Color image - convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # Apply to L channel only
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image - apply CLAHE directly
            enhanced = clahe.apply(image)

        return enhanced

    except Exception as e:
        print(f"Error applying CLAHE enhancement: {e}")
        return image

def preprocess_for_face_detection(image):
    """
    Multi-strategy preprocessing for improved face detection

    Returns list of preprocessed images to try for detection
    """
    try:
        if image is None or image.size == 0:
            return [image]

        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        preprocessed_images = []

        # 1. Original grayscale
        preprocessed_images.append(gray)

        # 2. CLAHE enhanced
        clahe_enhanced = apply_clahe_enhancement(gray, clip_limit=3.0, tileGridSize=(8, 8))
        preprocessed_images.append(clahe_enhanced)

        # 3. Histogram equalized
        hist_eq = cv2.equalizeHist(gray)
        preprocessed_images.append(hist_eq)

        # 4. Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        preprocessed_images.append(blurred)

        # 5. CLAHE with different parameters for difficult lighting
        clahe_strong = apply_clahe_enhancement(gray, clip_limit=5.0, tileGridSize=(6, 6))
        preprocessed_images.append(clahe_strong)

        return preprocessed_images

    except Exception as e:
        print(f"Error in preprocessing for face detection: {e}")
        return [image]

def enhance_image_for_recognition(face_image):
    """
    Enhance face image specifically for recognition accuracy

    Args:
        face_image: Detected face region

    Returns:
        Enhanced face image ready for recognition
    """
    try:
        if face_image is None or face_image.size == 0:
            return face_image

        # Apply CLAHE enhancement
        enhanced = apply_clahe_enhancement(face_image, clip_limit=3.0, tileGridSize=(8, 8))

        # Additional sharpening for better feature extraction
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced.copy()

        # Create sharpening kernel
        sharpening_kernel = np.array([[-1, -1, -1],
                                     [-1,  9, -1],
                                     [-1, -1, -1]])

        # Apply subtle sharpening
        sharpened = cv2.filter2D(gray, -1, sharpening_kernel * 0.3)

        # Convert back to BGR if original was color
        if len(face_image.shape) == 3:
            # Convert sharpened grayscale back to BGR
            enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

            # Blend with original enhanced image
            final_enhanced = cv2.addWeighted(enhanced, 0.7, enhanced_bgr, 0.3, 0)
            return final_enhanced
        else:
            return sharpened

    except Exception as e:
        print(f"Error enhancing image for recognition: {e}")
        return face_image

def normalize_image_size(image, target_size=(224, 224)):
    """
    Normalize image size for consistent processing

    Args:
        image: Input image
        target_size: Target dimensions (width, height)

    Returns:
        Resized image
    """
    try:
        if image is None or image.size == 0:
            return image

        # Resize image maintaining aspect ratio
        height, width = image.shape[:2]
        target_width, target_height = target_size

        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)

        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Create canvas with target size
        if len(image.shape) == 3:
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((target_height, target_width), dtype=np.uint8)

        # Calculate position to center the image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        # Place resized image on canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        return canvas

    except Exception as e:
        print(f"Error normalizing image size: {e}")
        return image