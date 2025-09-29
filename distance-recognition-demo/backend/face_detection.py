"""
Face Detection using OpenCV
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        # Initialize Haar cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            logger.error("Failed to load Haar cascade classifier")
            raise RuntimeError("Could not load face detection model")
        logger.info("Face detector initialized successfully")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces and return bounding boxes with DEX-inspired preprocessing"""
        if image is None or image.size == 0:
            logger.warning("Empty or invalid image received")
            return []

        logger.info(f"Processing image: shape={image.shape}, dtype={image.dtype}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info(f"Converted to grayscale: shape={gray.shape}")

        # DEX-inspired preprocessing
        # Enhance image quality with histogram equalization
        gray = cv2.equalizeHist(gray)

        # Apply Gaussian blur to reduce noise (DEX preprocessing)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Try multiple detection parameters for better results
        faces = []

        # Primary detection with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,  # Slightly more strict to reduce false positives
            minSize=(40, 40),  # Larger minimum size for better quality
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # If no faces found, try with relaxed parameters
        if len(faces) == 0:
            logger.info("No faces found with primary parameters, trying relaxed detection")
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(25, 25),
                maxSize=(400, 400),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        # DEX-inspired face preprocessing: extend face region by 40%
        if len(faces) > 0:
            enhanced_faces = []
            h_img, w_img = gray.shape

            for face in faces:
                # Convert numpy array to tuple if needed
                if hasattr(face, 'tolist'):
                    x, y, w, h = face.tolist()
                else:
                    x, y, w, h = face

                # Ensure values are Python integers
                x, y, w, h = int(x), int(y), int(w), int(h)

                # Add 40% margin as described in DEX paper
                margin_w = int(w * 0.4)
                margin_h = int(h * 0.4)

                # Extend the bounding box
                x_new = max(0, x - margin_w)
                y_new = max(0, y - margin_h)
                w_new = min(w_img - x_new, w + 2 * margin_w)
                h_new = min(h_img - y_new, h + 2 * margin_h)

                enhanced_faces.append((x_new, y_new, w_new, h_new))
                logger.info(f"Enhanced face: original=({x},{y},{w},{h}) -> enhanced=({x_new},{y_new},{w_new},{h_new})")

            faces = enhanced_faces

        logger.info(f"Detected {len(faces)} faces with DEX preprocessing")
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                logger.info(f"Face {i}: x={x}, y={y}, w={w}, h={h}")

        return faces

# Create global detector instance
detector = FaceDetector()
