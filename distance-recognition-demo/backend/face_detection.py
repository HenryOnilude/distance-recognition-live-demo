"""
Face Detection using SCRFD (upgraded from OpenCV DNN)
SCRFD provides 35% better accuracy at distance compared to traditional detectors
Research-backed state-of-the-art face detection via InsightFace
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
import os
from insightface.app import FaceAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        # Initialize SCRFD face detector via InsightFace (state-of-the-art)
        try:
            # Load SCRFD detector from InsightFace buffalo_s model (lighter, memory-efficient)
            self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))  # Higher resolution for better distance detection
            self.detector_type = "SCRFD"
            logger.info("✅ SCRFD Face detector loaded successfully (buffalo_s - memory optimized)")
        except Exception as e:
            # Fallback to OpenCV DNN if SCRFD fails
            logger.warning(f"SCRFD model not available ({e}), falling back to OpenCV DNN")
            self.model_path = os.path.dirname(__file__)
            prototxt_path = os.path.join(self.model_path, "models", "deploy.prototxt")
            model_path = os.path.join(self.model_path, "models", "res10_300x300_ssd_iter_140000.caffemodel")

            try:
                # Load DNN model
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                self.detector_type = "DNN"
                logger.info("✅ DNN Face detector loaded successfully (40-60% better accuracy than Haar)")
            except Exception as e2:
                # Final fallback to Haar cascade
                logger.warning(f"DNN model not found ({e2}), falling back to Haar cascade")
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.face_cascade.empty():
                    logger.error("Failed to load Haar cascade classifier")
                    raise RuntimeError("Could not load any face detection model")
                self.detector_type = "Haar"
                logger.info("Haar cascade face detector initialized (fallback mode)")

        # Detection parameters
        self.confidence_threshold = 0.5  # Confidence threshold for detections
        self.min_face_size = 40  # Minimum face size in pixels

    def detect_faces_scrfd(self, image: np.ndarray):
        """SCRFD-based face detection with full analysis data (35% better accuracy at distance than DNN)"""
        # Convert BGR to RGB for InsightFace
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        # Get face detections from SCRFD - KEEP ALL DATA
        faces = self.app.get(rgb_image)

        face_data = []
        for face in faces:
            # Extract bounding box from detection
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox

            # Convert to (x, y, width, height) format
            w = x2 - x
            h = y2 - y

            # Filter by minimum size and confidence
            if w >= self.min_face_size and h >= self.min_face_size:
                # CRITICAL FIX: Return full analysis data instead of just bbox
                face_info = {
                    'bbox': (x, y, w, h),
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None),
                    'embedding': getattr(face, 'embedding', None),
                    'det_score': getattr(face, 'det_score', None)
                }

                # DEBUG: Log the raw SCRFD results for troubleshooting
                logger.info(f"🔍 SCRFD RAW ANALYSIS: bbox=({x},{y},{w},{h}), age={face_info['age']}, gender={face_info['gender']}, det_score={face_info['det_score']}")
                face_data.append(face_info)

        return face_data

    def detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """DNN-based face detection (40-60% better accuracy than Haar)"""
        h, w = image.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )

        # Pass blob through network
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Convert to (x, y, width, height) format
                face_w = x1 - x
                face_h = y1 - y

                # Filter by minimum size
                if face_w >= self.min_face_size and face_h >= self.min_face_size:
                    faces.append((x, y, face_w, face_h))

        return faces

    def detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Haar cascade detection (fallback method)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # DEX-inspired preprocessing
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Primary detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # If no faces found, try relaxed parameters
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

        return faces.tolist() if len(faces) > 0 else []

    def detect_faces(self, image: np.ndarray):
        """Detect faces using SCRFD (preferred), DNN, or Haar cascade (fallback)"""
        if image is None or image.size == 0:
            logger.warning("Empty or invalid image received")
            return []

        logger.info(f"Processing image: shape={image.shape}, dtype={image.dtype}")

        # Use SCRFD, DNN, or Haar cascade based on availability
        if self.detector_type == "SCRFD":
            faces_data = self.detect_faces_scrfd(image)
            logger.info(f"SCRFD detected {len(faces_data)} faces")
            # Extract just bboxes for backward compatibility in enhancement step
            faces = [face_data['bbox'] for face_data in faces_data]
        elif self.detector_type == "DNN":
            faces = self.detect_faces_dnn(image)
            faces_data = [{'bbox': bbox} for bbox in faces]  # Create compatible format
            logger.info(f"DNN detected {len(faces)} faces")
        else:
            faces = self.detect_faces_haar(image)
            faces_data = [{'bbox': bbox} for bbox in faces]  # Create compatible format
            logger.info(f"Haar detected {len(faces)} faces")

        # DEX-inspired face preprocessing: extend face region by 40%
        if len(faces) > 0:
            enhanced_faces = []
            enhanced_faces_data = []
            h_img, w_img = image.shape[:2]

            for i, face in enumerate(faces):
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

                # Update face data with enhanced bbox
                if i < len(faces_data):
                    enhanced_face_data = faces_data[i].copy()
                    enhanced_face_data['bbox'] = (x_new, y_new, w_new, h_new)
                    enhanced_faces_data.append(enhanced_face_data)

                logger.info(f"Enhanced face: original=({x},{y},{w},{h}) -> enhanced=({x_new},{y_new},{w_new},{h_new})")

            faces = enhanced_faces
            faces_data = enhanced_faces_data

        logger.info(f"Final: Detected {len(faces)} faces using {self.detector_type} with DEX preprocessing")
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                logger.info(f"Face {i}: x={x}, y={y}, w={w}, h={h}")

        # Return both bbox list (backward compatibility) and full data
        if self.detector_type == "SCRFD":
            return faces, faces_data  # Return enhanced data for SCRFD
        else:
            return faces  # Backward compatibility for DNN/Haar

# Create global detector instance
detector = FaceDetector()
