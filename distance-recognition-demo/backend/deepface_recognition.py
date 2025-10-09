"""
Real DeepFace-based Face Recognition with Distance Research Integration
Replaces simulation.py with actual ML predictions adjusted by distance/quality research
"""

import cv2
import numpy as np
from deepface import DeepFace
from typing import Dict, Tuple, Optional
import time
import logging
from quality_scoring import calculate_quality_score, assess_image_quality_factors
from image_preprocessing import enhance_image_for_recognition, apply_clahe_enhancement
from distance_estimation import estimate_distance_from_face_size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridFaceRecognitionSystem:
    def __init__(self):
        """Initialize the hybrid system with DeepFace + distance research"""
        # Distance research parameters from your breakthrough
        self.distance_config = {
            "avg_face_width_cm": 14.0,
            "focal_length_pixels": 800,
            "min_distance": 1.5,
            "max_distance": 15.0
        }

        # Distance-adaptive confidence multipliers from your research
        self.confidence_adjustments = {
            "close": {"multiplier": 1.00, "threshold_adjust": 0.00},    # 2-4m: 89.1%
            "medium": {"multiplier": 0.92, "threshold_adjust": 0.05},   # 4-7m: 82.3%
            "far": {"multiplier": 0.81, "threshold_adjust": 0.10}       # 7-10m: 72.3%
        }

        # Expected accuracy by distance range (from your research)
        self.expected_accuracies = {
            "close": {"overall": 0.891, "age": 0.895, "gender": 0.850, "race": 0.966},
            "medium": {"overall": 0.823, "age": 0.750, "gender": 0.755, "race": 0.685},
            "far": {"overall": 0.723, "age": 0.712, "gender": 0.725, "race": 0.641}
        }

        logger.info("Hybrid Face Recognition System initialized with DeepFace + distance research")

    def classify_distance_range(self, distance_m: float) -> str:
        """Classify distance into research-based categories"""
        if distance_m <= 4.0:
            return "close"
        elif distance_m <= 7.0:
            return "medium"
        else:
            return "far"

    def preprocess_face_for_recognition(self, face_image):
        """Apply CLAHE preprocessing as specified in requirements"""
        try:
            # Apply CLAHE enhancement with specified parameters
            enhanced = apply_clahe_enhancement(
                face_image,
                clip_limit=3.0,
                tile_grid_size=(8, 8)
            )

            # Additional enhancement for recognition
            final_enhanced = enhance_image_for_recognition(enhanced)

            return final_enhanced

        except Exception as e:
            logger.error(f"Error in face preprocessing: {e}")
            return face_image

    def get_deepface_predictions(self, face_image):
        """Get real predictions from DeepFace for age, gender, race"""
        try:
            # Preprocess image
            processed_face = self.preprocess_face_for_recognition(face_image)

            # Run DeepFace analysis
            analysis = DeepFace.analyze(
                processed_face,
                actions=['age', 'gender'],
                enforce_detection=False,  # We already have detected face
                silent=True
            )

            # Handle both single result and list results
            if isinstance(analysis, list):
                analysis = analysis[0]

            return {
                'age': analysis.get('age', 25),
                'gender': analysis.get('gender', {})
            }

        except Exception as e:
            logger.error(f"DeepFace analysis failed: {e}")
            # Return fallback predictions
            return {
                'age': 25,
                'gender': {'Woman': 0.5, 'Man': 0.5}
            }

    def apply_distance_confidence_adjustment(self, prediction_confidence: float,
                                           distance_category: str,
                                           quality_score: float) -> float:
        """Apply your research-based confidence adjustments"""
        try:
            adjustment = self.confidence_adjustments[distance_category]

            # Apply distance multiplier and quality factor
            adjusted_confidence = (
                prediction_confidence *
                adjustment["multiplier"] *
                quality_score
            )

            return max(0.1, min(0.99, adjusted_confidence))

        except Exception as e:
            logger.error(f"Error in confidence adjustment: {e}")
            return prediction_confidence

    def process_gender_prediction(self, gender_data: dict, distance_category: str, quality_score: float):
        """Process gender prediction with distance adjustment"""
        try:
            # Extract highest confidence gender
            if 'Woman' in gender_data and 'Man' in gender_data:
                woman_conf = gender_data['Woman'] / 100.0  # DeepFace returns percentages
                man_conf = gender_data['Man'] / 100.0

                if woman_conf > man_conf:
                    predicted_gender = "Female"
                    raw_confidence = woman_conf
                else:
                    predicted_gender = "Male"
                    raw_confidence = man_conf
            else:
                # Fallback
                predicted_gender = "Male"
                raw_confidence = 0.5

            # Apply distance-based confidence adjustment
            adjusted_confidence = self.apply_distance_confidence_adjustment(
                raw_confidence, distance_category, quality_score
            )

            # Decision threshold with distance adjustment
            threshold = 0.5 + self.confidence_adjustments[distance_category]["threshold_adjust"]
            decision = "accept" if adjusted_confidence >= threshold else "reject"

            return {
                "predicted_class": predicted_gender,
                "confidence": round(adjusted_confidence, 3),
                "decision": decision,
                "expected_accuracy": self.expected_accuracies[distance_category]["gender"]
            }

        except Exception as e:
            logger.error(f"Error processing gender prediction: {e}")
            return {
                "predicted_class": "Male",
                "confidence": 0.5,
                "decision": "reject",
                "expected_accuracy": 0.7
            }

    def process_age_prediction(self, age_value: int, distance_category: str, quality_score: float):
        """Process age prediction with distance adjustment"""
        try:
            # Convert continuous age to binary classification (as in your research)
            if age_value <= 30:
                predicted_age_group = "Young"
                # Confidence based on how far from boundary
                raw_confidence = max(0.6, 1.0 - (age_value / 40.0))
            else:
                predicted_age_group = "Old"
                raw_confidence = max(0.6, min(0.95, (age_value - 20) / 60.0))

            # Apply distance-based confidence adjustment
            adjusted_confidence = self.apply_distance_confidence_adjustment(
                raw_confidence, distance_category, quality_score
            )

            # Decision threshold with distance adjustment
            threshold = 0.5 + self.confidence_adjustments[distance_category]["threshold_adjust"]
            decision = "accept" if adjusted_confidence >= threshold else "reject"

            return {
                "predicted_class": predicted_age_group,
                "confidence": round(adjusted_confidence, 3),
                "decision": decision,
                "expected_accuracy": self.expected_accuracies[distance_category]["age"]
            }

        except Exception as e:
            logger.error(f"Error processing age prediction: {e}")
            return {
                "predicted_class": "Young",
                "confidence": 0.5,
                "decision": "reject",
                "expected_accuracy": 0.7
            }

    def process_race_prediction(self, race_data: dict, distance_category: str, quality_score: float):
        """Process race/ethnicity prediction with distance adjustment"""
        try:
            # Find highest confidence race
            max_race = max(race_data.items(), key=lambda x: x[1])
            predicted_race = max_race[0]
            raw_confidence = max_race[1] / 100.0  # Convert percentage to decimal

            # Convert to binary classification (Light/Dark as in your research)
            if predicted_race.lower() in ['white', 'asian']:
                ethnicity_group = "Light"
            else:
                ethnicity_group = "Dark"

            # Apply distance-based confidence adjustment
            adjusted_confidence = self.apply_distance_confidence_adjustment(
                raw_confidence, distance_category, quality_score
            )

            # Decision threshold with distance adjustment
            threshold = 0.5 + self.confidence_adjustments[distance_category]["threshold_adjust"]
            decision = "accept" if adjusted_confidence >= threshold else "reject"

            return {
                "predicted_class": ethnicity_group,
                "confidence": round(adjusted_confidence, 3),
                "decision": decision,
                "expected_accuracy": self.expected_accuracies[distance_category]["race"]
            }

        except Exception as e:
            logger.error(f"Error processing race prediction: {e}")
            return {
                "predicted_class": "Light",
                "confidence": 0.5,
                "decision": "reject",
                "expected_accuracy": 0.7
            }

    def process_frame_analysis(self, face_bbox: Tuple[int, int, int, int],
                              face_image, image_shape: Tuple[int, int]) -> Dict:
        """
        Complete frame analysis with real DeepFace + distance research

        Args:
            face_bbox: (x, y, width, height) of detected face
            face_image: Cropped face image for analysis
            image_shape: (height, width) of original image
        """
        start_time = time.time()

        try:
            x, y, w, h = face_bbox

            # Step 1: Distance estimation using your formula
            distance_m = estimate_distance_from_face_size(face_bbox, image_shape)
            distance_category = self.classify_distance_range(distance_m)

            # Step 2: Quality assessment using your algorithms
            quality_score = calculate_quality_score(face_image)

            # Step 3: Real DeepFace predictions
            deepface_results = self.get_deepface_predictions(face_image)

            # Step 4: Apply distance-adaptive adjustments
            predictions = {}

            # Process gender prediction
            predictions["gender"] = self.process_gender_prediction(
                deepface_results["gender"], distance_category, quality_score
            )

            # Process age prediction
            predictions["age"] = self.process_age_prediction(
                deepface_results["age"], distance_category, quality_score
            )

            # Ethnicity prediction removed for ethical considerations

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            return {
                "success": True,
                "processing_time_ms": round(processing_time, 1),
                "distance_m": round(distance_m, 2),
                "distance_category": distance_category,
                "quality_score": round(quality_score, 3),
                "face_bbox": face_bbox,
                "predictions": predictions,
                "expected_overall_accuracy": round(self.expected_accuracies[distance_category]["overall"], 3),
                "method": "DeepFace + Distance Research"
            }

        except Exception as e:
            logger.error(f"Error in frame analysis: {e}")
            processing_time = (time.time() - start_time) * 1000

            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": round(processing_time, 1),
                "distance_m": 5.0,
                "distance_category": "medium",
                "quality_score": 0.5,
                "face_bbox": face_bbox,
                "predictions": {},
                "expected_overall_accuracy": 0.7,
                "method": "DeepFace + Distance Research"
            }

    def get_system_info(self) -> Dict:
        """Return system information"""
        return {
            "system_name": "Hybrid DeepFace + Distance Research System",
            "version": "2.0.0",
            "accuracy_range": "89.1% (close) to 72.3% (far)",
            "supported_tasks": ["gender", "age"],
            "distance_range": "2-10 meters",
            "processing_target": "<200ms per frame",
            "ml_backend": "DeepFace (VGG-Face, Facenet)",
            "research_integration": "Distance-adaptive confidence adjustment",
            "preprocessing": "CLAHE enhancement (clipLimit=3.0, tileGridSize=(8,8))",
            "status": "ready"
        }

# Create global system instance
recognition_system = HybridFaceRecognitionSystem()