"""
Face Recognition Demo Simulation
Uses extracted research data to simulate distance-adaptive face recognition
Based on 89.1% (close) -> 72.3% (far) accuracy degradation
"""

import random
from typing import Dict, Tuple, Optional
import math

class FaceRecognitionSimulator:
    def __init__(self):
        # Your extracted research data
        self.performance_model = {
            "2-4m": {"overall": 0.891, "age": 0.895, "gender": 0.850},
            "4-7m": {"overall": 0.823, "age": 0.750, "gender": 0.755},
            "7-10m": {"overall": 0.723, "age": 0.712, "gender": 0.725}
        }
        
        self.distance_config = {
            "avg_face_width_cm": 14.0,
            "focal_length_pixels": 800,
            "min_distance": 1.5,
            "max_distance": 15.0
        }
        
        self.confidence_model = {
            "close": {"multiplier": 1.00, "threshold_adjust": 0.00},
            "medium": {"multiplier": 0.92, "threshold_adjust": 0.05},
            "far": {"multiplier": 0.81, "threshold_adjust": 0.10}
        }
    
    def estimate_distance_from_face_width(self, face_width_pixels: int) -> float:
        """Estimate distance using face width and camera calibration"""
        if face_width_pixels <= 0:
            return self.distance_config["max_distance"]
        
        distance_cm = (
            self.distance_config["avg_face_width_cm"] * 
            self.distance_config["focal_length_pixels"]
        ) / face_width_pixels
        
        distance_m = distance_cm / 100.0
        
        # Clamp to reasonable range
        return max(
            self.distance_config["min_distance"],
            min(distance_m, self.distance_config["max_distance"])
        )
    
    def classify_distance_range(self, distance_m: float) -> str:
        """Classify distance into close/medium/far categories"""
        if distance_m <= 4.0:
            return "close"
        elif distance_m <= 7.0:
            return "medium"
        else:
            return "far"
    
    def get_range_key(self, distance_m: float) -> str:
        """Get the performance model key for a distance"""
        if distance_m <= 4.0:
            return "2-4m"
        elif distance_m <= 7.0:
            return "4-7m"
        else:
            return "7-10m"
    
    def simulate_quality_score(self, face_width: int, face_height: int, 
                             brightness: Optional[float] = None) -> float:
        """Simulate face quality score based on size and estimated brightness"""
        # Size component (larger faces = better quality)
        size_score = min(1.0, min(face_width, face_height) / 150.0)
        
        # Simulated brightness (if not provided)
        if brightness is None:
            brightness = random.uniform(0.3, 0.8)
        
        # Brightness component (optimal around 0.5)
        brightness_score = 1.0 - abs(brightness - 0.5) * 2.0
        
        # Simulated sharpness (larger faces tend to be sharper)
        sharpness_score = min(1.0, face_width / 200.0)
        
        # Combined quality score
        quality = (size_score * 0.5 + brightness_score * 0.3 + sharpness_score * 0.2)
        return max(0.1, min(1.0, quality))
    
    def generate_predictions(self, distance_m: float, quality_score: float) -> Dict:
        """Generate simulated predictions for age, gender"""
        range_key = self.get_range_key(distance_m)
        distance_category = self.classify_distance_range(distance_m)
        
        # Get base accuracies for this distance
        base_accuracies = self.performance_model[range_key]
        confidence_config = self.confidence_model[distance_category]
        
        predictions = {}
        
        for task in ["age", "gender"]:
            base_accuracy = base_accuracies[task]
            
            # Apply quality and distance adjustments
            adjusted_confidence = (
                base_accuracy * 
                confidence_config["multiplier"] * 
                quality_score
            )
            
            # Add some realistic noise
            noise = random.uniform(-0.05, 0.05)
            final_confidence = max(0.1, min(0.99, adjusted_confidence + noise))
            
            # Generate class prediction
            if task == "gender":
                classes = ["Female", "Male"]
                # Slightly bias toward one class based on confidence
                prediction_idx = 0 if final_confidence > 0.6 else 1
            elif task == "age":
                classes = ["Young", "Old"]
                prediction_idx = 0 if final_confidence > 0.65 else 1
            
            # Decision threshold with distance adjustment
            decision_threshold = 0.5 + confidence_config["threshold_adjust"]
            decision = "accept" if final_confidence >= decision_threshold else "reject"
            
            predictions[task] = {
                "predicted_class": classes[prediction_idx],
                "confidence": final_confidence,
                "decision": decision,
                "expected_accuracy": base_accuracy
            }
        
        return predictions
    
    def process_frame_simulation(self, face_bbox: Tuple[int, int, int, int], 
                               image_shape: Tuple[int, int]) -> Dict:
        """
        Simulate complete frame processing
        Args:
            face_bbox: (x, y, width, height) of detected face
            image_shape: (height, width) of image
        """
        x, y, w, h = face_bbox
        
        # Estimate distance from face width
        distance_m = self.estimate_distance_from_face_width(w)
        distance_category = self.classify_distance_range(distance_m)
        
        # Calculate quality score
        quality_score = self.simulate_quality_score(w, h)
        
        # Generate predictions
        predictions = self.generate_predictions(distance_m, quality_score)
        
        # Calculate processing time simulation (varies by distance/quality)
        base_processing_time = 85  # ms
        distance_penalty = (distance_m - 2.0) * 5  # farther = slightly slower
        quality_penalty = (1.0 - quality_score) * 15  # lower quality = slower
        
        processing_time = max(50, base_processing_time + distance_penalty + quality_penalty)
        processing_time += random.uniform(-10, 10)  # Add realistic variation
        
        return {
            "success": True,
            "processing_time_ms": round(processing_time, 1),
            "distance_m": round(distance_m, 2),
            "distance_category": distance_category,
            "quality_score": round(quality_score, 3),
            "face_bbox": face_bbox,
            "predictions": predictions,
            "expected_overall_accuracy": round(self.performance_model[self.get_range_key(distance_m)]["overall"], 3)
        }
    
    def get_system_info(self) -> Dict:
        """Return system information for the demo"""
        return {
            "system_name": "Distance-Adaptive Face Recognition Demo",
            "version": "1.0.0",
            "accuracy_range": "89.1% (close) to 72.3% (far)",
            "supported_tasks": ["gender", "age"],
            "distance_range": "2-10 meters",
            "processing_target": "<100ms per frame",
            "research_basis": "Hybrid CV + Deep Learning Ensemble",
            "status": "ready"
        }

# Create global simulator instance
simulator = FaceRecognitionSimulator()