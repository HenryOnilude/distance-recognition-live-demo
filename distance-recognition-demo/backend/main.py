"""
FastAPI server for Distance Recognition Live Demo
Combines face detection with research-based simulation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import time

from face_detection import detector
from simulation import simulator

app = FastAPI(title="Distance Recognition API", version="1.0.0")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Distance Recognition Live Demo API", "status": "running"}

@app.get("/system-info")
def get_system_info():
    return simulator.get_system_info()

@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze uploaded image for face recognition and distance estimation"""
    start_time = time.time()
    
    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = detector.detect_faces(image_cv)
        
        if not faces:
            return {"error": "No face detected", "processing_time_ms": round((time.time() - start_time) * 1000, 1)}
        
        # Use largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Estimate distance from face width
        distance_m = simulator.estimate_distance_from_face_width(w)
        distance_category = simulator.classify_distance_range(distance_m)
        
        # Generate predictions based on distance
        quality_score = 0.8  # Simplified for demo
        
        # DEX-inspired ensemble predictions with classification approach
        predictions = {}

        # Create a stable seed based on face position for consistent predictions
        face_seed = hash((x, y, w, h)) % 1000

        # Get expected accuracy for this distance
        range_key = "2-4m" if distance_m <= 4.0 else "4-7m" if distance_m <= 7.0 else "7-10m"

        # Simulate ensemble of 5 models (simplified from DEX's 20)
        ensemble_size = 5
        ensemble_predictions = {}

        for task in ["age", "gender"]:
            task_predictions = []

            for model_idx in range(ensemble_size):
                # Different seed for each model in ensemble
                np.random.seed(face_seed + model_idx * 100)

                if task == "gender":
                    # DEX-inspired gender classification with model variation
                    base_male_prob = 0.75 + (model_idx - 2) * 0.05  # Slight variation per model
                    base_male_prob = max(0.6, min(0.9, base_male_prob))

                    # Adjust probabilities based on distance
                    distance_factor = max(0.5, 1.0 - (distance_m - 2.0) / 8.0)
                    male_prob = base_male_prob * distance_factor + (1 - distance_factor) * 0.5

                    predicted_class = "Male" if np.random.random() < male_prob else "Female"
                    confidence = male_prob if predicted_class == "Male" else (1.0 - male_prob)

                elif task == "age":
                    # Age prediction with model variation
                    base_age = 25 + (model_idx - 2) * 2  # Age varies slightly per model
                    base_age = max(20, min(30, base_age))

                    # Create age probability distribution
                    age_probs = np.zeros(101)
                    for age in range(101):
                        sigma = 8.0 + distance_m * 2.0 + model_idx * 0.5  # Model variation
                        prob = np.exp(-((age - base_age) ** 2) / (2 * sigma ** 2))
                        age_probs[age] = prob

                    age_probs = age_probs / np.sum(age_probs)
                    expected_age = np.sum([age * age_probs[age] for age in range(101)])

                    predicted_class = "Young" if expected_age < 40 else "Old"
                    confidence = np.sum(age_probs[:40]) if predicted_class == "Young" else np.sum(age_probs[40:])


                task_predictions.append({
                    "class": predicted_class,
                    "confidence": confidence
                })

            # Ensemble voting: majority vote with confidence weighting
            class_votes = {}
            total_confidence = 0

            for pred in task_predictions:
                pred_class = pred["class"]
                pred_conf = pred["confidence"]

                if pred_class not in class_votes:
                    class_votes[pred_class] = 0
                class_votes[pred_class] += pred_conf
                total_confidence += pred_conf

            # Select class with highest weighted vote
            final_class = max(class_votes.keys(), key=lambda k: class_votes[k])
            final_confidence = class_votes[final_class] / total_confidence if total_confidence > 0 else 0.5

            ensemble_predictions[task] = {
                "predicted_class": final_class,
                "confidence": final_confidence
            }

        # Process ensemble results for each task
        for task in ["age", "gender"]:
            # Get ensemble prediction for this task
            ensemble_result = ensemble_predictions[task]
            predicted_class = ensemble_result["predicted_class"]
            confidence = ensemble_result["confidence"]

            # Get expected accuracy for this distance and task
            expected_accuracy = simulator.performance_model[range_key][task]

            # Adjust confidence based on expected model accuracy and quality
            confidence = confidence * expected_accuracy * quality_score
            confidence = max(0.1, min(0.99, confidence))

            decision = "accept" if confidence > 0.5 else "reject"

            predictions[task] = {
                "predicted_class": predicted_class,
                "confidence": round(confidence, 3),
                "decision": decision,
                "expected_accuracy": round(expected_accuracy, 3)
            }
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "processing_time_ms": round(processing_time, 1),
            "distance_m": round(distance_m, 2),
            "distance_category": distance_category,
            "quality_score": quality_score,
            "face_bbox": largest_face,
            "predictions": predictions,
            "expected_overall_accuracy": round(simulator.performance_model[range_key]["overall"], 3)
        }
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}", "processing_time_ms": round((time.time() - start_time) * 1000, 1)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
