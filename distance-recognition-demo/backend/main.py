"""
FastAPI server for Distance Recognition Live Demo
Hybrid DeepFace + Distance Research Integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import time
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from face_detection import detector
from quality_scoring import calculate_quality_score

# Try to import InsightFace (better accuracy), fallback to DeepFace, then simulation
USE_INSIGHTFACE = True  # Set to False to use DeepFace instead

try:
    if USE_INSIGHTFACE:
        from insightface_recognition import insightface_recognition_system as recognition_system
        logger.info("Using InsightFace + Distance Research System (Better accuracy for diverse faces)")
        SYSTEM_MODE = "insightface"
    else:
        from deepface_recognition import recognition_system
        logger.info("Using DeepFace + Distance Research System")
        SYSTEM_MODE = "deepface"
except ImportError as e:
    logger.warning(f"ML systems not available ({e}), using enhanced simulation")
    from fallback_recognition import fallback_system as recognition_system
    SYSTEM_MODE = "fallback"

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
    return recognition_system.get_system_info()

@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze uploaded image using hybrid DeepFace + distance research system"""
    start_time = time.time()

    try:
        logger.info(f"Processing frame analysis for file: {file.filename}")

        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Detect faces using existing face detection
        faces = detector.detect_faces(image_cv)

        if not faces:
            processing_time = (time.time() - start_time) * 1000
            logger.warning("No face detected in uploaded image")
            return {
                "error": "No face detected",
                "processing_time_ms": round(processing_time, 1),
                "suggestions": [
                    "Ensure face is clearly visible",
                    "Check lighting conditions",
                    "Face should be at least 60x60 pixels",
                    "Avoid extreme angles or occlusion"
                ]
            }

        # Use largest face for analysis
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Extract face region for DeepFace analysis
        face_region = image_cv[y:y+h, x:x+w]

        # Validate face region
        if face_region.size == 0 or w < 60 or h < 60:
            processing_time = (time.time() - start_time) * 1000
            logger.warning(f"Face too small for reliable analysis: {w}x{h}")
            return {
                "error": "Face too small for reliable analysis",
                "processing_time_ms": round(processing_time, 1),
                "face_size": f"{w}x{h}",
                "minimum_required": "60x60"
            }

        # Process with hybrid DeepFace + distance research system
        result = recognition_system.process_frame_analysis(
            face_bbox=largest_face,
            face_image=face_region,
            image_shape=image_cv.shape[:2]
        )

        logger.info(f"Analysis completed in {result.get('processing_time_ms', 0)}ms")
        return result

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Error in frame analysis: {str(e)}")
        return {
            "error": f"Processing failed: {str(e)}",
            "processing_time_ms": round(processing_time, 1),
            "error_type": type(e).__name__
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
