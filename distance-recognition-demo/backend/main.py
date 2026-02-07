"""
FastAPI server for Distance Recognition Live Demo
Hybrid DeepFace + Distance Research Integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
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

from quality_scoring import calculate_quality_score

# Lazy loading for ML models - don't load until first request
USE_INSIGHTFACE = True
SYSTEM_MODE = "insightface" if USE_INSIGHTFACE else "deepface"
recognition_system = None
detector = None

def get_detector():
    """Lazy load face detector on first request"""
    global detector
    if detector is None:
        logger.info("Loading face detector...")
        from face_detection import detector as face_detector
        detector = face_detector
        logger.info("✅ Face detector loaded successfully")
    return detector

def get_recognition_system():
    """Lazy load recognition system on first request"""
    global recognition_system
    if recognition_system is None:
        logger.info(f"Loading {SYSTEM_MODE} recognition system...")
        try:
            if USE_INSIGHTFACE:
                from insightface_recognition import insightface_recognition_system
                recognition_system = insightface_recognition_system
                logger.info("✅ InsightFace system loaded successfully")
            else:
                from deepface_recognition import recognition_system as deep_system
                recognition_system = deep_system
                logger.info("✅ DeepFace system loaded successfully")
        except ImportError as e:
            logger.warning(f"ML systems not available ({e}), using fallback")
            from fallback_recognition import fallback_system
            recognition_system = fallback_system
    return recognition_system

app = FastAPI(title="Distance Recognition API", version="1.0.0")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "https://synchrocv.com",
        "http://synchrocv.com",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app$",  # Allow all Vercel deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("🚀 FastAPI application starting...")
    logger.info(f"System mode: {SYSTEM_MODE}")
    logger.info("All models loaded successfully!")
    logger.info("=" * 50)

@app.get("/")
def root():
    return {"message": "Distance Recognition Live Demo API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": SYSTEM_MODE}

@app.get("/system-info")
def get_system_info():
    system = get_recognition_system()
    return system.get_system_info()

@app.get("/routes")
def list_routes():
    """Debug endpoint to list all registered routes"""
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": getattr(route, 'methods', ['WEBSOCKET' if 'websocket' in route.path else 'N/A'])
        })
    return {"routes": routes}

@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze uploaded image using hybrid DeepFace + distance research system"""
    start_time = time.time()

    try:
        logger.info(f"Processing frame analysis for file: {file.filename}")

        # SECURITY FIX: Input validation
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file size (limit to 10MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Check file type
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        file_ext = None
        if file.filename:
            file_ext = '.' + file.filename.lower().split('.')[-1] if '.' in file.filename else None

        if file_ext not in valid_extensions:
            raise HTTPException(status_code=400, detail=f"Invalid file type. Supported: {valid_extensions}")

        # FAST DECODE: Read bytes -> Decode directly to BGR
        try:
            nparr = np.frombuffer(contents, np.uint8)
            image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_cv is None:
                raise ValueError("Failed to decode image")
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(img_error)}")

        # Check image dimensions
        height, width = image_cv.shape[:2]
        if width < 50 or height < 50:
            raise HTTPException(status_code=400, detail="Image too small (minimum 50x50 pixels)")
        
        # Auto-resize large images instead of rejecting them
        if width > 4096 or height > 4096:
            logger.info(f"Resizing large image from {width}x{height}")
            scale = min(4096 / width, 4096 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_cv = cv2.resize(image_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized to {new_width}x{new_height}")

        # Detect faces using lazy-loaded face detection
        face_detector = get_detector()
        detection_result = face_detector.detect_faces(image_cv)

        # Handle both new SCRFD format and legacy format
        if face_detector.detector_type == "SCRFD" and isinstance(detection_result, tuple):
            faces, faces_data = detection_result
        else:
            faces = detection_result
            faces_data = None

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

        # Get corresponding face data if available
        largest_face_data = None
        if faces_data:
            # Find matching face_data for the largest face
            for face_data in faces_data:
                if face_data['bbox'] == largest_face:
                    largest_face_data = face_data
                    break

        # Extract face region for analysis
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

        # Process with recognition system (InsightFace + distance research)
        system = get_recognition_system()
        result = system.process_frame_analysis(
            face_bbox=largest_face,
            face_image=face_region,
            full_image=image_cv,
            image_shape=image_cv.shape[:2],
            face_data=largest_face_data  # Pass SCRFD analysis data
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

@app.websocket("/ws/analyze-stream")
async def websocket_analyze_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time frame analysis
    Protocol:
    - Client sends: Binary JPEG frame data
    - Server responds: JSON analysis result
    - Backpressure: Server processes one frame at a time
    """
    await websocket.accept()
    logger.info("🔌 WebSocket connection established")

    # Pre-load models on first connection to avoid cold start
    face_detector = get_detector()
    recognition_sys = get_recognition_system()

    frame_count = 0

    try:
        while True:
            # Receive binary frame data from client
            frame_data = await websocket.receive_bytes()
            frame_count += 1
            start_time = time.time()

            try:
                # Decode JPEG bytes directly to BGR image
                nparr = np.frombuffer(frame_data, np.uint8)
                image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image_cv is None:
                    await websocket.send_json({
                        "error": "Failed to decode frame",
                        "frame_count": frame_count
                    })
                    continue

                # Detect faces
                detection_result = face_detector.detect_faces(image_cv)

                # Handle both SCRFD format and legacy format
                if face_detector.detector_type == "SCRFD" and isinstance(detection_result, tuple):
                    faces, faces_data = detection_result
                else:
                    faces = detection_result
                    faces_data = None

                if not faces:
                    processing_time = (time.time() - start_time) * 1000
                    await websocket.send_json({
                        "error": "No face detected",
                        "processing_time_ms": round(processing_time, 1),
                        "frame_count": frame_count
                    })
                    continue

                # Use largest face for analysis
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face

                # Get corresponding face data if available
                largest_face_data = None
                if faces_data:
                    for face_data in faces_data:
                        if face_data['bbox'] == largest_face:
                            largest_face_data = face_data
                            break

                # Extract face region
                face_region = image_cv[y:y+h, x:x+w]

                # Validate face region
                if face_region.size == 0 or w < 60 or h < 60:
                    processing_time = (time.time() - start_time) * 1000
                    await websocket.send_json({
                        "error": "Face too small for reliable analysis",
                        "processing_time_ms": round(processing_time, 1),
                        "face_size": f"{w}x{h}",
                        "frame_count": frame_count
                    })
                    continue

                # Process with recognition system
                result = recognition_sys.process_frame_analysis(
                    face_bbox=largest_face,
                    face_image=face_region,
                    full_image=image_cv,
                    image_shape=image_cv.shape[:2],
                    face_data=largest_face_data
                )

                # Add frame count for debugging
                result['frame_count'] = frame_count

                # Send result back to client
                await websocket.send_json(result)

                logger.info(f"✅ Frame {frame_count} processed in {result.get('processing_time_ms', 0)}ms")

            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                logger.error(f"❌ Error processing frame {frame_count}: {str(e)}")
                await websocket.send_json({
                    "error": f"Processing failed: {str(e)}",
                    "processing_time_ms": round(processing_time, 1),
                    "frame_count": frame_count,
                    "error_type": type(e).__name__
                })

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected after {frame_count} frames")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
