# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Distance-Adaptive Face Recognition Live Demo showcasing accuracy degradation with distance (89.1% close → 72.3% far). The system processes live webcam feeds and provides real-time distance estimation and demographic predictions.

## Architecture

**Backend (Python/FastAPI):**
- `distance-recognition-demo/backend/main.py` - FastAPI server with `/analyze-frame` endpoint
- `distance-recognition-demo/backend/face_detection.py` - OpenCV Haar cascade face detection
- `distance-recognition-demo/backend/simulation.py` - Research-based accuracy simulation (contains performance models for 2-4m, 4-7m, 7-10m ranges)
- `distance-recognition-demo/backend/distance_estimation.py` - Distance calculation from face width using 14cm average face size

**Frontend (React):**
- `distance-recognition-demo/frontend/` - React app with webcam capture and real-time results display

## Development Commands

**Backend Setup:**
```bash
cd distance-recognition-demo/backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn opencv-python numpy pillow python-multipart
```

**Run Backend Server:**
```bash
cd distance-recognition-demo/backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend Setup:**
```bash
cd distance-recognition-demo/frontend
npm install
npm run dev
```

## Key Implementation Details

**Distance Estimation:**
- Uses face width in pixels with formula: `distance = (14cm * focal_length_pixels) / face_width_pixels`
- Default focal length: 800 pixels
- Distance ranges: 2-4m (close), 4-7m (medium), 7-10m (far)

**Performance Model:**
- Close (2-4m): 89.1% overall accuracy
- Medium (4-7m): 82.3% overall accuracy
- Far (7-10m): 72.3% overall accuracy
- Includes separate models for age, gender, ethnicity predictions

**Face Detection:**
- Uses OpenCV Haar cascades (`haarcascade_frontalface_default.xml`)
- Minimum face size: 60x60 pixels
- Returns bounding boxes as (x, y, width, height)

## API Endpoints

**POST /analyze-frame**
- Accepts image file upload
- Returns JSON with distance, predictions, confidence scores, and processing time
- Expected response format includes quality_score, distance_category, and accuracy expectations

## Current Development Status

- ✅ Backend environment setup
- ✅ simulation.py created with research data
- ⏳ face_detection.py (in progress)
- ⏳ main.py FastAPI server
- ⏳ React frontend components

## Common Issues

- Ensure virtual environment is activated before running backend
- OpenCV installation may require additional system dependencies on some platforms
- CORS errors: Check frontend origin in FastAPI CORS middleware

## Deployment

- Backend: Railway (Python/FastAPI)
- Frontend: Vercel (React)
- Environment variables needed: CORS origins for frontend domain