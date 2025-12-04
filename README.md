# Distance Recognition Live Demo

A real-time face recognition system that demonstrates how accuracy degrades with distance. This demo processes live webcam feeds to provide distance estimation and demographic predictions, showcasing the relationship between distance and recognition performance.

## 🎯 Key Features

- **Distance-Adaptive Recognition**: Accuracy decreases from 89.1% (close) to 72.3% (far)
- **Real-time Processing**: Live webcam feed analysis
- **Distance Estimation**: Calculates distance based on face width
- **Demographic Predictions**: Gender estimation with confidence scoring
- **Performance Visualization**: Shows accuracy expectations by distance range

## 🛠️ Tech Stack

### Frontend
- **Next.js 15.5.4** - React framework with Turbopack for fast builds
- **React 19.1.0** - UI component library
- **TypeScript 5** - Type-safe JavaScript
- **Tailwind CSS 4** - Utility-first CSS framework
- **Axios 1.12.2** - HTTP client for API requests

### Backend
- **Python 3.11** - Core programming language
- **FastAPI** - High-performance web framework
- **Uvicorn** - ASGI server
- **InsightFace** - SCRFD face detection model
- **OpenCV (cv2)** - Computer vision library
- **NumPy** - Numerical computing
- **Pillow** - Image processing
- **Custom Ensemble Model** - ResNet50 + MobileNetV2 + MultiScale CNN for gender recognition

## 📊 Performance Model

| Distance Range | Overall Accuracy | Use Case |
|----------------|------------------|----------|
| 2-4m (Close)   | 89.1%           | Security checkpoints |
| 4-7m (Medium)  | 82.3%           | Retail monitoring |
| 7-10m (Far)    | 72.3%           | Crowd surveillance |

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **Face Detection**: InsightFace SCRFD model
- **Distance Calculation**: Based on 14cm average face width
- **Gender Recognition**: Advanced ensemble model with distance-aware confidence
- **API Endpoint**: `/analyze-frame` for image processing

### Frontend (React)
- **Webcam Capture**: Real-time video stream
- **Results Display**: Distance, predictions, and confidence scores
- **Performance Indicators**: Visual accuracy expectations

## 🚀 Quick Start

### Backend Setup
```bash
cd distance-recognition-demo/backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install fastapi uvicorn opencv-python numpy pillow python-multipart
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd distance-recognition-demo/frontend
npm install
npm run dev
```

## 📡 API Usage

**POST /analyze-frame**
- Upload image file
- Returns JSON with distance, predictions, confidence scores, and processing time
- Includes quality score and accuracy expectations

## 🎛️ Technical Details

- **Distance Formula**: `distance = (14cm × focal_length_pixels) / face_width_pixels`
- **Default Focal Length**: 800 pixels
- **Minimum Face Size**: 60×60 pixels
- **Detection Method**: InsightFace SCRFD (State-of-the-art face detection)

## 🔧 Common Issues

- Ensure virtual environment is activated for backend
- OpenCV may require additional system dependencies
- Check CORS settings for frontend-backend communication

## 📈 Use Cases

- **Security Systems**: Adaptive thresholds based on distance
- **Retail Analytics**: Customer demographics at various distances
- **Access Control**: Distance-aware authentication
- **Research**: Face recognition performance analysis