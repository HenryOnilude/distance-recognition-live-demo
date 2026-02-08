# Distance Recognition Live Demo

Real-time face recognition system demonstrating how accuracy degrades with distance

Live Demo Next.js FastAPI Railway

🚀 [Live Demo](https://synchrocv.com) | 📖 Documentation | 🤝 Contributing

## 📸 Demo

### 🎥 Live Application
🚀 Try the [Live Demo](https://synchrocv.com)

Real-time face detection, distance estimation, and demographic predictions

## 🎯 Key Features

- **Distance-Adaptive Recognition:** Accuracy ranges from 89.1% (close) to 72.3% (far)
- **Real-time WebSocket Streaming:** High-performance video inference with binary JPEG transport
- **Self-Healing Connectivity:** Automatic connection recovery with exponential backoff strategies for handling network interruptions
- **Demographic Predictions:** Gender estimation with distance-aware confidence scoring
- **Performance Visualization:** Interactive display of accuracy expectations by distance
- **Minimalist UI:** Clean, professional interface built with Tailwind CSS

## 🛠️ Tech Stack

### Frontend
- **Next.js 16.0.7** - React framework with App Router & Turbopack
- **React 19.1.0** - Modern UI component library
- **TypeScript 5** - Type-safe JavaScript
- **Tailwind CSS 4** - Utility-first CSS framework
- **Native WebSockets** - Bi-directional real-time communication

### Backend
- **Python 3.11** - Core programming language
- **FastAPI** - High-performance async web framework
- **Uvicorn** - Lightning-fast ASGI server
- **InsightFace** - SCRFD face detection model (buffalo_s)
- **TensorFlow 2.19.0** - Machine learning framework
- **OpenCV** - Computer vision library

### Deployment
- **Vercel** - Frontend hosting with automatic deployments
- **Railway** - Backend containerization & hosting
- **Docker** - Containerization for consistent deployments

## 📊 Performance Model

| Distance Range | Overall Accuracy | Typical Use Case |
|---|---|---|
| 2-4m (Close) | 89.1% | Security checkpoints, access control |
| 4-7m (Medium) | 82.3% | Retail monitoring, customer analytics |
| 7-10m (Far) | 72.3% | Crowd surveillance, public safety |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Webcam     │  │    Image     │  │   Results    │      │
│  │   Stream     │  │   Upload     │  │   Display    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
└─────────│──────────────────│────────────────────────────────┘
          │                  │
          │ WebSocket        │ HTTPS/REST
          │ (Streaming)      │ (Static Upload)
          ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    SCRFD     │→ │   Distance   │→ │    Gender    │      │
│  │  Detection   │  │  Estimation  │  │ Recognition  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- **Face Detection:** InsightFace SCRFD model for state-of-the-art face detection
- **Distance Calculation:** Heuristic linear ratio based on face-to-image width proportion
- **Resilient Network Layer:** Custom hook implementing exponential backoff (1s, 2s, 4s...) to gracefully handle server restarts or network drops
- **API Endpoint:** Dual-protocol architecture supporting both WebSocket streaming (live) and REST (static upload)

## ⚠️ Limitations & Design Decisions

### Distance Estimation
The current implementation uses a heuristic linear ratio (`face_width / image_width → distance`) rather than a full Pinhole Camera Model.

**Why?** A proper implementation requires camera intrinsic calibration (e.g. checkerboard method), which creates high friction for a web demo.

**Trade-off:** Seamless UX over absolute accuracy. The distance metric serves as a visual feedback mechanism to demonstrate low-latency WebSocket transport — not as a production depth sensor.

### Focal Length
Assumes a static focal length since the browser MediaStream API (`getUserMedia`) does not expose camera intrinsics. Different cameras will affect accuracy.

### Primary Goal
This project was built to benchmark real-time WebSocket streaming architecture (binary JPEG transport, exponential backoff, backpressure control) — distance estimation is the use case, not the main contribution.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Git

### Local Development

**1. Clone the repository**
```bash
git clone https://github.com/HenryOnilude/distance-recognition-live-demo.git
cd distance-recognition-live-demo
```

**2. Backend Setup**
```bash
cd distance-recognition-demo/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend will be available at http://localhost:8000

**3. Frontend Setup**
```bash
cd distance-recognition-demo/frontend

# Install dependencies
npm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Run development server
npm run dev
```
Frontend will be available at http://localhost:3000

## 📡 API & Protocol Reference

### 1. Real-time Streaming (WebSocket)
**Endpoint:** `ws://<host>/ws/analyze-stream`

- **Client:** Sends binary JPEG frames
- **Server:** Returns JSON analysis of the frame
- **Recovery:** Frontend automatically attempts reconnection with exponential backoff (1s → 30s max)

### 2. Static Analysis (REST)
**Endpoint:** `POST /analyze-frame`

**Response:**
```json
{
  "success": true,
  "processing_time_ms": 1234,
  "distance_m": 3.5,
  "distance_category": "close",
  "quality_score": 0.85,
  "predictions": {
    "gender": {
      "predicted_class": "male",
      "confidence": 0.92
    }
  }
}
```

## 🎛️ Technical Details

### Configuration
- **Minimum Face Size:** 60×60 pixels
- **Detection Model:** InsightFace SCRFD buffalo_s (memory-optimized)
- **Max Image Size:** 4096×4096 pixels (auto-resized)
- **Processing Timeout:** 60 seconds
- **WebSocket Reconnection:** 10 attempts with exponential backoff (max 30s delay)

## 🔧 Deployment

### Frontend (Vercel)
```bash
# Environment variables needed:
NEXT_PUBLIC_API_URL=https://your-backend-url.up.railway.app
```

### Backend (Railway)
```
# Railway deploys from Dockerfile
# Set root directory to: distance-recognition-demo/backend
# Port: 8000
```

## 🐛 Troubleshooting

### Backend Issues
- **Models not loading:** Ensure sufficient memory (8GB recommended for production)
- **Slow first request:** Models download on first API call (~30-60 seconds)
- **CORS errors:** Check `allow_origins` in main.py includes your frontend URL

### Frontend Issues
- **"Reconnecting..." Loop:** Backend may be waking up from sleep (Railway free tier) - wait up to 3 minutes for automatic reconnection
- **Camera not working:** Grant browser permissions for camera access

### WebSocket Issues
- **Connection Failed:** Check if backend WebSocket endpoint is accessible at `/ws/analyze-stream`
- **Frequent Disconnects:** Network instability - automatic reconnection will handle this
- **Max Retries Reached:** Click "Retry Connection" button or check backend logs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

**Henry Onilude**
- GitHub: [@HenryOnilude](https://github.com/HenryOnilude)

## 🌟 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for state-of-the-art face detection
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- Vercel & Railway for hosting platforms
