# Distance Recognition Live Demo
Real-time face recognition system demonstrating how accuracy degrades with distance

<div align="center">

**Real-time face recognition system demonstrating how accuracy degrades with distance**

[![Live Demo](https://img.shields.io/badge/demo-live-success?style=for-the-badge)](https://synchrocv.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16.0.7-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Railway](https://img.shields.io/badge/Railway-Deployed-0B0D0E?style=for-the-badge&logo=railway)](https://railway.app/)

[**🚀 Live Demo**](https://synchrocv.com/) | [**📖 Documentation**](#-tech-stack) | [**🤝 Contributing**](#-contributing)

</div>

---

## 📸 Demo

### 🎥 Live Application

> **[🚀 Try the Live Demo](https://synchrocv.com/)**

### Demo Video

![Demo](screenshots/demo.gif)

*Real-time face detection, distance estimation, and demographic predictions*

### Screenshots

<div align="center">

**Upload Interface**
<img width="1111" height="754" alt="Screenshot 2025-12-10 at 02 31 22" src="https://github.com/user-attachments/assets/3bca23fb-c475-4f76-a6ab-5bed6a232985" />

**Analysis Results**

https://github.com/user-attachments/assets/469de057-402d-4a47-8ce4-660d46f3a83e


<img width="1312" height="783" alt="Screenshot 2025-12-10 at 15 03 20" src="https://github.com/user-attachments/assets/511f216d-c9cf-4eb9-8153-5ba46bf158b9" />

**Camera View**


https://github.com/user-attachments/assets/454423d9-5179-4913-9259-1443689bd5fc



</div>

---

## 🎯 Key Features

- **Distance-Adaptive Recognition:** Accuracy ranges from 89.1% (close) to 72.3% (far)
- **Real-time WebSocket Streaming:** High-performance video inference with binary JPEG transport
- **Self-Healing Connectivity:** Automatic connection recovery with exponential backoff strategies for handling network interruptions
- **Demographic Predictions:** Gender estimation with distance-aware confidence scoring
- **Performance Visualization:** Interactive display of accuracy expectations by distance
- **Minimalist UI:** Clean, professional interface built with Tailwind CSS

---

## 🛠️ Tech Stack

### Frontend
- **[Next.js 16.0.7](https://nextjs.org/)** - React framework with App Router & Turbopack
- **[React 19.1.0](https://react.dev/)** - Modern UI component library
- **[TypeScript 5](https://www.typescriptlang.org/)** - Type-safe JavaScript
- **[Tailwind CSS 4](https://tailwindcss.com/)** - Utility-first CSS framework
- **[Native WebSockets](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)** - Bi-directional real-time communication

### Backend
- **[Python 3.11](https://www.python.org/)** - Core programming language
- **[FastAPI](https://fastapi.tiangolo.com/)** - High-performance async web framework
- **[Uvicorn](https://www.uvicorn.org/)** - Lightning-fast ASGI server
- **[InsightFace](https://github.com/deepinsight/insightface)** - SCRFD face detection model (buffalo_s)
- **[TensorFlow 2.19.0](https://www.tensorflow.org/)** - Machine learning framework
- **[OpenCV](https://opencv.org/)** - Computer vision library

### Deployment
- **[Vercel](https://vercel.com/)** - Frontend hosting with automatic deployments
- **[Railway](https://railway.app/)** - Backend containerization & hosting
- **[Docker](https://www.docker.com/)** - Containerization for consistent deployments

---

## 📊 Performance Model

| Distance Range | Overall Accuracy | Typical Use Case |
|----------------|------------------|------------------|
| 2-4m (Close)   | **89.1%** | Security checkpoints, access control |
| 4-7m (Medium)  | **82.3%** | Retail monitoring, customer analytics |
| 7-10m (Far)    | **72.3%** | Crowd surveillance, public safety |

---

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

---

## 🧠 ML Model Training

### Dataset
- **CelebA** — 202,599 aligned celebrity face images
- **Labels used:** Binary gender (Male/Female) + age (Young/Old) from 40 attribute annotations
- **Train/validation split:** 80/20

### Models Trained
Three gender classification models were trained using transfer learning:

| Model | Architecture | Size | Accuracy (Close) | Accuracy (Far) |
|---|---|---|---|---|
| Fast | MobileNetV2 + attention | 22MB | ~85% | ~78% |
| Accurate | ResNet50 + attention | 238MB | ~90% | ~80% |
| Ensemble | ResNet50 + MobileNetV2 + VGG16 multi-scale | 513MB | ~91% | ~82% |

### Training Techniques
- **Transfer Learning:** ImageNet pre-trained backbones with two-phase fine-tuning
- **ArcFace Loss:** Additive angular margin loss for improved gender discrimination
- **Focal Loss:** Handles class imbalance by focusing on hard examples
- **Spatial & Channel Attention:** Learns to focus on gender-discriminative facial regions
- **Distance Degradation Simulation:** Training augmentation simulating resolution loss, Gaussian blur, and noise at increasing distances to make predictions robust at range

### Why the Accuracy Degrades with Distance
The accuracy figures in the performance table are derived from model evaluation under simulated distance conditions. As distance increases, face resolution drops, blur increases, and signal quality degrades — the models were explicitly trained and evaluated against these conditions.

## 📊 Performance Model

| Distance Range | Overall Accuracy | Typical Use Case |
|---|---|---|
| 2-4m (Close) | 89.1% | Security checkpoints, access control |
| 4-7m (Medium) | 82.3% | Retail monitoring, customer analytics |
| 7-10m (Far) | 72.3% | Crowd surveillance, public safety | 

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Git

### Local Development

#### 1. Clone the repository
```bash
git clone https://github.com/HenryOnilude/distance-recognition-live-demo.git
cd distance-recognition-live-demo
```

#### 2. Backend Setup
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

Backend will be available at `http://localhost:8000`

#### 3. Frontend Setup
```bash
cd distance-recognition-demo/frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

---



## 📡 API & Protocol Reference

### 1. Real-time Streaming (WebSocket)

**Endpoint:** `ws://<host>/ws/analyze-stream`

Designed for high-throughput live video inference.

- **Client:** Sends binary JPEG frames
- **Server:** Returns JSON analysis of the frame including bounding boxes and demographics
- **Recovery:** Frontend automatically attempts reconnection with exponential backoff (1s → 30s max)

### 2. Static Analysis (REST)

**Endpoint:** `POST /analyze-frame`

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** Image file (JPEG, PNG, WebP)

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

---

## 🎛️ Technical Details

### Distance Estimation Formula
```
distance = (14cm × focal_length_pixels) / face_width_pixels
```

### Configuration
- **Default Focal Length**: 800 pixels (calibrated for standard webcams)
- **Minimum Face Size**: 60×60 pixels
- **Detection Model**: InsightFace SCRFD buffalo_s (memory-optimized)
- **Max Image Size**: 4096×4096 pixels (auto-resized)
- **Processing Timeout**: 60 seconds
- **WebSocket Reconnection**: 10 attempts with exponential backoff (max 30s delay)

---

## ⚠️ Limitations & Design Decisions

### Distance Estimation
The current implementation uses a heuristic linear ratio (`face_width / image_width → distance`) rather than a full Pinhole Camera Model.

**Why?** A proper implementation requires camera intrinsic calibration (e.g. checkerboard method), which creates high friction for a web demo.

**Trade-off:** Seamless UX over absolute accuracy. The distance metric serves as a visual feedback mechanism to demonstrate low-latency WebSocket transport — not as a production depth sensor.

### Focal Length
Assumes a static focal length since the browser MediaStream API (`getUserMedia`) does not expose camera intrinsics. Different cameras will affect accuracy.

### Primary Goal
This project was built to benchmark real-time WebSocket streaming architecture (binary JPEG transport, exponential backoff, backpressure control) — distance estimation is the use case, not the main contribution.

## 🔧 Deployment

### Frontend (Vercel)
```bash
# Vercel will automatically deploy from main branch
# Environment variables needed:
# NEXT_PUBLIC_API_URL=https://your-backend-url.up.railway.app
```

### Backend (Railway)
```bash
# Railway deploys from Dockerfile
# Set root directory to: distance-recognition-demo/backend
# Port: 8000
# WebSocket support: Enabled by default
```

---

## 🎛️ Technical Details

### Configuration
- **Minimum Face Size:** 60×60 pixels
- **Detection Model:** InsightFace SCRFD buffalo_s (memory-optimized)
- **Max Image Size:** 4096×4096 pixels (auto-resized)
- **Processing Timeout:** 60 seconds
- **WebSocket Reconnection:** 10 attempts with exponential backoff (max 30s delay)
## 📈 Use Cases

- **🔒 Security Systems**: Adaptive authentication thresholds based on distance
- **🛍️ Retail Analytics**: Customer demographics analysis at various distances
- **🚪 Access Control**: Distance-aware entry management
- **🔬 Research**: Face recognition performance degradation studies
- **📊 Surveillance**: Crowd monitoring with accuracy awareness

---

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
---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Henry Onilude**

- GitHub: [@HenryOnilude](https://github.com/HenryOnilude)
- LinkedIn: [Henry Onilude](https://linkedin.com/in/henryonilude)

---

## 🌟 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for state-of-the-art face detection
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Vercel](https://vercel.com/) & [Railway](https://railway.app/) for hosting platforms

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

[![Star this repo](https://img.shields.io/github/stars/HenryOnilude/distance-recognition-live-demo?style=social)](https://github.com/HenryOnilude/distance-recognition-live-demo)

</div>
