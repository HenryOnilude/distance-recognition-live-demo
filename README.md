# Distance Recognition Live Demo

<div align="center">

**Real-time face recognition system demonstrating how accuracy degrades with distance**

[![Live Demo](https://img.shields.io/badge/demo-live-success?style=for-the-badge)](https://distance-recognition-live-demo.vercel.app/)
[![Next.js](https://img.shields.io/badge/Next.js-16.0.7-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Railway](https://img.shields.io/badge/Railway-Deployed-0B0D0E?style=for-the-badge&logo=railway)](https://railway.app/)

[**🚀 Live Demo**](https://distance-recognition-live-demo.vercel.app/) | [**📖 Documentation**](#-tech-stack) | [**🤝 Contributing**](#-contributing)

</div>

---

## 📸 Demo

### 🎥 Live Application

> **[🚀 Try the Live Demo](https://distance-recognition-live-demo.vercel.app/)**

### Demo Video

![Demo](screenshots/demo.gif)

*Real-time face detection, distance estimation, and demographic predictions*

### Screenshots

<div align="center">

**Upload Interface**

![Homepage](screenshots/homepage.png)

**Analysis Results**

![Results](screenshots/results.png)

**Camera View**

![Camera](screenshots/camera.png)

</div>

---

## 🎯 Key Features

- ✨ **Distance-Adaptive Recognition**: Accuracy ranges from 89.1% (close) to 72.3% (far)
- 📹 **Real-time Processing**: Live webcam feed analysis with < 2s response time
- 📏 **Distance Estimation**: Calculates distance based on facial width using computer vision
- 👤 **Demographic Predictions**: Gender estimation with confidence scoring
- 📊 **Performance Visualization**: Interactive display of accuracy expectations by distance
- 🎨 **Minimalist UI**: Clean, professional interface built with Tailwind CSS

---

## 🛠️ Tech Stack

### Frontend
- **[Next.js 16.0.7](https://nextjs.org/)** - React framework with App Router & Turbopack
- **[React 19.1.0](https://react.dev/)** - Modern UI component library
- **[TypeScript 5](https://www.typescriptlang.org/)** - Type-safe JavaScript
- **[Tailwind CSS 4](https://tailwindcss.com/)** - Utility-first CSS framework
- **[Axios 1.12.2](https://axios-http.com/)** - HTTP client for API requests

### Backend
- **[Python 3.11](https://www.python.org/)** - Core programming language
- **[FastAPI](https://fastapi.tiangolo.com/)** - High-performance async web framework
- **[Uvicorn](https://www.uvicorn.org/)** - Lightning-fast ASGI server
- **[InsightFace](https://github.com/deepinsight/insightface)** - SCRFD face detection model (buffalo_s)
- **[TensorFlow 2.19.0](https://www.tensorflow.org/)** - Machine learning framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[NumPy](https://numpy.org/)** - Numerical computing

### Deployment
- **[Vercel](https://vercel.com/)** - Frontend hosting with automatic deployments
- **[Railway](https://railway.app/)** - Backend containerization & hosting
- **[Docker](https://www.docker.com/)** - Containerization for consistent deployments

---

## 📊 Performance Model

| Distance Range | Overall Accuracy | Typical Use Case |
|----------------|------------------|------------------|
| 2-4m (Close)   | **89.1%**        | Security checkpoints, access control |
| 4-7m (Medium)  | **82.3%**        | Retail monitoring, customer analytics |
| 7-10m (Far)    | **72.3%**        | Crowd surveillance, public safety |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Webcam     │  │  Image       │  │   Results    │      │
│  │   Capture    │  │  Upload      │  │   Display    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTPS/REST API
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   SCRFD      │→ │  Distance    │→ │   Gender     │      │
│  │ Detection    │  │ Estimation   │  │ Recognition  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Components

**Face Detection**: InsightFace SCRFD model for state-of-the-art face detection
**Distance Calculation**: Based on 14cm average face width using pinhole camera model
**Gender Recognition**: Advanced ensemble model with distance-aware confidence
**API Endpoint**: `/analyze-frame` for image processing with quality scoring

---

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

## 📡 API Reference

### `POST /analyze-frame`

Analyze an image for face recognition and distance estimation.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** Image file (JPEG, PNG, WebP, BMP)

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
      "confidence": 0.92,
      "decision": "high_confidence",
      "expected_accuracy": 0.891
    }
  },
  "expected_overall_accuracy": 0.891
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "mode": "insightface"
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

---

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
```

---

## 📈 Use Cases

- **🔒 Security Systems**: Adaptive authentication thresholds based on distance
- **🛍️ Retail Analytics**: Customer demographics analysis at various distances
- **🚪 Access Control**: Distance-aware entry management
- **🔬 Research**: Face recognition performance degradation studies
- **📊 Surveillance**: Crowd monitoring with accuracy awareness

---

## 🐛 Troubleshooting

### Backend Issues
- **Models not loading**: Ensure sufficient memory (8GB recommended for production)
- **Slow first request**: Models download on first API call (~30-60 seconds)
- **CORS errors**: Check `allow_origin_regex` in `main.py` includes your frontend URL

### Frontend Issues
- **"Failed to connect"**: Verify `NEXT_PUBLIC_API_URL` environment variable is set
- **Camera not working**: Grant browser permissions for camera access
- **Build errors**: Ensure Node.js 18+ and compatible Next.js version

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
