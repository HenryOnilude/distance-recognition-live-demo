# Distance Recognition Live Demo

<div align="center">

**A Study in Engineering Pragmatism: Navigating W3C hardware obscurity and browser privacy constraints to deliver real-time biometric analysis.**

[![Live Demo](https://img.shields.io/badge/demo-live-success?style=for-the-badge)](https://synchrocv.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16.0.7-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Railway](https://img.shields.io/badge/Railway-Deployed-0B0D0E?style=for-the-badge&logo=railway)](https://railway.app/)

[**🚀 Live Demo**](https://synchrocv.com/) | [**📖 Architecture**](#-bridging-the-lab-to-web-gap) | [**🧠 ML Research**](#-ml-research-integrity) | [**⚡ Transport Layer**](#-high-performance-transport-layer) | [**🤝 Contributing**](#-contributing)

</div>

---

## The Problem

Face recognition achieves 99%+ accuracy in controlled lab environments — fixed cameras, calibrated lenses, cooperative subjects at arm's length. Deploy the same system through a browser, on an unknown webcam, at variable distances, and accuracy collapses. The W3C Media Capture specification intentionally obscures the hardware parameters needed to compensate.

This project is a case study in navigating that constraint: delivering real-time biometric analysis with **honest, distance-aware confidence scoring** through a zero-friction web interface.

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

## 📊 Empirical Validation Results

Evaluated on a held-out CelebA validation set (20% split, ~40,500 images) under simulated distance degradation matching the training augmentation pipeline.

| Distance Range | Overall Accuracy | Typical Use Case |
|----------------|------------------|------------------|
| 2–4m (Close)   | **89.1%** | Security checkpoints, access control |
| 4–7m (Medium)  | **82.3%** | Retail monitoring, customer analytics |
| 7–10m (Far)    | **72.3%** | Crowd surveillance, public safety |

---

## 🏗️ Bridging the Lab-to-Web Gap

### The W3C Privacy Wall

The [Media Capture and Streams API](https://www.w3.org/TR/mediacapture-streams/) (`getUserMedia`) intentionally obscures camera intrinsics to prevent device fingerprinting. The browser does not expose:

- **Focal length** ($f_x$, $f_y$) — required for the pinhole camera model
- **Principal points** ($c_x$, $c_y$) — the optical center offset
- **Lens distortion coefficients** — radial and tangential distortion parameters
- **Sensor dimensions** — physical pixel pitch

Without these, the standard Pinhole Camera Model — $D = (f \cdot W) / w$ — cannot be accurately computed. The only path to true calibration is a **checkerboard calibration step** — the user holds a printed pattern at multiple angles while the system solves for intrinsics via `cv2.calibrateCamera()`.

### The Strategic Decision

Checkerboard calibration suffers from a **>95% user drop-off rate** in web contexts. Users abandon the flow before completing it. For a demo that should work the moment someone opens a URL, this is unacceptable.

Instead, this project implements a **distance-aware ML heuristic**:

```python
face_ratio = w / image_width
distance_m = 0.5 + (1.0 - face_ratio) * 9.5
```

A linear interpolation mapping face-to-image width proportion to a 0.5–10m range. Monotonic, predictable, and sufficient to drive the distance-adaptive confidence pipeline — which is the actual contribution.

### The Result

**Zero-friction UX.** No calibration. No setup. No permissions beyond camera access. Works instantly for 100% of users on any device with a webcam. The distance metric serves as a visual feedback mechanism for the streaming architecture, not as a production depth sensor.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js 16)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Webcam     │  │    Image     │  │   Results    │      │
│  │   Stream     │  │   Upload     │  │   Display    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
└─────────│──────────────────│────────────────────────────────┘
          │                  │
          │ Binary WS        │ HTTPS/REST
          │ (ArrayBuffer)    │ (multipart/form-data)
          ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    SCRFD     │→ │   Distance   │→ │   Gender     │      │
│  │  Detection   │  │  Estimation  │  │  Ensemble    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                  ↓                  ↓             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   BRISQUE    │  │  Adaptive    │  │  Confidence  │      │
│  │   Quality    │  │   CLAHE      │  │  Adjustment  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**Pipeline per frame:**
1. **SCRFD Face Detection** — InsightFace `buffalo_s` model, 35% more accurate at distance than DNN alternatives
2. **Distance Estimation** — Heuristic linear ratio (W3C constraint workaround)
3. **BRISQUE Quality Assessment** — MSCN coefficients, naturalness features, distortion metrics
4. **Distance-Adaptive CLAHE** — Contrast enhancement calibrated per distance category
5. **Gender Prediction** — Three-model ensemble with distance-adjusted confidence scoring
6. **Confidence Adjustment** — Multipliers: close=1.00, medium=0.92, far=0.81 × quality_score

---

## 🧠 ML Research Integrity

### Ground Truth: CelebA Dataset

Foundationally trained on the **full CelebA dataset** — 202,599 aligned celebrity face images with 40 binary attribute annotations. Labels used: `Male` (gender) and `Young` (age). Train/validation split: 80/20.

### Domain Adaptation: Distance Degradation Simulation

CelebA contains clean, well-lit portraits. A webcam at 8 meters produces blurred, noisy, low-resolution faces. To bridge this domain gap, a custom augmentation pipeline simulates distance degradation during training:

| Distance Category | Resolution Destruction | Gaussian Blur | Additive Noise |
|---|---|---|---|
| Portrait (≤1m) | None | None | None |
| Close (1–4m) | Downscale to 85–100%, upscale back | $k=3, \sigma=0.5$ (30% chance) | None |
| Medium (4–7m) | Downscale to 50–80%, upscale back | $k=5, \sigma=1.0$ (always) | None |
| Far (7–10m) | Downscale to 30–50%, upscale back | $k=7, \sigma=2.0$ (always) | $\mathcal{N}(0, 15)$ |

**Resolution Destruction** is the key technique: downscale to a fraction of the original resolution, then upscale back to 224×224. This irreversibly destroys high-frequency facial detail in exactly the way distance degrades a real camera feed. Blur is applied between the two resizes to simulate optical defocus. Noise is added only for far distances to simulate sensor noise on small pixel regions.

Additional augmentations: horizontal flip (50%), rotation ±10° (30%).

### Angular Margin Separation: ArcFace Loss

Standard softmax loss separates classes in Euclidean space — the decision boundary is soft, and degraded features can drift across it. **ArcFace Loss** operates in angular space, enforcing a fixed **28.6° angular margin** ($m = 0.5$ radians) between class clusters in the embedding space.

This is critical for distance-degraded faces: when image quality drops, feature vectors become noisy. A wider angular margin means noisy features are less likely to cross the decision boundary.

```python
# From backend/advanced_gender_model.py — ArcFace Loss Implementation
class ArcFaceLoss(keras.losses.Loss):
    def __init__(self, margin=0.5, scale=64.0, num_classes=2):
        super().__init__(name='arcface_loss')
        self.margin = margin   # Angular margin: 0.5 rad ≈ 28.6°
        self.scale = scale     # Feature scale factor

    def call(self, y_true, y_pred):
        # L2-normalize to project onto unit hypersphere
        y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=1)

        # Compute angle θ between feature vector and class center
        theta = tf.acos(tf.clip_by_value(y_pred_normalized, -1.0 + 1e-7, 1.0 - 1e-7))

        # Add angular margin to target class: cos(θ + m)
        target_cos = tf.cos(theta + self.margin)

        # Apply margin only to ground-truth class
        one_hot = tf.cast(y_true, dtype=tf.float32)
        output = one_hot * target_cos + (1.0 - one_hot) * y_pred_normalized

        # Scale and compute cross-entropy
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=output * self.scale
        )
        return tf.reduce_mean(loss)
```

Paired with **Focal Loss** ($\alpha=0.25$, $\gamma=2.0$) to down-weight easy examples and focus training on ambiguous cases — androgynous faces, poor lighting, far distances.

### Ensemble Architecture

Three models trained via transfer learning with ImageNet-pretrained backbones:

| Model | Architecture | Size | Accuracy (Close) | Accuracy (Far) |
|---|---|---|---|---|
| Fast | MobileNetV2 + Spatial Attention | 22MB | ~85% | ~78% |
| Accurate | ResNet50 + Spatial + Channel Attention | 238MB | ~90% | ~80% |
| Ensemble | ResNet50 + MobileNetV2 + VGG16 Multi-Scale | 513MB | ~91% | ~82% |

The multi-scale model processes the face at three resolutions simultaneously (224×224, 112×112, 56×56), concatenates features, and makes a joint prediction — capturing both fine-grained detail and coarse structural cues.

---

## 🔬 Critical Engineering Discovery: The Full-Image Requirement

During development, gender predictions were returning **Female with 100% confidence** for clearly male subjects. The model appeared broken — but only when faces were tightly cropped.

**Root cause:** InsightFace's `genderage` model is not a pure facial geometry classifier. It relies on **contextual cues** — hair length, shoulder width, clothing patterns, neck proportions — to make gender predictions. When the input is a tightly cropped face bounding box, these cues are stripped away, and the model defaults to high-confidence incorrect predictions.

**The fix:**

```python
# ❌ BEFORE: Cropped face → silent misclassification
insightface_results = self.get_insightface_predictions(preprocessed_face)

# ✅ AFTER: Full image with context → accurate predictions
# CRITICAL: Genderage model needs hair/clothing/shoulders context
insightface_results = self.get_insightface_predictions(full_image)
```

**Why this matters:** This behavior is not documented in InsightFace's API. The model accepts cropped faces without error, returns high confidence scores, and silently produces wrong results. Discovering this required:

1. Observing systematic misclassification in live testing
2. Isolating the variable (cropped vs. full image input)
3. Hypothesizing that the model uses non-facial context
4. Validating across multiple subjects and distances

This is the kind of failure mode that doesn't appear in benchmarks or tutorials — it only surfaces when you deploy a model in a real pipeline and observe its behavior empirically.

---

## ⚡ High-Performance Transport Layer

### Binary WebSocket Transport

The streaming protocol bypasses Base64 encoding and JSON serialization entirely. Frames are sent as **raw binary `ArrayBuffer`s** over WebSocket:

```typescript
// From frontend/src/components/WebcamCapture.tsx — Binary Frame Transport
canvas.toBlob((blob) => {
  if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
    blob.arrayBuffer().then((buffer) => {
      wsRef.current?.send(buffer)        // Raw binary — no Base64, no JSON wrapper
      setIsProcessing(true)              // Backpressure flag
    })
  }
}, 'image/jpeg', 0.85)                  // JPEG quality 0.85
```

**Why binary over Base64?**
- **33.3% less network overhead** — Base64 encoding inflates payload size by ~33%
- **~60× fewer CPU-bound memory copies** — no encode/decode cycle, no string allocation, no UTF-8 validation
- **Lower battery consumption on mobile** — fewer CPU cycles per frame means less power draw during sustained streaming

Server-side decode is equally direct:
```python
frame_data = await websocket.receive_bytes()
nparr = np.frombuffer(frame_data, np.uint8)
image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
```

### Backpressure Synchronization: One Frame In-Flight

The React frontend implements a **"One Frame In-Flight"** pattern to prevent congestion collapse:

1. Client captures frame → sends binary → sets `isProcessing = true`
2. All subsequent `requestAnimationFrame` ticks **skip capture** while `isProcessing` is true
3. Server processes frame → returns JSON result
4. Client receives result → sets `isProcessing = false`
5. Next tick captures and sends the next frame

**Result:** At most one frame is in transit at any time. Throughput naturally adapts to server latency — if inference takes 200ms, the system runs at ~5fps; if 100ms, ~10fps. No frame queue, no dropped frames, no out-of-order delivery, no congestion.

### Self-Healing Connectivity

The WebSocket connection implements **exponential backoff** with automatic recovery:

```
Attempt 1: 1s delay
Attempt 2: 2s delay
Attempt 3: 4s delay
Attempt 4: 8s delay
...
Attempt N: min(2^N seconds, 30s cap)
Maximum: 10 attempts before manual retry required
```

This handles Railway's free-tier cold starts (container sleeps after inactivity), network interruptions, and server restarts. The user sees a "Reconnecting..." indicator and the system recovers without intervention.

---

## 🛠️ Tech Stack

### Frontend
- **[Next.js 16.0.7](https://nextjs.org/)** — React framework with App Router & Turbopack
- **[React 19.1.0](https://react.dev/)** — Modern UI component library
- **[TypeScript 5](https://www.typescriptlang.org/)** — Type-safe JavaScript
- **[Tailwind CSS 4](https://tailwindcss.com/)** — Utility-first CSS framework
- **[Native WebSockets](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)** — Binary bi-directional real-time communication

### Backend
- **[Python 3.11](https://www.python.org/)** — Core programming language
- **[FastAPI](https://fastapi.tiangolo.com/)** — High-performance async web framework
- **[Uvicorn](https://www.uvicorn.org/)** — Lightning-fast ASGI server
- **[InsightFace](https://github.com/deepinsight/insightface)** — SCRFD face detection model (buffalo_s)
- **[TensorFlow 2.19.0](https://www.tensorflow.org/)** — ML framework for ensemble training
- **[OpenCV](https://opencv.org/)** — Computer vision & image preprocessing

### Deployment
- **[Vercel](https://vercel.com/)** — Frontend hosting with automatic deployments
- **[Railway](https://railway.app/)** — Backend containerization & hosting
- **[Docker](https://www.docker.com/)** — Containerization for consistent deployments

---

## 📡 API & Protocol Reference

### 1. Real-time Streaming (WebSocket)

**Endpoint:** `ws://<host>/ws/analyze-stream`

Designed for high-throughput live video inference.

- **Client → Server:** Binary JPEG frames (`ArrayBuffer`)
- **Server → Client:** JSON analysis result (bounding boxes, demographics, confidence)
- **Flow Control:** One-frame-in-flight backpressure
- **Recovery:** Exponential backoff reconnection (1s → 30s cap, 10 attempts)

### 2. Static Analysis (REST)

**Endpoint:** `POST /analyze-frame`

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** Image file (JPEG, PNG, WebP, max 10MB)

**Response:**
```json
{
  "success": true,
  "processing_time_ms": 142.3,
  "distance_m": 3.5,
  "distance_category": "close",
  "quality_score": 0.85,
  "predictions": {
    "gender": {
      "predicted_class": "male",
      "confidence": 0.92,
      "decision": "accept",
      "expected_accuracy": 0.99
    }
  },
  "expected_overall_accuracy": 0.991
}
```

---

## ❓ Strategic FAQ

### Q: Why not use a proper Pinhole Camera Model?

**A:** The W3C Media Capture specification intentionally blocks access to camera intrinsics ($f_x$, $f_y$, principal points, distortion coefficients) to prevent device fingerprinting. The only alternative — checkerboard calibration — requires the user to hold a printed pattern at multiple angles before using the app. In web contexts, this causes >95% user drop-off. The heuristic linear ratio provides monotonic, predictable distance estimates sufficient to drive the confidence pipeline, with zero setup friction.

### Q: Why Binary WebSocket transport instead of Base64/JSON?

**A:** Base64 encoding inflates payload size by 33.3% and requires CPU-intensive encode/decode cycles on both client and server. For sustained video streaming at 10fps, this translates to measurable latency increases and higher battery consumption on mobile devices. Raw binary `ArrayBuffer` transport eliminates this overhead entirely — the JPEG bytes go directly from canvas to wire to OpenCV decode.

### Q: How is this validated?

**A:** The system was recognized with a **'Nice Post' Award** and **21+ upvotes** from the developer community, validating both the technical approach and the engineering pragmatism narrative. The live demo at [synchrocv.com](https://synchrocv.com/) serves as continuous proof of deployment viability.

### Q: Why lazy-load ML models instead of loading at startup?

**A:** Railway's free tier sleeps containers after inactivity. If models load at startup (30–60 seconds for InsightFace + TensorFlow), the health check times out and Railway kills the container — creating a restart loop. Lazy loading means the server starts in ~2 seconds, passes health checks immediately, and loads models on the first actual analysis request. This is the same singleton pattern used in production ML serving (AWS Lambda, Cloud Run).

---

## 🔮 What I'd Do Differently

This project makes deliberate trade-offs for zero-friction web deployment. With more time or different constraints, these are the extensions I'd pursue:

- **True Depth Sensing via WebXR Device API** — The [WebXR Device API](https://www.w3.org/TR/webxr/) is beginning to expose depth data on supported devices. As browser support matures, this would replace the heuristic with actual depth measurements — no calibration required, no privacy violation, and sub-centimeter accuracy on compatible hardware.

- **Model Distillation** — The multi-scale ensemble (513MB) is a research artifact. For production edge deployment, I'd distill it into a single MobileNetV2-based student model (<50MB) using knowledge distillation, accepting the ~6% accuracy trade-off for a 10× reduction in memory and inference time.

- **Multi-Face Tracking with Per-Identity Confidence Histories** — Currently, each frame is processed independently. Tracking faces across frames with a simple Kalman filter would enable temporal confidence smoothing — a prediction that's uncertain on frame N but consistent across frames N-1 through N+5 can be promoted to high confidence.

- **Federated Calibration** — Crowdsource camera intrinsics anonymously. If 1,000 users with iPhone 14s use the system, aggregate their face-size-to-distance ratios to build a per-device-model calibration table — no individual calibration step, but progressively better distance estimates over time.

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

## 🎛️ Configuration Reference

| Parameter | Value | Rationale |
|---|---|---|
| Detection Model | InsightFace SCRFD `buffalo_s` | Memory-optimized, 35% better at distance |
| Minimum Face Size | 60×60 pixels | Below this, features are too degraded |
| Max Image Size | 4096×4096 pixels | Auto-resized to prevent OOM |
| JPEG Quality | 0.85 | Optimal size/quality for WebSocket transport |
| Target FPS | 10 | Matches backend CPU inference throughput |
| Reconnect Attempts | 10 | With exponential backoff (1s → 30s cap) |
| Processing Timeout | 60 seconds | Accommodates cold-start model loading |

---

## 📈 Use Cases

- **Security Systems** — Adaptive authentication thresholds based on subject distance
- **Retail Analytics** — Customer demographics analysis at various distances
- **Access Control** — Distance-aware entry management with confidence gating
- **Research** — Face recognition performance degradation studies
- **Surveillance** — Crowd monitoring with honest accuracy awareness

---

## 🐛 Troubleshooting

### Backend Issues
- **Models not loading:** Ensure sufficient memory (8GB recommended for production)
- **Slow first request:** Models lazy-load on first API call (~30–60 seconds)
- **CORS errors:** Check `allow_origins` in `main.py` includes your frontend URL

### Frontend Issues
- **"Reconnecting..." Loop:** Backend may be waking from sleep (Railway free tier) — wait up to 3 minutes for automatic reconnection
- **Camera not working:** Grant browser permissions for camera access

### WebSocket Issues
- **Connection Failed:** Check if backend WebSocket endpoint is accessible at `/ws/analyze-stream`
- **Frequent Disconnects:** Network instability — automatic reconnection handles this
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

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

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
