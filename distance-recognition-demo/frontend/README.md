# Distance Recognition Live Demo

A real-time system to estimate the distance of a face from a webcam without using depth sensors (like LiDAR or FaceID).

Live Demo: [synchrocv.com](https://synchrocv.com)

## What It Does

Streams your webcam → Detects faces using InsightFace (SCRFD) → Estimates distance from camera → Updates in real-time (~10fps)

Also supports static image upload via REST API.

## The Stack

- **Frontend:** Next.js 16 (TypeScript + TailwindCSS)
- **Backend:** FastAPI + InsightFace + OpenCV
- **Model:** InsightFace buffalo_s with SCRFD face detector
- **Transport:** Native WebSockets (binary JPEG streaming)
- **Deployment:** Vercel (frontend) + Railway (backend)

## Technical Highlights

- Self-healing WebSocket with exponential backoff (1s → 30s)
- Backpressure control (doesn't flood server with frames)
- Dual-protocol architecture (WebSocket for streaming, REST for uploads)
- SCRFD face detection with landmark analysis

## Performance

- Inference: ~1.5s per frame (Railway CPU)
- Accuracy: 2-10m range (89% at 2-4m, 72% at 7-10m)

## Limitations & Design Decisions

### Distance Estimation
The current implementation uses a heuristic linear ratio (face_width / image_width → distance) rather than a full Pinhole Camera Model.

**Why?** A proper implementation requires camera intrinsic calibration (e.g. checkerboard method), which creates high friction for a web demo.

**Trade-off:** Seamless UX over absolute accuracy. The distance metric serves as a visual feedback mechanism to demonstrate low-latency WebSocket transport — not as a production depth sensor.

### Focal Length
Assumes a static focal length since the browser MediaStream API (getUserMedia) does not expose camera intrinsics. Different cameras will affect accuracy.

### Primary Goal
This project was built to benchmark real-time WebSocket streaming architecture (binary JPEG transport, exponential backoff, backpressure control) — distance estimation is the use case, not the main contribution.

## Local Development

**Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Environment Variables

Create `frontend/.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production, set `NEXT_PUBLIC_API_URL` in your Vercel environment variables.
