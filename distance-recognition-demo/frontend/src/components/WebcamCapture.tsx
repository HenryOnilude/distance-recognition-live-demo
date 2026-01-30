'use client'

import { useRef, useEffect, useState } from 'react'

interface WebcamCaptureProps {
  onImageCapture: (imageBlob: Blob) => void
  isProcessing: boolean
}

export default function WebcamCapture({ onImageCapture, isProcessing }: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isActive, setIsActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const startCamera = async () => {
    try {
      console.log('Starting camera...')
      setError(null)

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
          facingMode: 'user'
        }
      })

      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setIsActive(true)
        console.log('Camera active!')
      }
    } catch (err) {
      console.error('Camera error:', err)
      setError(err instanceof Error ? err.message : 'Camera error occurred')
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setIsActive(false)
    console.log('Camera stopped')
  }

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current || isProcessing || !isActive) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    if (!ctx || video.videoWidth === 0) return

    // OPTIMIZATION: Resize to 640px (Model native resolution)
    const scaleFactor = 640 / video.videoWidth
    const newHeight = video.videoHeight * scaleFactor

    canvas.width = 640
    canvas.height = newHeight

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    canvas.toBlob((blob) => {
      if (blob) {
        console.log('Frame captured, blob size:', blob.size, 'bytes')
        onImageCapture(blob)
      }
    }, 'image/jpeg', 0.85)
  }

  // Auto-capture every 2 seconds when active
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null

    if (isActive && !isProcessing) {
      interval = setInterval(captureFrame, 2000)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isActive, isProcessing])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="relative">
      {/* Video Element */}
      <div className="relative bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full object-cover"
          style={{
            height: '500px',
            backgroundColor: '#000000'
          }}
        />

        {/* Processing overlay */}
        {isProcessing && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="text-white text-sm">Processing...</div>
          </div>
        )}
      </div>

      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Controls */}
      <div className="mt-6 flex gap-3">
        {!isActive ? (
          <button
            onClick={startCamera}
            className="px-6 py-2 bg-slate-900 hover:bg-slate-800 text-white text-sm"
          >
            Start Camera
          </button>
        ) : (
          <>
            <button
              onClick={captureFrame}
              disabled={isProcessing}
              className="px-6 py-2 bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 text-white text-sm disabled:cursor-not-allowed"
            >
              Capture
            </button>
            <button
              onClick={stopCamera}
              className="px-6 py-2 border border-slate-300 hover:bg-slate-50 text-slate-900 text-sm"
            >
              Stop
            </button>
          </>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-3 border border-slate-300 text-sm text-slate-900">
          {error}
        </div>
      )}
    </div>
  )
}