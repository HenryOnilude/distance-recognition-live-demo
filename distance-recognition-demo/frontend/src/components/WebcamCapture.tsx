'use client'

import { useRef, useEffect, useState, useCallback } from 'react'

interface AnalysisResult {
  success?: boolean
  processing_time_ms?: number
  distance_m?: number
  distance_category?: string
  quality_score?: number
  predictions?: any
  expected_overall_accuracy?: number
  error?: string
  frame_count?: number
}

interface WebcamCaptureProps {
  onAnalysisResult: (result: AnalysisResult) => void
  isProcessing: boolean
  setIsProcessing: (processing: boolean) => void
}

export default function WebcamCapture({ onAnalysisResult, isProcessing, setIsProcessing }: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isActive, setIsActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')
  const [fps, setFps] = useState(0)
  const [reconnectAttempt, setReconnectAttempt] = useState(0)
  const [isReconnecting, setIsReconnecting] = useState(false)

  const streamRef = useRef<MediaStream | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const lastFrameTimeRef = useRef<number>(0)
  const frameCountRef = useRef<number>(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const shouldReconnectRef = useRef<boolean>(false)

  // Reconnection configuration
  const MAX_RECONNECT_ATTEMPTS = 10
  const INITIAL_RECONNECT_DELAY = 1000 // 1 second
  const MAX_RECONNECT_DELAY = 30000 // 30 seconds

  // Calculate exponential backoff delay
  const calculateBackoffDelay = (attempt: number): number => {
    const delay = INITIAL_RECONNECT_DELAY * Math.pow(2, attempt)
    return Math.min(delay, MAX_RECONNECT_DELAY)
  }

  // Attempt reconnection with exponential backoff
  const attemptReconnection = useCallback(() => {
    // Don't reconnect if we shouldn't or if max attempts reached
    if (!shouldReconnectRef.current) {
      console.log('⏸️ Reconnection disabled (camera stopped)')
      return
    }

    if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      console.log('❌ Max reconnection attempts reached')
      setConnectionStatus('disconnected')
      setIsReconnecting(false)
      setError('Connection failed. Please retry manually.')
      return
    }

    const delay = calculateBackoffDelay(reconnectAttempt)
    console.log(`🔄 Reconnecting in ${delay}ms (attempt ${reconnectAttempt + 1}/${MAX_RECONNECT_ATTEMPTS})`)

    setIsReconnecting(true)
    setConnectionStatus('connecting')

    // Clear any existing timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    // Schedule reconnection
    reconnectTimeoutRef.current = setTimeout(() => {
      setReconnectAttempt(prev => prev + 1)
      connectWebSocket()
    }, delay)
  }, [reconnectAttempt, MAX_RECONNECT_ATTEMPTS])

  // WebSocket connection handler
  const connectWebSocket = useCallback(() => {
    // Cleanup: Close existing connection if any
    if (wsRef.current) {
      console.log('🧹 Cleaning up existing WebSocket connection')
      wsRef.current.onclose = null // Remove old event listeners
      wsRef.current.onerror = null
      wsRef.current.close()
      wsRef.current = null
    }

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const wsUrl = apiUrl.replace('http://', 'ws://').replace('https://', 'wss://')

    console.log('🔌 Connecting to WebSocket:', `${wsUrl}/ws/analyze-stream`)
    setConnectionStatus('connecting')

    const ws = new WebSocket(`${wsUrl}/ws/analyze-stream`)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      console.log('✅ WebSocket connected')
      setConnectionStatus('connected')
      setError(null)

      // Reset reconnection state on successful connection
      setReconnectAttempt(0)
      setIsReconnecting(false)
    }

    ws.onmessage = (event) => {
      try {
        const result: AnalysisResult = JSON.parse(event.data)

        // Update FPS counter
        const now = performance.now()
        if (lastFrameTimeRef.current > 0) {
          const delta = now - lastFrameTimeRef.current
          const currentFps = 1000 / delta
          setFps(Math.round(currentFps * 10) / 10)
        }
        lastFrameTimeRef.current = now
        frameCountRef.current++

        // Pass result to parent
        onAnalysisResult(result)

        // Clear processing flag to allow next frame
        setIsProcessing(false)

        console.log(`📊 Frame ${result.frame_count} processed in ${result.processing_time_ms}ms`)
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err)
        setIsProcessing(false)
      }
    }

    ws.onerror = (error) => {
      console.error('❌ WebSocket error:', error)
      setError('WebSocket connection error')
      setConnectionStatus('disconnected')

      // Trigger automatic reconnection
      if (shouldReconnectRef.current) {
        console.log('🔄 Connection error detected, attempting reconnection...')
        attemptReconnection()
      }
    }

    ws.onclose = (event) => {
      console.log('🔌 WebSocket closed', event.code, event.reason)
      setConnectionStatus('disconnected')

      // Only auto-reconnect if it wasn't an intentional disconnect
      // Code 1000 = normal closure, 1001 = going away
      if (shouldReconnectRef.current && event.code !== 1000 && event.code !== 1001) {
        console.log('🔄 Unexpected disconnect, attempting reconnection...')
        attemptReconnection()
      }
    }

    wsRef.current = ws
  }, [onAnalysisResult, setIsProcessing, attemptReconnection])

  const disconnectWebSocket = useCallback(() => {
    // Clear reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    // Disable auto-reconnection
    shouldReconnectRef.current = false

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.onclose = null // Prevent reconnection trigger
      wsRef.current.onerror = null
      wsRef.current.close()
      wsRef.current = null
    }

    // Reset state
    setConnectionStatus('disconnected')
    setReconnectAttempt(0)
    setIsReconnecting(false)
  }, [])

  const startCamera = async () => {
    try {
      console.log('📹 Starting camera...')
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
        console.log('✅ Camera active!')

        // Enable auto-reconnection
        shouldReconnectRef.current = true

        // Reset reconnection state
        setReconnectAttempt(0)
        setIsReconnecting(false)

        // Connect WebSocket after camera starts
        connectWebSocket()
      }
    } catch (err) {
      console.error('❌ Camera error:', err)
      setError(err instanceof Error ? err.message : 'Camera error occurred')
    }
  }

  const stopCamera = () => {
    // Stop animation frame loop
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    // Disconnect WebSocket (clears reconnection)
    disconnectWebSocket()

    // Stop camera stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setIsActive(false)
    setFps(0)
    frameCountRef.current = 0
    console.log('⏹️ Camera stopped')
  }

  // Manual retry function (resets attempts and reconnects)
  const handleManualRetry = () => {
    console.log('🔄 Manual retry triggered')
    setReconnectAttempt(0)
    setIsReconnecting(false)
    setError(null)
    shouldReconnectRef.current = true
    connectWebSocket()
  }

  // Capture and send frame via WebSocket with backpressure control
  const captureAndSendFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return
    if (wsRef.current.readyState !== WebSocket.OPEN) return
    if (isProcessing) return // BACKPRESSURE: Don't send if still processing

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
      if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
        // Convert blob to ArrayBuffer and send via WebSocket
        blob.arrayBuffer().then((buffer) => {
          wsRef.current?.send(buffer)
          setIsProcessing(true) // Set processing flag for backpressure
        })
      }
    }, 'image/jpeg', 0.85)
  }, [isProcessing, setIsProcessing])

  // requestAnimationFrame loop for smooth frame capture
  useEffect(() => {
    if (!isActive || connectionStatus !== 'connected') {
      return
    }

    let lastCaptureTime = 0
    const targetFps = 10 // Target 10fps to match backend CPU limit
    const frameInterval = 1000 / targetFps

    const frameLoop = (timestamp: number) => {
      if (!isActive) return

      // Throttle to target FPS
      if (timestamp - lastCaptureTime >= frameInterval) {
        captureAndSendFrame()
        lastCaptureTime = timestamp
      }

      animationFrameRef.current = requestAnimationFrame(frameLoop)
    }

    animationFrameRef.current = requestAnimationFrame(frameLoop)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [isActive, connectionStatus, captureAndSendFrame])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Clear reconnection timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }

      // Stop camera and disconnect
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

        {/* Status Overlay */}
        {isActive && (
          <div className="absolute top-4 left-4 flex gap-2">
            {/* Connection Status */}
            <div className={`px-2 py-1 text-xs font-medium rounded ${
              connectionStatus === 'connected' ? 'bg-green-500/90 text-white' :
              connectionStatus === 'connecting' && isReconnecting ? 'bg-yellow-500/90 text-white' :
              connectionStatus === 'connecting' ? 'bg-yellow-500/90 text-white' :
              'bg-red-500/90 text-white'
            }`}>
              {connectionStatus === 'connected' ? '🟢 Live' :
               connectionStatus === 'connecting' && isReconnecting ?
                 `🟡 Reconnecting... (${reconnectAttempt}/${MAX_RECONNECT_ATTEMPTS})` :
               connectionStatus === 'connecting' ? '🟡 Connecting...' :
               reconnectAttempt >= MAX_RECONNECT_ATTEMPTS ? '🔴 Connection Failed' :
               '🔴 Disconnected'}
            </div>

            {/* FPS Counter */}
            {connectionStatus === 'connected' && fps > 0 && (
              <div className="px-2 py-1 text-xs font-medium bg-blue-500/90 text-white rounded">
                {fps} FPS
              </div>
            )}
          </div>
        )}

        {/* Processing overlay */}
        {isProcessing && (
          <div className="absolute inset-0 bg-black/30 flex items-center justify-center">
            <div className="text-white text-xs">Processing...</div>
          </div>
        )}
      </div>

      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Controls */}
      <div className="mt-6 flex gap-3 items-center">
        {!isActive ? (
          <button
            onClick={startCamera}
            className="px-6 py-2 bg-slate-900 hover:bg-slate-800 text-white text-sm"
          >
            Start Streaming
          </button>
        ) : (
          <>
            <button
              onClick={stopCamera}
              className="px-6 py-2 bg-red-600 hover:bg-red-700 text-white text-sm"
            >
              Stop Streaming
            </button>

            {/* Manual Retry Button - Shows when max attempts reached */}
            {connectionStatus === 'disconnected' && reconnectAttempt >= MAX_RECONNECT_ATTEMPTS && (
              <button
                onClick={handleManualRetry}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm"
              >
                Retry Connection
              </button>
            )}

            <div className="text-xs text-slate-500">
              {connectionStatus === 'connected'
                ? `Streaming at ~${fps || 10}fps`
                : isReconnecting
                ? `Retrying in ${calculateBackoffDelay(reconnectAttempt - 1)}ms...`
                : connectionStatus === 'connecting'
                ? 'Connecting to server...'
                : reconnectAttempt >= MAX_RECONNECT_ATTEMPTS
                ? 'Connection failed - click Retry'
                : 'Disconnected'}
            </div>
          </>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-3 border border-red-300 bg-red-50 text-sm text-red-900">
          <div className="font-medium mb-1">Connection Error</div>
          <div>{error}</div>
          {reconnectAttempt >= MAX_RECONNECT_ATTEMPTS && (
            <div className="mt-2 text-xs text-red-700">
              Possible causes: Server offline, network issues, or firewall blocking WebSocket connections.
            </div>
          )}
        </div>
      )}
    </div>
  )
}