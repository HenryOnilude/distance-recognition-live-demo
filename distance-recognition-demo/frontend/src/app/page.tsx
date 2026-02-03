'use client'

import { useState, useCallback } from 'react'
import WebcamCapture from '@/components/WebcamCapture'
import ImageUpload from '@/components/ImageUpload'
import DistanceMeter from '@/components/DistanceMeter'
import ResultsPanel from '@/components/ResultsPanel'

interface AnalysisResult {
  success?: boolean
  processing_time_ms?: number
  distance_m?: number
  distance_category?: string
  quality_score?: number
  predictions?: {
    [key: string]: {
      predicted_class: string
      confidence: number
      decision: string
      expected_accuracy: number
    }
  }
  expected_overall_accuracy?: number
  error?: string
  frame_count?: number
}

export default function Home() {
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null)
  const [activeTab, setActiveTab] = useState<'camera' | 'upload'>('camera')

  // Handler for WebSocket analysis results (Camera tab)
  const handleAnalysisResult = useCallback((result: AnalysisResult) => {
    setResult(result)
    setLastUpdateTime(new Date())
  }, [])

  // Handler for HTTP upload (Upload tab)
  const handleImageUpload = useCallback(async (imageBlob: Blob) => {
    setIsProcessing(true)

    try {
      const formData = new FormData()
      formData.append('file', imageBlob, 'webcam-frame.jpg')

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/analyze-frame`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      setResult(data)
      setLastUpdateTime(new Date())
    } catch (error) {
      console.error('Analysis failed:', error)
      setResult({
        success: false,
        error: 'Failed to connect to analysis server',
        processing_time_ms: 0,
        distance_m: 0,
        distance_category: 'unknown',
        quality_score: 0,
        predictions: {},
        expected_overall_accuracy: 0
      })
    } finally {
      setIsProcessing(false)
    }
  }, [])

    return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <h1 className="text-lg font-medium text-slate-900">Distance Recognition</h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Stats Bar */}
        <div className="grid grid-cols-3 gap-8 mb-12 pb-12 border-b border-slate-200">
          <div>
            <p className="text-3xl font-light text-slate-900">
              {result?.expected_overall_accuracy ? `${(result.expected_overall_accuracy * 100).toFixed(0)}%` : '–'}
            </p>
            <p className="text-sm text-slate-500 mt-1">Accuracy</p>
          </div>
          <div>
            <p className="text-3xl font-light text-slate-900">{result?.processing_time_ms || '–'}ms</p>
            <p className="text-sm text-slate-500 mt-1">Processing</p>
          </div>
          <div>
            <p className="text-3xl font-light text-slate-900">{result?.distance_m ? `${result.distance_m.toFixed(1)}m` : '–'}</p>
            <p className="text-sm text-slate-500 mt-1">Distance</p>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Input Section */}
          <div className="lg:col-span-2">
            <div className="border border-slate-200">
              {/* Tabs */}
              <div className="flex border-b border-slate-200">
                <button
                  onClick={() => setActiveTab('camera')}
                  className={`flex-1 px-6 py-3 text-sm ${
                    activeTab === 'camera'
                      ? 'text-slate-900 border-b-2 border-slate-900'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  Camera
                </button>
                <button
                  onClick={() => setActiveTab('upload')}
                  className={`flex-1 px-6 py-3 text-sm ${
                    activeTab === 'upload'
                      ? 'text-slate-900 border-b-2 border-slate-900'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  Upload
                </button>
              </div>

              {/* Content */}
              <div className="p-8">
                {activeTab === 'camera' ? (
                  <WebcamCapture
                    onAnalysisResult={handleAnalysisResult}
                    isProcessing={isProcessing}
                    setIsProcessing={setIsProcessing}
                  />
                ) : (
                  <ImageUpload onImageAnalyze={handleImageUpload} isProcessing={isProcessing} />
                )}
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-12">
            {/* Distance Meter */}
            <div>
              <h3 className="text-sm font-medium text-slate-900 mb-6">Distance</h3>
              <DistanceMeter
                distance={result?.distance_m || 0}
                category={result?.distance_category || 'unknown'}
                accuracy={result?.expected_overall_accuracy || 0}
              />
            </div>

            {/* Recognition Results */}
            <div>
              <h3 className="text-sm font-medium text-slate-900 mb-6">Recognition</h3>
              <ResultsPanel result={result} lastUpdate={lastUpdateTime} />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}