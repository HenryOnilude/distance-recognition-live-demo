'use client'

interface Prediction {
  predicted_class: string
  confidence: number
  decision: string
  expected_accuracy: number
}

interface AnalysisResult {
  success: boolean
  processing_time_ms: number
  distance_m: number
  distance_category: string
  quality_score: number
  predictions: { [key: string]: Prediction }
  expected_overall_accuracy: number
  error?: string
}

interface ResultsPanelProps {
  result: AnalysisResult | null
  lastUpdate: Date | null
}

export default function ResultsPanel({ result, lastUpdate }: ResultsPanelProps) {
  if (!result) {
    return (
      <div className="text-sm text-slate-400">
        No data yet
      </div>
    )
  }

  if (result.error) {
    return (
      <div className="text-sm text-slate-900">
        {result.error}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Predictions */}
      {Object.entries(result.predictions).map(([task, prediction]) => (
        <div key={task} className="space-y-2">
          <div className="flex justify-between items-baseline">
            <span className="text-sm text-slate-500 capitalize">{task}</span>
            <span className="text-sm text-slate-900">{prediction.predicted_class}</span>
          </div>
          <div className="flex justify-between items-baseline">
            <span className="text-xs text-slate-400">Confidence</span>
            <span className="text-xs text-slate-600">{(prediction.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      ))}

      {/* Quality Score */}
      {result.success && (
        <div className="pt-4 border-t border-slate-200">
          <div className="flex justify-between items-baseline">
            <span className="text-sm text-slate-500">Quality</span>
            <span className="text-sm text-slate-900">{(result.quality_score * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Timestamp */}
      {lastUpdate && (
        <div className="text-xs text-slate-400 pt-4 border-t border-slate-200">
          {lastUpdate.toLocaleTimeString()}
        </div>
      )}
    </div>
  )
}