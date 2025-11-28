'use client'

interface DistanceMeterProps {
  distance: number
  category: string
  accuracy: number
}

export default function DistanceMeter({ distance, category, accuracy }: DistanceMeterProps) {
  const getCategoryLabel = (cat: string) => {
    switch (cat) {
      case 'portrait':
        return 'Portrait'
      case 'close':
        return 'Close'
      case 'medium':
        return 'Medium'
      case 'far':
        return 'Far'
      default:
        return 'Unknown'
    }
  }

  const distancePercent = Math.min(Math.max((distance - 0.5) / 9.5 * 100, 0), 100)

  return (
    <div className="space-y-8">
      {/* Main Distance Display */}
      <div>
        <div className="text-5xl font-light text-slate-900 mb-2">
          {distance > 0 ? distance.toFixed(1) : '–'}
        </div>
        <div className="text-sm text-slate-500">
          {getCategoryLabel(category)} range
        </div>
      </div>

      {/* Distance Range Bar */}
      <div className="space-y-3">
        <div className="relative h-1 bg-slate-200">
          {distance > 0 && (
            <div
              className="absolute top-0 h-full transition-all duration-500"
              style={{ left: `${distancePercent}%` }}
            >
              <div className="w-2 h-2 -ml-1 -mt-0.5 bg-slate-900"></div>
            </div>
          )}
        </div>
        <div className="flex justify-between text-xs text-slate-400">
          <span>0.5m</span>
          <span>10m</span>
        </div>
      </div>

      {/* Accuracy */}
      <div className="pt-4 border-t border-slate-200">
        <div className="flex justify-between items-center">
          <span className="text-sm text-slate-500">Accuracy</span>
          <span className="text-sm text-slate-900">
            {accuracy > 0 ? `${(accuracy * 100).toFixed(0)}%` : '–'}
          </span>
        </div>
      </div>
    </div>
  )
}