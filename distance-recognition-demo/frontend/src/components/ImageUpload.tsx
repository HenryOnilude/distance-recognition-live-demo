'use client'

import { useState, useRef } from 'react'

// TypeScript interface defining the props this component accepts
interface ImageUploadProps {
  onImageAnalyze: (imageBlob: Blob) => void  // Callback function to send image to parent
  isProcessing: boolean                       // Whether analysis is in progress
}

export default function ImageUpload({ onImageAnalyze, isProcessing }: ImageUploadProps) {
  // STATE MANAGEMENT
  const [selectedImage, setSelectedImage] = useState<File | null>(null)  // The actual file
  const [preview, setPreview] = useState<string | null>(null)            // URL for preview display
  const [isDragging, setIsDragging] = useState(false)                    // Drag-over visual feedback
  const fileInputRef = useRef<HTMLInputElement>(null)                    // Reference to hidden file input

  // HANDLER: When user selects file via file picker
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file)
      setPreview(URL.createObjectURL(file))  // Create temporary URL for preview
    }
  }

  // HANDLER: When user drops a file onto the drop zone
  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file)
      setPreview(URL.createObjectURL(file))
    }
  }

  // HANDLER: Send image to parent component for analysis
  const handleAnalyze = () => {
    if (selectedImage) {
      onImageAnalyze(selectedImage)
    }
  }

  // HANDLER: Clear selected image and free memory
  const clearImage = () => {
    setSelectedImage(null)
    if (preview) URL.revokeObjectURL(preview)  // Free memory from blob URL
    setPreview(null)
  }

  return (
    <div className="space-y-4">
      {/* Hidden file input - always available */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* CONDITIONAL RENDER: Show upload zone OR preview */}
      {!preview ? (
        //  UPLOAD DROP ZONE 
        <div
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed p-16 text-center cursor-pointer ${
            isDragging
              ? 'border-slate-900 bg-slate-50'
              : 'border-slate-300 hover:border-slate-400'
          }`}
        >
          <div className="flex flex-col items-center space-y-3">
            <p className="text-sm text-slate-600">
              {isDragging ? 'Drop image here' : 'Drop image or click to browse'}
            </p>
            <p className="text-xs text-slate-400">JPG, PNG, GIF up to 10MB</p>
          </div>
        </div>
      ) : (
        //IMAGE PREVIEW 
        <div className="space-y-4">
          <div className="relative bg-black">
            <img
              src={preview}
              alt="Selected image"
              className="w-full h-[400px] object-contain"
            />

            {/* Clear button (X) */}
            <button
              onClick={clearImage}
              className="absolute top-3 right-3 w-8 h-8 bg-white hover:bg-slate-100 text-slate-900 flex items-center justify-center"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* Processing overlay */}
            {isProcessing && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <div className="text-white text-sm">Processing...</div>
              </div>
            )}
          </div>

          {/* File info */}
          <div className="text-sm text-slate-500">
            {selectedImage?.name} · {selectedImage ? `${(selectedImage.size / 1024 / 1024).toFixed(1)} MB` : ''}
          </div>

          {/* Action buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleAnalyze}
              disabled={isProcessing}
              className="px-6 py-2 bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 text-white text-sm disabled:cursor-not-allowed"
            >
              Analyze
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-6 py-2 border border-slate-300 hover:bg-slate-50 text-slate-900 text-sm"
            >
              Change
            </button>
          </div>
        </div>
      )}
    </div>
  )
}