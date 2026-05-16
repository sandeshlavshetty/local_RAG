"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { API_ENDPOINTS } from "@/lib/api-config"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { FileText, ImageIcon, Mic, Upload, X, CheckCircle, AlertCircle, Clock, FolderOpen, Trash2 } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface UploadedFile {
  id?: string // local only
  name: string
  type?: "pdf" | "image" | "audio"
  size?: string
  sizeBytes?: number
  status?: "processing" | "upserted" | "failed"
  progress?: number
  uploadedAt?: Date | string
  modality?: string
  file_size?: number
  upload_date?: string
  chunk_count?: number
  page_count?: number | null
  pages?: number[]
  filename?: string
}

export function DocumentPanel() {
  const router = useRouter()
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [documents, setDocuments] = useState<UploadedFile[]>([])
  const [loadingDocs, setLoadingDocs] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [activeTab, setActiveTab] = useState("all")
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [refreshing, setRefreshing] = useState(false)
  // Fetch all documents from backend
  // Fetch documents, optionally mark as manual refresh
  const fetchDocuments = async (isManualRefresh = false) => {
    if (uploading && isManualRefresh) {
      setError("Cannot refresh while uploading. Please wait.")
      return
    }
    if (isManualRefresh) setRefreshing(true)
    setLoadingDocs(true)
    setError(null)
    try {
      const response = await fetch(API_ENDPOINTS.DOCUMENTS_LIST)
      const data = await response.json()
      setDocuments(data.documents || [])
    } catch (err) {
      setError("Failed to fetch documents.")
    } finally {
      setLoadingDocs(false)
      if (isManualRefresh) setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const droppedFiles = Array.from(e.dataTransfer.files)
    handleFiles(droppedFiles)
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files)
      handleFiles(selectedFiles)
    }
  }

  // Upload files to backend
  const handleFiles = async (fileList: File[]) => {
    setUploading(true)
    setError(null)
    let anyUploaded = false
    for (const file of fileList) {
      const formData = new FormData()
      formData.append("file", file)
      try {
        const response = await fetch(API_ENDPOINTS.UPLOAD_FILE, {
          method: "POST",
          body: formData,
        })
        const result = await response.json()
        if (result.error) {
          setError(result.error)
        } else {
          anyUploaded = true
        }
      } catch (err) {
        setError("Upload failed.")
      }
    }
    setUploading(false)
    if (anyUploaded) {
      fetchDocuments()
    }
  }

  const getFileType = (mimeType: string): "pdf" | "image" | "audio" => {
    if (mimeType.includes("pdf")) return "pdf"
    if (mimeType.includes("image")) return "image"
    if (mimeType.includes("audio")) return "audio"
    return "pdf" // default
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const getFileIcon = (type: string) => {
    switch (type) {
      case "pdf":
        return <FileText className="h-5 w-5 text-red-500" />
      case "image":
        return <ImageIcon className="h-5 w-5 text-blue-500" />
      case "audio":
        return <Mic className="h-5 w-5 text-green-500" />
      default:
        return <FileText className="h-5 w-5" />
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "processing":
        return <Clock className="h-4 w-4 animate-spin text-yellow-500" />
      case "upserted":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "failed":
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return null
    }
  }

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id))
  }

  const clearAllFiles = () => {
    setFiles([])
  }

  const retryFailedFiles = () => {
    setFiles((prev) => prev.map((f) => (f.status === "failed" ? { ...f, status: "processing", progress: 0 } : f)))
  }

  // Filter documents by tab, slice to 10 for preview
  const getFilteredDocuments = () => {
    let filtered: UploadedFile[]
    switch (activeTab) {
      case "pdf":
        filtered = documents.filter((f) => f.modality === "text" || f.type === "pdf")
        break
      case "image":
        filtered = documents.filter((f) => f.modality === "image" || f.type === "image")
        break
      case "audio":
        filtered = documents.filter((f) => f.modality === "audio" || f.type === "audio")
        break
      default:
        filtered = documents
    }
    return filtered.slice(0, 3)
  }

  const stats = {
    total: documents.length,
    processed: documents.length, // all are processed from backend
    processing: uploading ? 1 : 0,
    failed: error ? 1 : 0,
    totalSize: documents.reduce((acc, f) => acc + (f.file_size || f.sizeBytes || 0), 0),
  }

  return (
    <Card className="h-full flex flex-col">
  <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Document Library
            {documents.length > 0 && (
              <Badge variant="secondary" className="ml-2">
                {documents.length}
              </Badge>
            )}
            <Button
              variant="outline"
              size="sm"
              className="ml-4"
              onClick={() => fetchDocuments(true)}
              disabled={uploading || refreshing}
            >
              {refreshing ? "Refreshing..." : "Refresh"}
            </Button>
            {loadingDocs && !refreshing && (
              <span className="ml-2 text-xs text-muted-foreground">Fetching documents...</span>
            )}
          </CardTitle>
          {documents.length > 0 && (
            <div className="flex gap-2">
              {stats.failed > 0 && (
                <Button variant="outline" size="sm" onClick={retryFailedFiles}>
                  Retry Failed
                </Button>
              )}
              <Button variant="outline" size="sm" onClick={clearAllFiles}>
                <Trash2 className="h-4 w-4 mr-1" />
                Clear All
              </Button>
            </div>
          )}
        </div>
      </CardHeader>

  <CardContent className="flex-1 flex flex-col gap-4">
        {/* Drop Zone */}
  <motion.div
          className={`
            border-2 border-dashed rounded-xl p-6 text-center transition-all duration-200
            ${
              dragActive
                ? "border-primary bg-primary/5 scale-105"
                : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/20"
            }
          `}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
        >
          <motion.div
            animate={{
              y: dragActive ? -5 : 0,
              scale: dragActive ? 1.1 : 1,
            }}
            transition={{ duration: 0.2 }}
          >
            <Upload className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
          </motion.div>
          <p className="text-base font-medium mb-1">Drop files here or click to browse</p>
          <p className="text-sm text-muted-foreground mb-4">Supports PDF, Images (PNG, JPG, WebP), and Audio files</p>
          <Button variant="outline" onClick={() => fileInputRef.current?.click()} className="relative" disabled={uploading}>
            <FolderOpen className="h-4 w-4 mr-2" />
            {uploading ? "Uploading..." : "Browse Files"}
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.png,.jpg,.jpeg,.webp,.mp3,.wav,.m4a"
            onChange={handleFileInput}
            className="hidden"
          />
        </motion.div>

        {/* File Tabs */}
        {/* Show loading indicator while fetching, but keep old docs visible */}
        {loadingDocs && !refreshing && (
          <div className="text-center py-2 text-muted-foreground text-sm">Fetching documents...</div>
        )}
        {documents.length > 0 && (
          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="all">All ({documents.length})</TabsTrigger>
              <TabsTrigger value="pdf">PDF ({documents.filter((f) => f.modality === "text" || f.type === "pdf").length})</TabsTrigger>
              <TabsTrigger value="image">Images ({documents.filter((f) => f.modality === "image" || f.type === "image").length})</TabsTrigger>
              <TabsTrigger value="audio">Audio ({documents.filter((f) => f.modality === "audio" || f.type === "audio").length})</TabsTrigger>
            </TabsList>

            <TabsContent value={activeTab} className="flex-1 mt-4">
              <div className="space-y-2 h-full overflow-y-auto pr-2">
                <AnimatePresence>
                  {getFilteredDocuments().map((file, idx) => (
                    <motion.div
                      key={file.name + idx}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -100 }}
                      className="bg-card border rounded-lg p-3 hover:shadow-sm transition-shadow"
                    >
                      <div className="flex items-start gap-3">
                        <div className="mt-0.5">{getFileIcon((file.modality || file.type || "pdf") as string)}</div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-1">
                            <p className="text-sm font-medium truncate pr-2">{file.name || file.filename}</p>
                            <div className="flex items-center gap-2 flex-shrink-0">
                              {/* Status icons not shown for backend docs */}
                            </div>
                          </div>
                          <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                            <span>{formatFileSize(file.file_size || file.sizeBytes || 0)}</span>
                            <span>{file.upload_date ? new Date(file.upload_date).toLocaleString() : ""}</span>
                          </div>
                          {file.chunk_count !== undefined && (
                            <div className="text-xs text-muted-foreground">Chunks: {file.chunk_count}</div>
                          )}
                          {file.page_count !== undefined && file.page_count !== null && (
                            <div className="text-xs text-muted-foreground">Pages: {file.page_count}</div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                {documents.length > 10 && (
                  <div className="flex justify-center mt-4">
                    <button
                      className="px-4 py-2 rounded bg-primary text-primary-foreground hover:bg-primary/80 transition"
                      onClick={() => router.push("/all-documents")}
                    >
                      View All
                    </button>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        )}

        {/* Stats */}
        {documents.length > 0 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="border-t pt-4 mt-auto">
            <div className="grid grid-cols-2 gap-4 mb-3">
              <div className="text-center">
                <p className="text-lg font-bold text-green-500">{stats.processed}</p>
                <p className="text-xs text-muted-foreground">Processed</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-bold text-yellow-500">{stats.processing}</p>
                <p className="text-xs text-muted-foreground">Processing</p>
              </div>
            </div>
            <div className="text-center">
              <p className="text-sm text-muted-foreground">
                Total: {formatFileSize(stats.totalSize)} • {stats.total} files
              </p>
            </div>
            {error && <p className="text-xs text-red-500 mt-2">{error}</p>}
          </motion.div>
        )}

        {/* Empty State */}
  {documents.length === 0 && !loadingDocs && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex-1 flex items-center justify-center text-center py-8"
          >
            <div>
              <FolderOpen className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
              <p className="text-muted-foreground">No documents uploaded yet</p>
              <p className="text-sm text-muted-foreground/70 mt-1">Start by uploading your first document</p>
            </div>
          </motion.div>
        )}
  </CardContent>
    </Card>
  )
}