'use client';
import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { motion, AnimatePresence } from "framer-motion"
import { API_ENDPOINTS } from "@/lib/api-config"

export default function AllDocumentsPage() {
  const [documents, setDocuments] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    setLoading(true)
    fetch(API_ENDPOINTS.DOCUMENTS_LIST)
      .then((res) => res.json())
      .then((data) => setDocuments(data.documents || []))
      .finally(() => setLoading(false))
  }, [])

  return (
    <Card className="max-w-3xl mx-auto mt-8">
      <CardHeader>
        <CardTitle>All Documents</CardTitle>
        <Badge variant="secondary">{documents.length}</Badge>
      </CardHeader>
      <CardContent>
        {loading && <div className="text-center py-2 text-muted-foreground text-sm">Loading...</div>}
        <div className="space-y-2">
          <AnimatePresence>
            {documents.map((file, idx) => (
              <motion.div
                key={file.name + idx}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -100 }}
                className="bg-card border rounded-lg p-3 hover:shadow-sm transition-shadow"
              >
                <div className="flex items-start gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-sm font-medium truncate pr-2">{file.name || file.filename}</p>
                    </div>
                    <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                      <span>{file.file_size ? `${(file.file_size / 1024 / 1024).toFixed(2)} MB` : ""}</span>
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
        </div>
      </CardContent>
    </Card>
  )
}