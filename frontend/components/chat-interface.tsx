
"use client"

import React, { useEffect, useRef, useState } from "react"
import { MultimodalInput } from "@/components/multimodal-input"
import { DragDropOverlay } from "@/components/drag-drop-overlay"
import { useDragDrop } from "@/hooks/use-drag-drop"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import {
  MessageCircle,
  Send,
  Mic,
  ImageIcon,
  Paperclip,
  Copy,
  RefreshCw,
  Bot,
  User,
  X,
} from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface MessageAttachment {
  type: "image" | "audio" | "document"
  url: string
  name: string
  size?: string
}

interface Citation {
  label: string
  url?: string
}

interface Message {
  id: string
  type: "user" | "ai"
  content: string
  timestamp: Date
  attachments?: MessageAttachment[]
  isTyping?: boolean
  sources?: Citation[]
  queryUsed?: string  // The query that was used for retrieval
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      type: "ai",
      content:
        "Hello! I'm your RAG assistant. You can upload documents, images, or audio and ask me anything about them!",
      timestamp: new Date(),
      sources: [],
    },
  ])

  const [inputValue, setInputValue] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [showMultimodalInput, setShowMultimodalInput] = useState(false)

  // pending attachments that will be sent with next text message (or can be sent alone)
  const [pendingAttachments, setPendingAttachments] = useState<
    MessageAttachment[]
  >([])

  const messagesEndRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLInputElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const imageInputRef = useRef<HTMLInputElement | null>(null)

  // recording refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordingChunksRef = useRef<Blob[]>([])

  const {
    isDragActive: isChatDragActive,
    isDragReject: isChatDragReject,
    dragProps: chatDragProps,
  } = useDragDrop({
    onDrop: (files) => {
      // instead of immediately sending on drop, add to pending attachments
      handleFiles(files)
    },
    accept: ["image/*", "audio/*", ".pdf", ".txt", ".doc", ".docx"],
    multiple: true,
  })

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // helper: convert FileList or File[] -> MessageAttachment[] and add to pending
  const handleFiles = (files: FileList | File[]) => {
    const fArray = Array.from(files)
    const attachments: MessageAttachment[] = fArray.map((file) => {
      const type = file.type.startsWith("image/")
        ? "image"
        : file.type.startsWith("audio/")
        ? "audio"
        : "document"
      return {
        type,
        url: URL.createObjectURL(file),
        name: file.name,
        size: formatFileSize(file.size),
      }
    })
    setPendingAttachments((prev) => [...prev, ...attachments])
  }

  async function handleCitationClick(label: string) {
  const fileName = label.split(' ')[0];
  console.log('Citation clicked:', fileName);
  // Open the backend URL directly, which streams the file
  window.open(`http://127.0.0.1:8000/documents/${fileName}`, "_blank", "noopener,noreferrer");
}

  const removePendingAttachment = (index: number) => {
    setPendingAttachments((prev) => prev.filter((_, i) => i !== index))
  }

  const sendMessage = async () => {
    // if there's no text and no attachments, do nothing
    if (!inputValue.trim() && pendingAttachments.length === 0) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue.trim() || pendingAttachments.map((p) => p.name).join(", "),
      timestamp: new Date(),
      attachments: pendingAttachments.length ? pendingAttachments : undefined,
    }

    // append user message
    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setPendingAttachments([])
    setIsTyping(true)

    // call backend with text and file (multipart/form-data)
    try {
      const formData = new FormData()
      formData.append("query", userMessage.content)
      
      // Add file if there's a pending attachment
      if (pendingAttachments.length > 0) {
        const attachment = pendingAttachments[0] // Get first attachment
        // Convert blob URL back to file - we need to fetch it
        const response = await fetch(attachment.url)
        const blob = await response.blob()
        formData.append("file", blob, attachment.name)
        console.log(`[DEBUG FRONTEND] Appending file to form: ${attachment.name}`)
      }
      
      const response = await fetch("http://127.0.0.1:8000/ask/", {
        method: "POST",
        // Don't set Content-Type header - browser will set it automatically with boundary
        body: formData,
      })
      const result = await response.json()

      let answerText = "No answer returned."
      let sources: Citation[] = []

      // robust parsing of possible result formats
      if (result.answer) {
        if (typeof result.answer === "string") {
          answerText = result.answer
          // If top-level citations exist, use them
          if (Array.isArray(result.citations)) {
            sources = result.citations.map((c: any) => ({
              label: c.source_file
                ? `${c.source_file}${c.page_num ? ` (p.${c.page_num})` : ""}`
                : c.text || c.title || JSON.stringify(c),
            }))
          }
        } else if (typeof result.answer === "object") {
          answerText =
            result.answer.answer ||
            result.answer.text ||
            JSON.stringify(result.answer)
          if (Array.isArray(result.answer.citations)) {
            sources = result.answer.citations.map((c: any) => {
              // Use source_file and page_num for label
              const label = c.source_file
                ? `${c.source_file}${c.page_num ? ` (p.${c.page_num})` : ""}`
                : c.text || c.title || JSON.stringify(c)
              return { label }
            })
          }
        }
      } else if (result.citations && Array.isArray(result.citations)) {
        // fallback if top-level citations exist
        sources = result.citations.map((c: any) => ({
          label: c.source_file
            ? `${c.source_file}${c.page_num ? ` (p.${c.page_num})` : ""}`
            : c.text || c.title || JSON.stringify(c),
        }))
      } else if (result.error) {
        answerText = result.error
      }

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: answerText,
        timestamp: new Date(),
        sources,
        queryUsed: result.query_used || "",  // Include the query used for retrieval
      }
      setMessages((prev) => [...prev, aiMessage])
    } catch (err) {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: "Sorry, there was an error getting the answer.",
        timestamp: new Date(),
        sources: [],
      }
      setMessages((prev) => [...prev, aiMessage])
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const formatTime = (date: Date) => {
    const hours = String(date.getHours()).padStart(2, "0")
    const minutes = String(date.getMinutes()).padStart(2, "0")
    return `${hours}:${minutes}`
  }

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content)
  }

  const copyCitation = (c: Citation) => {
    if (c.url) navigator.clipboard.writeText(c.url)
    else navigator.clipboard.writeText(c.label)
  }

  const regenerateResponse = async (messageId: string) => {
    const messageIndex = messages.findIndex((m) => m.id === messageId)
    if (messageIndex > 0) {
      const previousMessage = messages[messageIndex - 1]
      if (previousMessage.type === "user") {
        setIsTyping(true)
        try {
          const formData = new URLSearchParams()
          formData.append("query", previousMessage.content)
          const response = await fetch("http://127.0.0.1:8000/ask/", {
            method: "POST",
            headers: {
              "Content-Type": "application/x-www-form-urlencoded",
            },
            body: formData,
          })
          const result = await response.json()
          let answerText = result.answer || result.error || "No answer returned."
          // parse sources like in sendMessage (optional)
          let sources: Citation[] = []
          if (result.answer && typeof result.answer === "object" && Array.isArray(result.answer.citations)) {
            sources = result.answer.citations.map((c: any) => {
              if (typeof c === "string") {
                return { label: c, url: c.startsWith("http") ? c : undefined }
              }
              const url = c.url || c.source || c.source_file || c.link
              const label = c.title || c.source || c.source_file || url || JSON.stringify(c)
              return { label, url }
            })
          }

          const newResponse: Message = {
            id: Date.now().toString(),
            type: "ai",
            content: typeof answerText === "string" ? answerText : JSON.stringify(answerText),
            timestamp: new Date(),
            sources,
            queryUsed: result.query_used || "",  // Include the query used for retrieval
          }
          setMessages((prev) =>
            prev.map((m) => (m.id === messageId ? newResponse : m))
          )
        } catch {
          const newResponse: Message = {
            id: Date.now().toString(),
            type: "ai",
            content: "Sorry, there was an error getting the answer.",
            timestamp: new Date(),
            sources: [],
          }
          setMessages((prev) =>
            prev.map((m) => (m.id === messageId ? newResponse : m))
          )
        } finally {
          setIsTyping(false)
        }
      }
    }
  }

  // recording helpers
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      recordingChunksRef.current = []
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordingChunksRef.current.push(e.data)
      }
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordingChunksRef.current, { type: "audio/webm" })
        const url = URL.createObjectURL(blob)
        const fileName = `recording_${Date.now()}.webm`
        const attachment: MessageAttachment = {
          type: "audio",
          url,
          name: fileName,
          size: formatFileSize(blob.size),
        }
        // add to pending attachments so user can review and send
        setPendingAttachments((prev) => [...prev, attachment])
        // stop tracks
        stream.getTracks().forEach((t) => t.stop())
      }
      mediaRecorder.start()
      setIsRecording(true)
    } catch (err) {
      console.error("Could not start recording:", err)
      // you might show a toast here in the real app
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop()
    }
    setIsRecording(false)
    mediaRecorderRef.current = null
  }

  // file input triggers
  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files)
      e.currentTarget.value = ""
    }
  }

  const onImageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files)
      e.currentTarget.value = ""
    }
  }

  return (
    <Card
      className="w-full max-w-2xl mx-auto h-[700px] flex flex-col border"
      {...chatDragProps}
    >
      <DragDropOverlay
        isActive={isChatDragActive}
        isReject={isChatDragReject}
        message="Drop files to add to conversation"
        acceptedTypes={["image/*", "audio/*", ".pdf"]}
      />

      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <MessageCircle className="h-5 w-5" />
            Chat Interface
            {messages.length > 1 && (
              <Badge variant="secondary" className="ml-2">
                {messages.length - 1} messages
              </Badge>
            )}
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setMessages((prev) => prev.slice(0, 1))
              setPendingAttachments([])
            }}
          >
            Clear Chat
          </Button>
        </div>
      </CardHeader>

      {/* main content */}
      <CardContent className="flex-1 flex flex-col p-0 min-h-0">
        {/* messages list */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 min-h-0">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-3 ${
                  message.type === "user" ? "flex-row-reverse" : "flex-row"
                }`}
              >
                <Avatar className="w-8 h-8 flex-shrink-0">
                  <AvatarFallback
                    className={
                      message.type === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                    }
                  >
                    {message.type === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                  </AvatarFallback>
                </Avatar>

                <div className={`flex-1 max-w-full ${message.type === "user" ? "items-end" : "items-start"} flex flex-col`}>
                  <div
                    className={`rounded-2xl px-4 py-3 text-sm relative group break-words max-w-full overflow-hidden
                      ${message.type === "user" ? "bg-primary text-primary-foreground" : "bg-muted border border-border"}`}
                  >
                    <div
                      className={`absolute top-2 opacity-0 group-hover:opacity-100 transition-opacity ${message.type === "user" ? "-left-8" : "-right-8"}`}
                    >
                      <div className="flex flex-col gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copyMessage(message.content)}
                          className="h-6 w-6 p-0 bg-background/80 backdrop-blur-sm"
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                        {message.type === "ai" && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => regenerateResponse(message.id)}
                            className="h-6 w-6 p-0 bg-background/80 backdrop-blur-sm"
                          >
                            <RefreshCw className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                    </div>

                    {/* attachments preview inside each message (images, audio, docs) */}
                    {message.attachments && message.attachments.length > 0 && (
                      <div className="mb-2 flex flex-col gap-2">
                        {message.attachments.map((att, i) => (
                          <div key={i} className="flex items-center gap-2">
                            {att.type === "image" ? (
                              <img src={att.url} alt={att.name} className="max-h-40 rounded-md border" />
                            ) : att.type === "audio" ? (
                              <audio controls src={att.url} className="max-w-xs" />
                            ) : (
                              <a href={att.url} target="_blank" rel="noreferrer" className="underline">
                                {att.name} {att.size ? `(${att.size})` : ""}
                              </a>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    <p className="text-pretty leading-relaxed whitespace-pre-wrap">{message.content}</p>

                    {/* citations: only show for AI messages except the welcome message */}
                    {message.type === "ai" && message.id !== "welcome" && (
                      <>
                        {message.sources && message.sources.length > 0 ? (
                          <div className="mt-3 flex flex-wrap gap-2">
                            {message.sources.map((c, idx) => (
                              <Badge key={idx} variant="outline" className="text-xs flex items-center gap-1">
                                <span
                                  className="underline cursor-pointer text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded px-1 transition-colors"
                                  title={`Open ${c.label}`}
                                  onClick={() => handleCitationClick(c.label)}
                                  role="button"
                                  tabIndex={0}
                                  onKeyPress={e => { if (e.key === 'Enter') handleCitationClick(c.label); }}
                                >
                                  {c.label}
                                </span>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => copyCitation(c)}
                                  className="h-4 w-4 p-0"
                                  aria-label="Copy citation"
                                >
                                  <Copy className="h-3 w-3" />
                                </Button>
                              </Badge>
                            ))}
                          </div>
                        ) : null}
                      </>
                    )}
                  </div>

                  <p className="text-xs text-muted-foreground mt-1 px-1">{formatTime(message.timestamp)}</p>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* typing indicator */}
          {isTyping && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex gap-3">
              <Avatar className="w-8 h-8 flex-shrink-0">
                <AvatarFallback className="bg-muted">
                  <Bot className="h-4 w-4" />
                </AvatarFallback>
              </Avatar>
              <div className="bg-muted rounded-2xl px-4 py-3 flex items-center gap-2">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce [animation-delay:0.1s]" />
                  <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce [animation-delay:0.2s]" />
                </div>
                <span className="text-sm text-muted-foreground">AI is thinking...</span>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* typing / input area */}
        <div className="border-t bg-background px-3 py-3">
          {/* pending attachments preview (will be sent with next message) */}
          {pendingAttachments.length > 0 && (
            <div className="mb-2 flex gap-2 overflow-x-auto">
              {pendingAttachments.map((att, i) => (
                <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-xl border bg-muted/30">
                  <div className="flex flex-col">
                    <div className="text-sm font-medium">{att.name}</div>
                    {att.size && <div className="text-xs text-muted-foreground">{att.size}</div>}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removePendingAttachment(i)}
                    className="h-6 w-6 p-0"
                    aria-label="Remove attachment"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault()
              sendMessage()
            }}
            autoComplete="off"
          >
            <div className="flex-1 relative">
              <input
                ref={textareaRef as any}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Send a message..."
                className="w-full rounded-xl border px-4 py-3 text-base bg-background focus:outline-none focus:ring-2 focus:ring-primary/30 transition shadow-sm"
                disabled={isTyping}
              />

              {/* inline input action icons */}
              <div className="absolute right-2 top-2 flex items-center gap-1">
                {/* image upload */}
                <input
                  ref={imageInputRef}
                  type="file"
                  accept="image/*"
                  onChange={onImageInputChange}
                  className="hidden"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => imageInputRef.current?.click()}
                  className="h-8 w-8 p-0"
                  type="button"
                  aria-label="Attach image"
                >
                  <ImageIcon className="h-4 w-4" />
                </Button>

                {/* file upload */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt,.doc,.docx,audio/*"
                  onChange={onFileInputChange}
                  className="hidden"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  className="h-8 w-8 p-0"
                  type="button"
                  aria-label="Attach file"
                >
                  <Paperclip className="h-4 w-4" />
                </Button>

                {/* mic (record) */}
                <Button
                  variant={isRecording ? "destructive" : "ghost"}
                  size="sm"
                  onClick={() => {
                    if (isRecording) stopRecording()
                    else startRecording()
                  }}
                  className="h-8 w-8 p-0"
                  type="button"
                  aria-pressed={isRecording}
                  aria-label={isRecording ? "Stop recording" : "Start recording"}
                >
                  <Mic className="h-4 w-4" />
                </Button>

           
              </div>
            </div>

            <Button
              type="submit"
              disabled={!inputValue.trim() && pendingAttachments.length === 0}
              className="h-11 w-11 p-0 rounded-xl bg-primary text-primary-foreground hover:bg-primary/80 transition"
            >
              <Send className="h-4 w-4" />
            </Button>
          </form>

        </div>
      </CardContent>
    </Card>
  )
}
