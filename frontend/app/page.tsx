"use client"

import { useState, useEffect } from "react"
import { DocumentPanel } from "@/components/document-panel"
import { ChatInterface } from "@/components/chat-interface"
import { ThemeToggle } from "@/components/theme-toggle"
import { DragDropOverlay } from "@/components/drag-drop-overlay"
import { FloatingActionButton } from "@/components/floating-action-button"

import { useDragDrop } from "@/hooks/use-drag-drop"
import { motion, AnimatePresence } from "framer-motion"

export default function Home() {
  const [isDarkMode, setIsDarkMode] = useState(false)

  // On mount, set theme from localStorage
  useEffect(() => {
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme === 'dark') {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark');
    } else {
      setIsDarkMode(false);
      document.documentElement.classList.remove('dark');
    }
  }, []);
  const [showWelcome, setShowWelcome] = useState(true)

  const toggleTheme = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    if (newMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }

  // Hide welcome message after 5 seconds
  useEffect(() => {
    const timer = setTimeout(() => setShowWelcome(false), 5000)
    return () => clearTimeout(timer)
  }, [])

  const { isDragActive, isDragReject, dragProps } = useDragDrop({
    onDrop: (files) => {
      console.log("Files dropped globally:", files)
      // Here you would handle the global file drop
      // For now, we'll just log it
    },
    accept: [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".m4a"],
    multiple: true,
  })

  return (
    <div className={`min-h-screen bg-background transition-colors duration-300`} {...dragProps}>
      {/* Global Drag Drop Overlay */}
      <DragDropOverlay
        isActive={isDragActive}
        isReject={isDragReject}
        message="Drop files anywhere to get started"
        acceptedTypes={[".pdf", "image/*", "audio/*"]}
      />

      {/* Welcome Toast */}
      <AnimatePresence>
        {showWelcome && (
          <motion.div
            initial={{ opacity: 0, y: -100 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -100 }}
            className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50"
          >
            <div className="bg-card border rounded-lg shadow-lg p-4 max-w-md">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-primary to-chart-1 rounded-lg flex items-center justify-center">
                  <span className="text-primary-foreground font-bold">R</span>
                </div>
                <div>
                  <h3 className="font-semibold text-sm">Welcome to RAG Assistant!</h3>
                  <p className="text-xs text-muted-foreground">Upload documents and start chatting</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-40"
      >
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <motion.div className="flex items-center space-x-3" whileHover={{ scale: 1.02 }}>
            <div>
              <h1 className="text-xl font-semibold text-foreground">RAG Assistant</h1>
              <p className="text-sm text-muted-foreground">Multimodal Knowledge Interface</p>
            </div>
          </motion.div>

          <div className="flex items-center gap-3">
            <ThemeToggle isDark={isDarkMode} onToggle={toggleTheme} />
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-6 h-[calc(100vh-88px)]">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
          {/* Document Upload Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1, type: "spring", stiffness: 100 }}
            className="h-full"
            whileHover={{ scale: 1.01 }}
          >
            <DocumentPanel />
          </motion.div>

          {/* Chat Interface */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 100 }}
            className="h-full"
            whileHover={{ scale: 1.01 }}
          >
            <ChatInterface />
          </motion.div>
        </div>
      </main>

      {/* Floating Action Button */}
      <FloatingActionButton
        onNewChat={() => console.log("New chat")}
        onUploadDocument={() => console.log("Upload document")}
        onVoiceInput={() => console.log("Voice input")}
      />

      {/* Background Pattern */}
      <div className="fixed inset-0 -z-10 opacity-5">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-transparent to-chart-1/20" />
        <motion.div
          className="absolute inset-0"
          animate={{
            backgroundPosition: ["0% 0%", "100% 100%"],
          }}
          transition={{
            duration: 20,
            repeat: Number.POSITIVE_INFINITY,
            repeatType: "reverse",
          }}
          style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.15) 1px, transparent 0)`,
            backgroundSize: "20px 20px",
          }}
        />
      </div>
    </div>
  )
}
