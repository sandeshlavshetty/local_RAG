/**
 * Centralized API Configuration
 * Use this file instead of hardcoding endpoints in components
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000"

export const API_ENDPOINTS = {
  // Documents
  DOCUMENTS_LIST: `${API_BASE_URL}/documents/`,
  DOCUMENT_VIEW: (fileName: string) => `${API_BASE_URL}/documents/${fileName}`,
  UPLOAD_FILE: `${API_BASE_URL}/upload/`,

  // Chat & Queries
  ASK_QUESTION: `${API_BASE_URL}/ask/`,
  ASK_THREAD: `${API_BASE_URL}/ask/thread`,

  // Health Check
  HEALTH: `${API_BASE_URL}/health`,
  ROOT: `${API_BASE_URL}/`,
}

export { API_BASE_URL }