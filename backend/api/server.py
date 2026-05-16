from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

print("\n=== FASTAPI SERVER INITIALIZATION ===")
print("[DEBUG SERVER] Creating FastAPI app...")

app = FastAPI(title="Multimodal RAG API")
print("[DEBUG SERVER] FastAPI app created")

# CORS so frontend can call
print("[DEBUG SERVER] Adding CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("[DEBUG SERVER] CORS middleware added")

# Include routes
print("[DEBUG SERVER] Including routes...")
app.include_router(router)
print("[DEBUG SERVER] Routes included")

@app.on_event("startup")
async def startup_event():
    print("\n[DEBUG SERVER] FastAPI startup event triggered")

@app.on_event("shutdown")
async def shutdown_event():
    print("[DEBUG SERVER] FastAPI shutdown event triggered")

@app.get("/")
def root():
    return {"message": "✅ Multimodal RAG API running"}

@app.get("/health")
def health_check():
    print("[DEBUG SERVER] Health check endpoint called")
    return {"status": "healthy", "message": "API is running"}
    
print("[DEBUG SERVER] FastAPI app fully initialized")
print("=== END FASTAPI SERVER INITIALIZATION ===")
