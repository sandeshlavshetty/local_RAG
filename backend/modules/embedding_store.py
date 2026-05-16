import os
import json
import uuid
import faiss
from sentence_transformers import SentenceTransformer
from config import VECTOR_DIR

print("[DEBUG EMBED_STORE] Initializing embedding_store...")
print(f"[DEBUG EMBED_STORE] VECTOR_DIR: {VECTOR_DIR}")

os.makedirs(VECTOR_DIR, exist_ok=True)

print("[DEBUG EMBED_STORE] Loading embedding model...")
EMBEDDING_MODEL = SentenceTransformer("local_models/all-MiniLM-L6-v2")
print("[DEBUG EMBED_STORE] Embedding model loaded")

METADATA_FILE = os.path.join(VECTOR_DIR, "metadata.json")
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
print(f"[DEBUG EMBED_STORE] METADATA_FILE: {METADATA_FILE}")
print(f"[DEBUG EMBED_STORE] INDEX_FILE: {INDEX_FILE}")

# Load existing FAISS index if exists
print(f"[DEBUG EMBED_STORE] Checking if INDEX_FILE exists: {os.path.exists(INDEX_FILE)}")
if os.path.exists(INDEX_FILE):
    print("[DEBUG EMBED_STORE] Loading existing FAISS index...")
    index = faiss.read_index(INDEX_FILE)
    print(f"[DEBUG EMBED_STORE] FAISS index loaded. Total vectors: {index.ntotal}")
    print("[DEBUG EMBED_STORE] Loading metadata...")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata_store = json.load(f)
    print(f"[DEBUG EMBED_STORE] Metadata loaded. Total entries: {len(metadata_store)}")
else:
    print("[DEBUG EMBED_STORE] Creating new FAISS index...")
    index = faiss.IndexFlatL2(384)  # dimension for MiniLM-L6-v2
    metadata_store = {}
    print("[DEBUG EMBED_STORE] New FAISS index and metadata store created")

def add_to_index(text, modality, source_file, page_num=None):
    print(f"\n[DEBUG EMBED_STORE] add_to_index called")
    print(f"[DEBUG EMBED_STORE] Modality: {modality}, Source: {source_file}, Page: {page_num}")
    print(f"[DEBUG EMBED_STORE] Text length: {len(text)}")
    
    try:
        chunk_id = str(uuid.uuid4())
        print(f"[DEBUG EMBED_STORE] Generated chunk_id: {chunk_id}")
        
        print("[DEBUG EMBED_STORE] Encoding text...")
        embedding = EMBEDDING_MODEL.encode([text])
        print(f"[DEBUG EMBED_STORE] Embedding shape: {embedding.shape}")
        
        print(f"[DEBUG EMBED_STORE] Adding to FAISS index. Current total: {index.ntotal}")
        index.add(embedding)
        print(f"[DEBUG EMBED_STORE] Added to FAISS. New total: {index.ntotal}")
        
        metadata_store[chunk_id] = {
            "id": chunk_id,
            "modality": modality,
            "source_file": source_file,
            "page_num": page_num,
            "text_excerpt": text
        }
        print(f"[DEBUG EMBED_STORE] Metadata stored. Total metadata entries: {len(metadata_store)}")
        
        # Update BM25 index with new document
        try:
            from modules.retriever import update_bm25_index
            update_bm25_index(chunk_id, text)
        except Exception as e:
            print(f"[WARNING EMBED_STORE] Failed to update BM25 index: {str(e)}")
    except Exception as e:
        print(f"[ERROR EMBED_STORE] add_to_index failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def save_index():
    print(f"\n[DEBUG EMBED_STORE] save_index called")
    print(f"[DEBUG EMBED_STORE] Saving FAISS index to: {INDEX_FILE}")
    print(f"[DEBUG EMBED_STORE] FAISS index total vectors: {index.ntotal}")
    print(f"[DEBUG EMBED_STORE] Metadata entries: {len(metadata_store)}")
    
    try:
        faiss.write_index(index, INDEX_FILE)
        print("[DEBUG EMBED_STORE] FAISS index saved successfully")
        
        print(f"[DEBUG EMBED_STORE] Saving metadata to: {METADATA_FILE}")
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata_store, f, indent=2, ensure_ascii=False)
        print("[DEBUG EMBED_STORE] Metadata saved successfully")
    except Exception as e:
        print(f"[ERROR EMBED_STORE] save_index failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
