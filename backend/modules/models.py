"""
Centralized model definitions and helpers used across the project.

Exposes:
- get_embedding_model(), get_embedding_dimension()
- load_faiss_index(create_if_missing=True), save_faiss_index(index)
- load_metadata_store(), save_metadata_store(store)
- get_whisper_model(model_size="base")
- OLLAMA_VL_MODEL, INDEX_FILE, METADATA_FILE
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict
from dotenv import load_dotenv
from config import VECTOR_DIR

load_dotenv()

# Configurable via env var; default VL model name for Ollama
OLLAMA_VL_MODEL = os.getenv("OLLAMA_VL_MODEL", "qwen2.5vl:7b")

# Ensure vectorstore folder exists
os.makedirs(VECTOR_DIR, exist_ok=True)
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
METADATA_FILE = os.path.join(VECTOR_DIR, "metadata.json")
BM25_CORPUS_FILE = os.path.join(VECTOR_DIR, "bm25_corpus.json")
BM25_IDS_FILE = os.path.join(VECTOR_DIR, "bm25_ids.json")

# Singletons
_embedding_model: Any = None
_whisper_model: Any = None


def get_embedding_model() -> Any:
    """Return a singleton SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print("[DEBUG MODELS] Loading embedding model...")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            print(f"[ERROR MODELS] Failed to import SentenceTransformer: {str(e)}")
            raise RuntimeError(
                "sentence-transformers is not installed. Please install it to use embeddings."
            ) from e
        try:
            print("[DEBUG MODELS] Loading SentenceTransformer from local_models/all-MiniLM-L6-v2...")
            _embedding_model = SentenceTransformer("local_models/all-MiniLM-L6-v2")
            print("[DEBUG MODELS] Embedding model loaded successfully")
        except Exception as e:
            print(f"[ERROR MODELS] Failed to load embedding model: {str(e)}")
            raise
    return _embedding_model


def get_embedding_dimension() -> int:
    print("[DEBUG MODELS] Getting embedding dimension...")
    model = get_embedding_model()
    dim = int(getattr(model, "get_sentence_embedding_dimension", lambda: 384)())
    print(f"[DEBUG MODELS] Embedding dimension: {dim}")
    return dim


def load_faiss_index(create_if_missing: bool = True):
    """Load FAISS index; create empty IndexFlatL2 if missing and allowed."""
    print("[DEBUG MODELS] load_faiss_index called")
    print(f"[DEBUG MODELS] INDEX_FILE path: {INDEX_FILE}")
    print(f"[DEBUG MODELS] File exists: {os.path.exists(INDEX_FILE)}")
    print(f"[DEBUG MODELS] create_if_missing: {create_if_missing}")
    
    try:
        import faiss  # type: ignore
    except Exception as e:
        print(f"[ERROR MODELS] Failed to import faiss: {str(e)}")
        raise RuntimeError("faiss is not installed. Please install faiss to use the vector index.") from e

    if os.path.exists(INDEX_FILE):
        try:
            print("[DEBUG MODELS] Loading existing FAISS index...")
            index = faiss.read_index(INDEX_FILE)
            print(f"[DEBUG MODELS] FAISS index loaded. Total vectors: {index.ntotal}")
            return index
        except Exception as e:
            print(f"[ERROR MODELS] Failed to load FAISS index: {str(e)}")
            raise
    
    if not create_if_missing:
        print(f"[ERROR MODELS] Index file not found: {INDEX_FILE}")
        raise FileNotFoundError(f"FAISS index not found at {INDEX_FILE}")
    
    print("[DEBUG MODELS] Creating new empty FAISS index...")
    dim = get_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    print(f"[DEBUG MODELS] New FAISS index created with dimension: {dim}")
    return index


def save_faiss_index(index) -> None:
    print("[DEBUG MODELS] save_faiss_index called")
    print(f"[DEBUG MODELS] Saving index to: {INDEX_FILE}")
    print(f"[DEBUG MODELS] Index total vectors: {index.ntotal}")
    try:
        import faiss  # type: ignore
    except Exception as e:
        print(f"[ERROR MODELS] Failed to import faiss: {str(e)}")
        raise RuntimeError("faiss is not installed. Please install faiss to use the vector index.") from e
    
    try:
        faiss.write_index(index, INDEX_FILE)
        print("[DEBUG MODELS] FAISS index saved successfully")
    except Exception as e:
        print(f"[ERROR MODELS] Failed to save FAISS index: {str(e)}")
        raise


def load_metadata_store() -> Dict:
    print("[DEBUG MODELS] load_metadata_store called")
    print(f"[DEBUG MODELS] METADATA_FILE path: {METADATA_FILE}")
    print(f"[DEBUG MODELS] File exists: {os.path.exists(METADATA_FILE)}")
    
    if not os.path.exists(METADATA_FILE):
        print("[DEBUG MODELS] Metadata file not found, returning empty dict")
        return {}
    
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            store = json.load(f)
        print(f"[DEBUG MODELS] Metadata loaded. Total entries: {len(store)}")
        return store
    except Exception as e:
        print(f"[ERROR MODELS] Failed to load metadata: {str(e)}")
        raise


def save_metadata_store(store: Dict) -> None:
    print("[DEBUG MODELS] save_metadata_store called")
    print(f"[DEBUG MODELS] Saving {len(store)} metadata entries to: {METADATA_FILE}")
    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
        print("[DEBUG MODELS] Metadata saved successfully")
    except Exception as e:
        print(f"[ERROR MODELS] Failed to save metadata: {str(e)}")
        raise


def load_bm25_data() -> tuple:
    """Load BM25 corpus and document IDs from disk."""
    print("[DEBUG MODELS] load_bm25_data called")
    print(f"[DEBUG MODELS] BM25_CORPUS_FILE: {BM25_CORPUS_FILE}")
    print(f"[DEBUG MODELS] BM25_IDS_FILE: {BM25_IDS_FILE}")
    
    corpus = []
    doc_ids = []
    
    if os.path.exists(BM25_CORPUS_FILE) and os.path.exists(BM25_IDS_FILE):
        try:
            with open(BM25_CORPUS_FILE, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            with open(BM25_IDS_FILE, "r", encoding="utf-8") as f:
                doc_ids = json.load(f)
            print(f"[DEBUG MODELS] BM25 data loaded. Corpus: {len(corpus)} docs, IDs: {len(doc_ids)}")
        except Exception as e:
            print(f"[ERROR MODELS] Failed to load BM25 data: {str(e)}")
            corpus = []
            doc_ids = []
    else:
        print("[DEBUG MODELS] BM25 data files not found, will initialize from metadata")
    
    return corpus, doc_ids


def save_bm25_data(corpus: list, doc_ids: list) -> None:
    """Save BM25 corpus and document IDs to disk."""
    print("[DEBUG MODELS] save_bm25_data called")
    print(f"[DEBUG MODELS] Saving {len(corpus)} corpus entries and {len(doc_ids)} IDs")
    
    try:
        with open(BM25_CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        with open(BM25_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(doc_ids, f, indent=2, ensure_ascii=False)
        print("[DEBUG MODELS] BM25 data saved successfully")
    except Exception as e:
        print(f"[ERROR MODELS] Failed to save BM25 data: {str(e)}")
        raise


def get_whisper_model(model_size: str = "base"):
    global _whisper_model
    try:
        import whisper  # type: ignore
    except Exception as e:
        raise RuntimeError("openai-whisper is not installed. Please install it to use ASR.") from e

    if _whisper_model is None or getattr(_whisper_model, "model_name", None) != model_size:
        _whisper_model = whisper.load_model(model_size)
        try:
            _whisper_model.model_name = model_size  # type: ignore[attr-defined]
        except Exception:
            pass
    return _whisper_model


__all__ = [
    "get_embedding_model",
    "get_embedding_dimension",
    "load_faiss_index",
    "save_faiss_index",
    "load_metadata_store",
    "save_metadata_store",
    "load_bm25_data",
    "save_bm25_data",
    "get_whisper_model",
    "OLLAMA_VL_MODEL",
    "INDEX_FILE",
    "METADATA_FILE",
    "BM25_CORPUS_FILE",
    "BM25_IDS_FILE",
]
