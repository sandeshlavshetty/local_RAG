import ollama
from modules.models import (
    get_embedding_model,
    load_faiss_index,
    load_metadata_store,
    load_bm25_data,
    save_bm25_data,
)
import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import re
from collections import Counter

load_dotenv()

# Centralized loaders
EMBEDDING_MODEL = get_embedding_model()
index = load_faiss_index(create_if_missing=True)
metadata_store = load_metadata_store()

# BM25 and keyword matching setup
_bm25_index = None
_corpus = []
_doc_ids = []

def _initialize_bm25():
    """Initialize BM25 index from disk or metadata store."""
    global _bm25_index, _corpus, _doc_ids
    
    # First try to load from disk
    _corpus, _doc_ids = load_bm25_data()
    
    if _corpus and _doc_ids:
        # Load existing BM25 data from disk
        try:
            from rank_bm25 import BM25Okapi
            _bm25_index = BM25Okapi(_corpus)
            print(f"[DEBUG RETRIEVER] BM25 index loaded from disk with {len(_corpus)} documents")
            return
        except ImportError:
            print("[WARNING RETRIEVER] rank-bm25 not installed, BM25 retrieval disabled")
            _bm25_index = None
            return
    
    # If no saved data, initialize from metadata store
    if not metadata_store:
        return
    
    try:
        from rank_bm25 import BM25Okapi
        _corpus = []
        _doc_ids = []
        
        for chunk_id, metadata in metadata_store.items():
            text = metadata.get("text_excerpt", "")
            if text.strip():
                # Tokenize the text (simple whitespace split, can be improved)
                tokens = text.lower().split()
                _corpus.append(tokens)
                _doc_ids.append(chunk_id)
        
        if _corpus:
            _bm25_index = BM25Okapi(_corpus)
            # Save the newly created BM25 data to disk
            save_bm25_data(_corpus, _doc_ids)
            print(f"[DEBUG RETRIEVER] BM25 index initialized from metadata and saved to disk with {len(_corpus)} documents")
        else:
            print("[DEBUG RETRIEVER] No documents available for BM25 indexing")
    except ImportError:
        print("[WARNING RETRIEVER] rank-bm25 not installed, BM25 retrieval disabled")
        _bm25_index = None


def save_bm25_index():
    """Save current BM25 data to disk."""
    if _corpus and _doc_ids:
        save_bm25_data(_corpus, _doc_ids)
        print(f"[DEBUG RETRIEVER] BM25 data saved to disk: {len(_corpus)} documents")
    else:
        print("[DEBUG RETRIEVER] No BM25 data to save")


def update_bm25_index(new_chunk_id: str, new_text: str):
    """Rebuild BM25 index completely when new documents are added."""
    global _bm25_index, _corpus, _doc_ids
    
    print(f"\n[DEBUG RETRIEVER] === update_bm25_index CALLED ===")
    print(f"[DEBUG RETRIEVER] new_chunk_id: {new_chunk_id}")
    print(f"[DEBUG RETRIEVER] new_text length: {len(new_text)}")
    print(f"[DEBUG RETRIEVER] new_text preview: {new_text[:100]}...")
    
    if not new_text.strip():
        print("[DEBUG RETRIEVER] Text is empty, returning")
        return
    
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[WARNING RETRIEVER] rank-bm25 not installed, cannot update BM25")
        return
    
    # Rebuild entire BM25 index from current metadata store
    # This ensures IDF calculations are correct for all documents
    print(f"[DEBUG RETRIEVER] Rebuilding BM25 index after adding document: {new_chunk_id}")
    print(f"[DEBUG RETRIEVER] Current metadata_store size: {len(metadata_store)}")
    
    _corpus = []
    _doc_ids = []
    
    for chunk_id, metadata in metadata_store.items():
        text = metadata.get("text_excerpt", "")
        if text.strip():
            # Tokenize the text (simple whitespace split, can be improved)
            tokens = text.lower().split()
            _corpus.append(tokens)
            _doc_ids.append(chunk_id)
            print(f"[DEBUG RETRIEVER] Added doc {chunk_id[:8]}... tokens: {len(tokens)}, modality: {metadata.get('modality')}")
    
    if _corpus:
        _bm25_index = BM25Okapi(_corpus)
        # Save the complete rebuilt BM25 data to disk
        save_bm25_data(_corpus, _doc_ids)
        print(f"[DEBUG RETRIEVER] BM25 index rebuilt with {len(_corpus)} documents and saved to disk")
        print(f"[DEBUG RETRIEVER] === update_bm25_index COMPLETE ===\n")
    else:
        print("[DEBUG RETRIEVER] No documents available for BM25 indexing")

def retrieve_with_bm25(query: str, top_k: int = 3) -> List[Tuple[str, Dict]]:
    """Retrieve chunks using BM25 algorithm."""
    if _bm25_index is None:
        _initialize_bm25()
    
    if _bm25_index is None or not _corpus:
        return []
    
    # Tokenize query
    query_tokens = query.lower().split()
    
    # Get BM25 scores
    bm25_scores = _bm25_index.get_scores(query_tokens)
    
    # Get top-k results
    top_indices = bm25_scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        score = bm25_scores[idx]
        if score > 0:  # Only include relevant results
            chunk_id = _doc_ids[idx]
            if chunk_id in metadata_store:
                meta = metadata_store[chunk_id]
                results.append((meta["text_excerpt"], meta))
    
    return results

def retrieve_with_keywords(query: str, top_k: int = 3) -> List[Tuple[str, Dict]]:
    """Retrieve chunks using keyword matching."""
    if not metadata_store:
        return []
    
    query_lower = query.lower()
    # Extract keywords (words longer than 3 characters)
    keywords = [word for word in re.findall(r'\b\w+\b', query_lower) if len(word) > 2]
    
    if not keywords:
        return []
    
    results = []
    for chunk_id, metadata in metadata_store.items():
        text = metadata.get("text_excerpt", "").lower()
        if not text:
            continue
            
        # Count keyword matches
        match_count = 0
        matched_keywords = set()
        
        for keyword in keywords:
            if keyword in text:
                match_count += text.count(keyword)
                matched_keywords.add(keyword)
        
        if match_count > 0:
            # Score based on match count and keyword coverage
            coverage_score = len(matched_keywords) / len(keywords)
            total_score = match_count * coverage_score
            
            results.append((total_score, (metadata["text_excerpt"], metadata)))
    
    # Sort by score and take top-k
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = [item[1] for item in results[:top_k]]
    
    return top_results

def retrieve_with_tfidf(query: str, top_k: int = 3) -> List[Tuple[str, Dict]]:
    """Retrieve chunks using TF-IDF similarity."""
    if not metadata_store:
        return []
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("[WARNING RETRIEVER] scikit-learn not available for TF-IDF")
        return []
    
    # Prepare documents
    documents = []
    doc_ids = []
    for chunk_id, metadata in metadata_store.items():
        text = metadata.get("text_excerpt", "")
        if text.strip():
            documents.append(text)
            doc_ids.append(chunk_id)
    
    if not documents:
        return []
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Transform query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top-k results
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Threshold for relevance
            chunk_id = doc_ids[idx]
            if chunk_id in metadata_store:
                meta = metadata_store[chunk_id]
                results.append((meta["text_excerpt"], meta))
    
    return results

def retrieve_chunks(query, top_k=3, method="hybrid"):
    """
    Retrieve top_k chunks using specified method.
    Methods: 'semantic' (FAISS), 'bm25', 'keyword', 'tfidf', 'hybrid' (combines all)
    """
    if method == "semantic":
        return retrieve_with_semantic(query, top_k)
    elif method == "bm25":
        return retrieve_with_bm25(query, top_k)
    elif method == "keyword":
        return retrieve_with_keywords(query, top_k)
    elif method == "tfidf":
        return retrieve_with_tfidf(query, top_k)
    elif method == "hybrid":
        return retrieve_hybrid(query, top_k)
    else:
        return retrieve_hybrid(query, top_k)

def retrieve_with_semantic(query, top_k=3):
    """Retrieve chunks using semantic search (FAISS)."""
    try:
        ntotal = getattr(index, 'ntotal', 0)
    except Exception as e:
        print(f"[ERROR RETRIEVER] Failed to get index ntotal: {str(e)}")
        ntotal = 0

    if ntotal == 0 or not metadata_store:
        return []

    try:
        q_emb = EMBEDDING_MODEL.encode([query])
        D, I = index.search(q_emb, top_k)
        
        results = []
        all_keys = list(metadata_store.keys())
        
        for idx in I[0]:
            chunk_id = all_keys[idx]
            meta = metadata_store[chunk_id]
            results.append((meta["text_excerpt"], meta))
        
        return results
    except Exception as e:
        print(f"[ERROR RETRIEVER] Semantic search failed: {str(e)}")
        return []

def retrieve_hybrid(query, top_k=5):
    """
    Hybrid retrieval combining semantic search, BM25, and keyword matching.
    Returns deduplicated results ranked by combined scores.
    """
    # Get results from different methods
    semantic_results = retrieve_with_semantic(query, top_k=top_k)
    bm25_results = retrieve_with_bm25(query, top_k=top_k)
    keyword_results = retrieve_with_keywords(query, top_k=top_k)
    
    # Combine and deduplicate results
    combined_results = {}
    
    # Add semantic results with high weight
    for i, (text, meta) in enumerate(semantic_results):
        chunk_id = meta.get("id", f"semantic_{i}")
        score = (top_k - i) * 3.0  # Higher weight for semantic
        combined_results[chunk_id] = {
            "text": text,
            "meta": meta,
            "score": score,
            "methods": ["semantic"]
        }
    
    # Add BM25 results
    for i, (text, meta) in enumerate(bm25_results):
        chunk_id = meta.get("id", f"bm25_{i}")
        score = (top_k - i) * 2.0  # Medium weight for BM25
        if chunk_id in combined_results:
            combined_results[chunk_id]["score"] += score
            combined_results[chunk_id]["methods"].append("bm25")
        else:
            combined_results[chunk_id] = {
                "text": text,
                "meta": meta,
                "score": score,
                "methods": ["bm25"]
            }
    
    # Add keyword results
    for i, (text, meta) in enumerate(keyword_results):
        chunk_id = meta.get("id", f"keyword_{i}")
        score = (top_k - i) * 1.0  # Lower weight for keywords
        if chunk_id in combined_results:
            combined_results[chunk_id]["score"] += score
            combined_results[chunk_id]["methods"].append("keyword")
        else:
            combined_results[chunk_id] = {
                "text": text,
                "meta": meta,
                "score": score,
                "methods": ["keyword"]
            }
    
    # Sort by combined score and return top-k
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]["score"], reverse=True)
    final_results = [(data["text"], data["meta"]) for chunk_id, data in sorted_results[:top_k]]
    
    return final_results

def retrieve_answer(query, top_k=3, retrieval_method="hybrid"):
    """
    LLM-driven RAG: Ask Ollama what information it needs,
    then retrieve relevant chunks using specified method, and answer with citations.
    
    retrieval_method options: 'semantic', 'bm25', 'keyword', 'tfidf', 'hybrid'
    """
    # If no documents are indexed, short-circuit with a helpful message
    try:
        index_count = getattr(index, 'ntotal', 0)
        if index_count == 0 or not metadata_store:
            return {
                "answer": (
                    "No documents are indexed yet. Please upload a PDF/Image/Audio via /upload "
                    "before asking questions."
                ),
                "citations": [],
                "query_used": ""
            }
    except Exception as e:
        print(f"[ERROR RETRIEVER] Exception checking index: {str(e)}")
        return {
            "answer": "Vector index unavailable. Please upload a PDF/Image/Audio via /upload to initialize the index.",
            "citations": [],
            "query_used": ""
        }

    # Step 1: Ask Ollama what context it needs
    instruction = (
        "You are an assistant with access to a vector store of documents.\n"
        "Decide what information you need to answer the user's question. "
        "Return a clear query or keywords for retrieval.\n"
        f"User Question: {query}"
    )
    
    try:
        retrieval_hint = ollama.chat(
            model=os.getenv("OLLAMA_VL_MODEL", "qwen3:8b"),
            messages=[
                {"role": "system", "content": "You are a helpful expert assistant."},
                {"role": "user", "content": instruction}
            ]
        )["message"]["content"]
    except Exception as e:
        print(f"[ERROR RETRIEVER] Failed to generate retrieval hint: {str(e)}")
        return {
            "answer": "Error generating retrieval query. Please try again.",
            "citations": [],
            "query_used": ""
        }

    # Step 2: Retrieve chunks
    try:
        chunks = retrieve_chunks(retrieval_hint, top_k, method=retrieval_method)
    except Exception as e:
        print(f"[ERROR RETRIEVER] Failed to retrieve chunks: {str(e)}")
        return {
            "answer": "Error retrieving information from documents. Please try again.",
            "citations": [],
            "query_used": retrieval_hint
        }

    # Step 3: Build context and citations
    context_text = ""
    citations = []
    for chunk, meta in chunks:
        context_text += f"[Source: {meta['source_file']}, page: {meta.get('page_num', 'N/A')}]\n{chunk}\n\n"
        citations.append({
            "text": chunk,
            "source_file": meta.get("source_file"),
            "page_num": meta.get("page_num")
        })

    # Step 4: Get final answer from Ollama
    final_prompt = (
        f"Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context_text}\nQuestion: {query}"
    )

    try:
        response = ollama.chat(
            model=os.getenv("OLLAMA_VL_MODEL", "qwen3:8b"),
            messages=[
                {"role": "system", "content": "You are a helpful expert assistant."},
                {"role": "user", "content": final_prompt}
            ]
        )
        answer_content = response["message"]["content"]

        return {
            "answer": answer_content,
            "citations": citations,
            "query_used": retrieval_hint
        }
    except Exception as e:
        print(f"[ERROR RETRIEVER] Failed to generate answer: {str(e)}")
        return {
            "answer": "Error generating answer. Please try again.",
            "citations": citations,
            "query_used": retrieval_hint
        }

# Example usage
if __name__ == "__main__":
    user_query = "What are the key points about caching in web applications?"
    answer = retrieve_answer(user_query, top_k=3)
    print("Answer with citations:\n", answer)
