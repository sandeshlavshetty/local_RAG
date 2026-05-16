import os
import json
import shutil
import mimetypes
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from modules.rag_pipeline import process_inputs
from modules.embedding_store import add_to_index, save_index, METADATA_FILE
from modules.pdf_processor import process_pdf
from modules.image_processor import describe_image
from modules.audio_processor import transcribe_audio
from modules.retriever import retrieve_answer
from modules.utils import chunk_text_with_overlap
from config import BASE_DIR, VECTOR_DIR, PROCESSED_DIR
from modules.models import get_embedding_dimension

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

router = APIRouter()

# Function to save audio transcriptions for debugging
def save_audio_transcription(filename: str, transcription: str, timestamp: str):
    """Save transcribed audio to a JSON file for debugging."""
    os.makedirs("debug_audio", exist_ok=True)
    debug_file = "debug_audio/audio_transcriptions.json"
    
    # Load existing transcriptions or create new list
    transcriptions = []
    if os.path.exists(debug_file):
        try:
            with open(debug_file, "r") as f:
                transcriptions = json.load(f)
        except:
            transcriptions = []
    
    # Add new transcription
    transcriptions.append({
        "timestamp": timestamp,
        "filename": filename,
        "transcription": transcription,
        "length": len(transcription)
    })
    
    # Save back to file
    with open(debug_file, "w") as f:
        json.dump(transcriptions, f, indent=2)
    
    print(f"[DEBUG] Audio transcription saved to {debug_file}")

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (PDF/Image/Audio), process & store in FAISS.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    save_path = f"data/uploads/{file.filename}"
    os.makedirs("data/uploads", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    if file_ext == ".txt":
        results = process_pdf(save_path)
        for r in results:
            chunks = chunk_text_with_overlap(r["text"])
            for chunk in chunks:
                add_to_index(chunk, "text", file.filename, page_num=r["page"])

    elif file_ext in [".png", ".jpg", ".jpeg"]:
        caption = describe_image(save_path)
        chunks = chunk_text_with_overlap(caption)
        for chunk in chunks:
            add_to_index(chunk, "image", file.filename)

    elif file_ext in [".mp3", ".wav", ".m4a"]:
        print(f"\n[DEBUG UPLOAD] Processing audio file: {file.filename}")
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        transcript = transcribe_audio(save_path)
        print(f"[DEBUG UPLOAD] Transcript length: {len(transcript)}")
        print(f"[DEBUG UPLOAD] Transcript preview: {transcript[:200]}...")
        
        # Save transcription to JSON for debugging
        save_audio_transcription(file.filename, transcript, timestamp)
        
        chunks = chunk_text_with_overlap(transcript)
        print(f"[DEBUG UPLOAD] Created {len(chunks)} chunks from audio transcript")
        
        for i, chunk in enumerate(chunks):
            print(f"[DEBUG UPLOAD] Adding chunk {i+1}/{len(chunks)}, chunk length: {len(chunk)}")
            add_to_index(chunk, "audio", file.filename)
        
        print(f"[DEBUG UPLOAD] All audio chunks added to index")

    else:
        return {"error": "Unsupported file type"}

    print(f"[DEBUG UPLOAD] Calling save_index()")
    save_index()
    print(f"[DEBUG UPLOAD] Upload complete for {file.filename}")
    return {"message": f"{file.filename} processed & indexed."}


@router.post("/ask/")
async def ask_question(
    query: str = Form(...), 
    file: UploadFile | None = File(None),
    retrieval_method: str = Form("hybrid", description="Retrieval method: semantic, bm25, keyword, tfidf, hybrid")
):
    """
    Ask a question. If a PDF/Image/Audio file is attached, first extract its
    content (like in /upload but without indexing) and append to the query.
    Then retrieve the answer from FAISS + LLM.
    
    retrieval_method options:
    - semantic: FAISS semantic search only
    - bm25: BM25 algorithm only
    - keyword: Keyword matching only
    - tfidf: TF-IDF similarity only
    - hybrid: Combined semantic + BM25 + keyword (default)
    """
    print("\n=== /ask/ ENDPOINT CALLED ===")
    print(f"[DEBUG] Query: {query[:100]}..." if len(query) > 100 else f"[DEBUG] Query: {query}")
    print(f"[DEBUG] File attached: {file is not None}")
    print(f"[DEBUG] Retrieval method: {retrieval_method}")
    if file:
        print(f"[DEBUG] File name: {file.filename}")
    
    augmented_query = query

    # If a file is provided, extract its content and append to the query
    if file is not None:
        print("\n[DEBUG] File processing started...")
        print(f"[DEBUG] File object received: {file}")
        print(f"[DEBUG] File filename: {file.filename}")
        try:
            file_ext = os.path.splitext(file.filename)[1].lower()
            print(f"[DEBUG] ✓ File extension extracted: '{file_ext}'")
            print(f"[DEBUG] ✓ Full filename: '{file.filename}'")
            print(f"[DEBUG] ✓ File extension type: {type(file_ext)}")
            
            os.makedirs("data/uploads", exist_ok=True)
            # Use a unique filename to avoid collisions
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            save_path = f"data/uploads/ask_{timestamp}_{file.filename}"
            print(f"[DEBUG] File save path: {save_path}")

            with open(save_path, "wb") as f:
                f.write(await file.read())
            print(f"[DEBUG] File saved successfully")

            extracted_text = None

            print(f"[DEBUG] === CONDITION CHECK ===")
            print(f"[DEBUG] Checking: file_ext == '.pdf' ? {file_ext} == '.pdf' ? {file_ext == '.pdf'}")
            print(f"[DEBUG] Checking: file_ext in ['.png', '.jpg', '.jpeg'] ? {file_ext in ['.png', '.jpg', '.jpeg']}")
            print(f"[DEBUG] Checking: file_ext in ['.mp3', '.wav', ...] ? {file_ext in ['.mp3', '.wav', '.m4a', '.webm', '.ogg', '.flac']}")
            
            if file_ext == ".pdf":
                print("[DEBUG] ✓ ENTERED PDF CONDITION")
                results = process_pdf(save_path)  # list of {"page": int, "text": str}
                print(f"[DEBUG] PDF processed, got {len(results)} results")
                extracted_text = "\n\n".join(r["text"] for r in results if r.get("text"))
                print(f"[DEBUG] Extracted text length: {len(extracted_text)}")

            elif file_ext in [".png", ".jpg", ".jpeg"]:
                print("[DEBUG] Processing image...")
                extracted_text = describe_image(save_path)
                print(f"[DEBUG] Image caption length: {len(extracted_text)}")

            elif file_ext in [".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac"]:
                print(f"[DEBUG] Processing audio file with extension: {file_ext}")
                print(f"[DEBUG] Audio file size: {os.path.getsize(save_path)} bytes")
                try:
                    extracted_text = transcribe_audio(save_path)
                    print(f"[DEBUG] Audio transcription completed")
                    print(f"[DEBUG] Audio transcription length: {len(extracted_text)}")
                    if extracted_text:
                        print(f"[DEBUG] Audio transcription preview: {extracted_text[:200]}...")
                        # Save transcription to JSON for debugging
                        save_audio_transcription(file.filename, extracted_text, timestamp)
                    else:
                        print(f"[DEBUG] ⚠️ WARNING: Audio transcription returned empty string!")
                        # Still save empty transcription for debugging
                        save_audio_transcription(file.filename, "", timestamp)
                except Exception as e:
                    print(f"[DEBUG] ⚠️ Audio transcription exception: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    extracted_text = None
                    # Save error state
                    save_audio_transcription(file.filename, f"[ERROR] {str(e)}", timestamp)

            else:
                print(f"[DEBUG] Unsupported file type: {file_ext}")
                return {"error": "Unsupported file type"}

            if extracted_text and extracted_text.strip():
                print(f"[DEBUG] ✓ Text extracted successfully (length: {len(extracted_text)})")
                print(f"[DEBUG] Original query: {query}")
                augmented_query = (
                    f"{query}\n\nAdditional context from attached file ({file.filename}):\n{extracted_text}"
                )
                print(f"[DEBUG] Augmented query length: {len(augmented_query)}")
                print(f"[DEBUG] Augmented query preview:\n{augmented_query[:300]}...")
            else:
                print(f"[DEBUG] ✗ No text extracted from file or text is empty")
                print(f"[DEBUG] extracted_text type: {type(extracted_text)}, value: '{extracted_text}'")
                print(f"[DEBUG] ⚠️ WARNING: File was processed but no text was extracted!")
                print(f"[DEBUG] Will use original query only: {query}")
                augmented_query = query
        except Exception as e:
            print(f"[ERROR] File processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    print(f"\n[DEBUG] === QUERY SUMMARY ===")
    print(f"[DEBUG] Using query (length: {len(augmented_query)})")
    print(f"[DEBUG] Query content:\n{augmented_query}")
    print(f"[DEBUG] Retrieval method: {retrieval_method}")
    print(f"[DEBUG] Calling retrieve_answer...")
    try:
        answer = retrieve_answer(augmented_query, retrieval_method=retrieval_method)
        print("[DEBUG] retrieve_answer completed successfully")
        print(f"[DEBUG] Answer length: {len(answer.get('answer', ''))}")
        print(f"[DEBUG] Citations count: {len(answer.get('citations', []))}")
        # answer is a dict {"answer": str, "citations": [...]}
        return answer
    except Exception as e:
        print(f"[ERROR] retrieve_answer failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


@router.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """
    Return a file from the uploads folder by filename.
    Safely resolves the path to prevent directory traversal.
    """
    uploads_dir = os.path.join(BASE_DIR, "data", "uploads")
    safe_name = os.path.basename(filename)
    file_path = os.path.join(uploads_dir, safe_name)

    real_uploads = os.path.realpath(uploads_dir)
    real_file = os.path.realpath(file_path)

    # Prevent path traversal and ensure file exists
    if not real_file.startswith(real_uploads) or not os.path.isfile(real_file):
        # Keep consistency with other routes that return JSON errors
        # You can switch to: raise HTTPException(status_code=404, detail=...) if preferred
        return {"error": f"File '{filename}' not found"}

    return FileResponse(real_file, filename=safe_name)


@router.delete("/reset/")
async def reset_all():
    """
    Danger: Clear all data.
    - Deletes uploads (data/uploads)
    - Deletes vectorstore files (index.faiss, metadata.json)
    - Deletes processed outputs (processed/ and data/processed)
    - Resets in-memory index/metadata and writes fresh empty files
    """
    uploads_dir = os.path.join(BASE_DIR, "data", "uploads")
    data_processed_dir = os.path.join(BASE_DIR, "data", "processed")
    temp_dir = os.path.join(BASE_DIR, "temp")

    deleted = {"files": 0, "dirs": 0}

    def rm_tree(path: str):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            deleted["dirs"] += 1

    def rm_file(path: str):
        if os.path.isfile(path):
            try:
                os.remove(path)
                deleted["files"] += 1
            except Exception:
                pass

    # 1) Clear uploads
    if os.path.isdir(uploads_dir):
        shutil.rmtree(uploads_dir, ignore_errors=True)
        os.makedirs(uploads_dir, exist_ok=True)

    # 2) Clear vectorstore (index + metadata, including nested dirs)
    if os.path.isdir(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR, ignore_errors=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)

    # 3) Clear processed outputs
    if os.path.isdir(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if os.path.isdir(data_processed_dir):
        shutil.rmtree(data_processed_dir, ignore_errors=True)
    os.makedirs(data_processed_dir, exist_ok=True)

    # 4) Clear temp folder used by PDF processing (if present)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 5) Reset in-memory embedding store and write fresh empty index/metadata
    try:
        from modules import embedding_store as es
        dim = get_embedding_dimension()
        if faiss is not None:
            es.index = faiss.IndexFlatL2(dim)
        else:
            es.index = None  # type: ignore
        es.metadata_store = {}
        # Write fresh empty index/metadata to prevent future read errors
        if faiss is not None and es.index is not None:
            faiss.write_index(es.index, os.path.join(VECTOR_DIR, "index.faiss"))
        with open(os.path.join(VECTOR_DIR, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({}, f)
    except Exception as e:
        # Non-fatal; continue
        pass

    return {
        "message": "All data cleared: uploads, vectorstore, processed outputs.",
        "uploads_dir": uploads_dir,
        "vectorstore_dir": VECTOR_DIR,
        "processed_dirs": [PROCESSED_DIR, data_processed_dir],
        "deleted_counts": deleted,
    }


@router.get("/documents/")
async def get_all_documents():
    """
    Get a list of all uploaded documents with their metadata.
    """
    try:
        # Check if metadata file exists
        if not os.path.exists(METADATA_FILE):
            return {"documents": [], "total_count": 0, "message": "No documents uploaded yet"}
        
        # Load metadata
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata_store = json.load(f)
        
        # Group by source file to get unique documents
        documents_map = {}
        
        for chunk_id, metadata in metadata_store.items():
            source_file = metadata["source_file"]
            modality = metadata["modality"]
            
            if source_file not in documents_map:
                # Get file stats if file exists
                file_path = f"data/uploads/{source_file}"
                file_size = None
                upload_date = None
                
                if os.path.exists(file_path):
                    file_stats = os.stat(file_path)
                    file_size = file_stats.st_size
                    upload_date = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                
                documents_map[source_file] = {
                    "filename": source_file,
                    "modality": modality,
                    "file_size": file_size,
                    "upload_date": upload_date,
                    "chunk_count": 0,
                    "pages": set() if modality == "text" else None
                }
            
            # Increment chunk count
            documents_map[source_file]["chunk_count"] += 1
            
            # Add page numbers for PDF files
            if modality == "text" and metadata.get("page_num"):
                documents_map[source_file]["pages"].add(metadata["page_num"])
        
        # Convert sets to sorted lists and prepare final response
        documents = []
        for doc_info in documents_map.values():
            if doc_info["pages"] is not None:
                doc_info["pages"] = sorted(list(doc_info["pages"]))
                doc_info["page_count"] = len(doc_info["pages"])
            else:
                doc_info.pop("pages", None)
                doc_info["page_count"] = None
            
            documents.append(doc_info)
        
        # Sort by upload date (most recent first)
        documents.sort(key=lambda x: x["upload_date"] or "", reverse=True)
        
        return {
            "documents": documents,
            "total_count": len(documents),
            "total_chunks": len(metadata_store)
        }
        
    except Exception as e:
        return {"error": f"Failed to retrieve documents: {str(e)}"}


@router.get("/documents/{filename}")
async def get_document_file(filename: str):
    uploads_dir = os.path.join(BASE_DIR, "data", "uploads")
    safe_name = os.path.basename(filename)
    file_path = os.path.join(uploads_dir, safe_name)

    real_uploads = os.path.realpath(uploads_dir)
    real_file = os.path.realpath(file_path)

    if not real_file.startswith(real_uploads) or not os.path.isfile(real_file):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    # Guess MIME type to help the browser render inline (pdf/image/audio)
    media_type = mimetypes.guess_type(real_file)[0] or "application/octet-stream"

    # Force inline display instead of download
    headers = {
        "X-Source-Path": os.path.relpath(real_file, BASE_DIR),
        "Content-Disposition": f'inline; filename="{safe_name}"'
    }
    # Do not pass filename= to avoid automatic 'attachment' disposition
    return FileResponse(real_file, media_type=media_type, headers=headers)
