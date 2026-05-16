import fitz  # this comes from PyMuPDF

from PIL import Image
import os
import ollama

print("[DEBUG PDF] Initializing pdf_processor...")

def extract_page_with_vllm(page, page_number):
    print(f"\n[DEBUG PDF] extract_page_with_vllm called for page {page_number}")
    
    try:
        print("[DEBUG PDF] Converting page to image...")
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        temp_path = f"temp/page_{page_number}.png"
        os.makedirs("temp", exist_ok=True)
        img.save(temp_path, "PNG")
        print(f"[DEBUG PDF] Page saved to: {temp_path}")
        
        prompt = (
            "Extract all text from this page. "
            "If a region contains an image with text, read it. "
            "If a region contains a non-text image, describe it in detail."
        )
        
        model = os.getenv("OLLAMA_VL_MODEL", "gemma3:4b")
        print(f"[DEBUG PDF] Calling ollama.chat with model: {model} for page {page_number}")
        print("[DEBUG PDF] Sending page to Ollama...")
        
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a document OCR and image description assistant."},
                {"role": "user", "content": prompt, "images": [temp_path]}
            ]
        )
        
        print(f"[DEBUG PDF] Ollama response received for page {page_number}")
        content = response["message"]["content"]
        print(f"[DEBUG PDF] Response length: {len(content)}")
        return content
    except Exception as e:
        print(f"[ERROR PDF] extract_page_with_vllm failed for page {page_number}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_pdf(pdf_path):
    print(f"\n[DEBUG PDF] process_pdf called")
    print(f"[DEBUG PDF] PDF path: {pdf_path}")
    print(f"[DEBUG PDF] File exists: {os.path.exists(pdf_path)}")
    
    try:
        print("[DEBUG PDF] Opening PDF...")
        doc = fitz.open(pdf_path)
        print(f"[DEBUG PDF] PDF opened. Total pages: {len(doc)}")
        
        results = []
        for i, page in enumerate(doc):
            print(f"\n[DEBUG PDF] Processing page {i+1}/{len(doc)}...")
            vllm_text = extract_page_with_vllm(page, i+1)
            results.append({
                "page": i+1,
                "text": vllm_text
            })
            print(f"[DEBUG PDF] Page {i+1} completed")
        
        doc.close()
        print(f"[DEBUG PDF] PDF processing completed. Total pages processed: {len(results)}")
        return results
    except Exception as e:
        print(f"[ERROR PDF] process_pdf failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
