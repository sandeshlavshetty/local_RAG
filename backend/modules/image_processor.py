import ollama
import os
from dotenv import load_dotenv

print("[DEBUG IMAGE] Initializing image_processor...")
load_dotenv()

def describe_image(image_path):
    print(f"\n[DEBUG IMAGE] describe_image called")
    print(f"[DEBUG IMAGE] Image path: {image_path}")
    print(f"[DEBUG IMAGE] File exists: {os.path.exists(image_path)}")
    
    try:
        prompt = "Describe this image in detail. If text is present, extract it."
        model = os.getenv("OLLAMA_VL_MODEL", "qwen2.5vl:7b")
        print(f"[DEBUG IMAGE] Calling ollama.chat with model: {model}")
        print("[DEBUG IMAGE] Sending image to Ollama...")
        
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are an image caption and OCR assistant."},
                {"role": "user", "content": prompt, "images": [image_path]}
            ]
        )
        
        print("[DEBUG IMAGE] Ollama response received")
        content = response["message"]["content"]
        print(f"[DEBUG IMAGE] Response length: {len(content)}")
        return content
    except Exception as e:
        print(f"[ERROR IMAGE] describe_image failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
