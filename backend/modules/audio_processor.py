import os
import torch
from faster_whisper import WhisperModel

print("\n[DEBUG AUDIO] Initializing audio_processor...")
# Pick device based on CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"[DEBUG AUDIO] Device: {DEVICE}, Compute type: {COMPUTE_TYPE}")

# Use model size string and local cache directory
MODEL_SIZE = "small"  # Change to "medium", "large-v2", etc. if needed
LOCAL_DIR = "local_models"

print(f"[DEBUG AUDIO] Loading Faster-Whisper model '{MODEL_SIZE}' with cache in {LOCAL_DIR} on {DEVICE} ...")
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=LOCAL_DIR)
    print("[DEBUG AUDIO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR AUDIO] Failed to load Whisper model: {str(e)}")
    raise

def transcribe_audio(audio_path):
    """
    Transcribe audio using locally cached faster-whisper model (no timestamps).
    """
    print(f"\n[DEBUG AUDIO] transcribe_audio called")
    print(f"[DEBUG AUDIO] Audio path: {audio_path}")
    print(f"[DEBUG AUDIO] File exists: {os.path.exists(audio_path)}")
    
    if not os.path.exists(audio_path):
        print(f"[ERROR AUDIO] Audio file not found at {audio_path}")
        return ""
    
    file_size = os.path.getsize(audio_path)
    print(f"[DEBUG AUDIO] File size: {file_size} bytes")
    
    if file_size == 0:
        print(f"[ERROR AUDIO] Audio file is empty (0 bytes)")
        return ""
    
    try:
        print("[DEBUG AUDIO] Starting transcription with Faster-Whisper...")
        print(f"[DEBUG AUDIO] Model size: {MODEL_SIZE}")
        print(f"[DEBUG AUDIO] Device: {DEVICE}, Compute type: {COMPUTE_TYPE}")
        
        segments, info = model.transcribe(audio_path, beam_size=2, vad_filter=True)
        print("[DEBUG AUDIO] Model.transcribe() completed")
        print(f"[DEBUG AUDIO] Info: {info}")
        
        segment_list = list(segments)
        print(f"[DEBUG AUDIO] Total segments returned: {len(segment_list)}")
        
        if len(segment_list) == 0:
            print(f"[DEBUG AUDIO] ⚠️ WARNING: No segments detected in audio!")
            return ""
        
        transcript_parts = []
        for i, segment in enumerate(segment_list):
            print(f"[DEBUG AUDIO] Segment {i}: text='{segment.text}', confidence={getattr(segment, 'confidence', 'N/A')}")
            if segment.text and segment.text.strip():
                transcript_parts.append(segment.text.strip())
        
        transcript = " ".join(transcript_parts)
        print(f"[DEBUG AUDIO] Total transcript length: {len(transcript)} characters")
        print(f"[DEBUG AUDIO] Transcript: {transcript}")
        
        if not transcript or not transcript.strip():
            print(f"[DEBUG AUDIO] ⚠️ WARNING: Transcription resulted in empty text!")
            return ""
        
        return transcript
    except Exception as e:
        print(f"[ERROR AUDIO] Transcription failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
