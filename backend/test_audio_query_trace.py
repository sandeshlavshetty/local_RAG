#!/usr/bin/env python3
"""
Test script to trace audio query pipeline
Usage: python test_audio_query_pipeline.py
"""

print("\n" + "="*80)
print("AUDIO QUERY PIPELINE TRACE TEST")
print("="*80)

print("\n[TEST] Checking file format support...")
audio_extensions = [".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac"]
print(f"Supported audio formats: {audio_extensions}")

test_files = [
    "recording_1775759276871.webm",
    "audio.wav",
    "audio.mp3",
    "audio.m4a"
]

print("\n[TEST] Checking which test files would be supported:")
for filename in test_files:
    ext = filename[filename.rfind('.'):]
    supported = "✓ yes" if ext in audio_extensions else "✗ NO"
    print(f"  {filename:40} {supported}")

print("\n[TEST] Simulating /ask/ endpoint with audio file...")
print("""
Expected pipeline when POST /ask/ with audio file:

1. File received with query "find document related to collision"
   └─ /ask/ endpoint receives file and query

2. File saved and processed
   └─ File extension detected (.webm, .wav, etc)
   └─ [DEBUG] Processing audio file with extension: .webm
   └─ transcribe_audio() is called
   └─ Audio converted to text

3. Query augmentation
   └─ Original query: "find document related to collision"
   └─ Audio transcription: "The pad smashed into my face..."
   └─ [DEBUG] ✓ Text extracted successfully
   └─ Augmented query created

4. retrieve_answer() called with augmented query
   └─ Full query sent (original + audio content)
   └─ [DEBUG] ✓ Multi-part query detected!
   └─ LLM generates retrieval hint
   └─ retrieve_chunks() called with hint
   └─ Results retrieved and returned

5. Answer generated
   └─ Using retrieved chunks + full query
   └─ LLM generates final answer
""")

print("\n[TEST] What to look for in logs:")
print("""
✓ GOOD (should see all of these):
  [DEBUG] Processing audio file with extension: .webm
  [DEBUG] Audio transcription completed
  [DEBUG] ✓ Text extracted successfully (length: XXX)
  [DEBUG] Augmented query length: XXX
  [DEBUG] === QUERY SUMMARY ===
  [DEBUG] Using query (length: XXX)
  [DEBUG] ✓ Multi-part query detected!
  [DEBUG] Retrieved 2 chunks using hybrid
  [DEBUG] Hybrid result - ID: xxxxx, Modality: audio

✗ PROBLEM (should NOT see these):
  [DEBUG] ✗ No text extracted from file
  [ERROR] File processing failed
  [DEBUG] Single query (no additional file context)
  [DEBUG] Retrieved 0 chunks
""")

print("\n[TEST] Audio format issue check:")
print(f"""
The uploaded file "recording_1775759276871.webm" has extension: .webm
Is .webm in supported list? {'YES ✓' if '.webm' in audio_extensions else 'NO ✗'}
""")

print("\n[TEST] Next steps:")
print("""
1. Check if uvicorn is running with the new code
2. Upload an audio file about collision
3. Send query: "find document related to collision" with the audio file
4. Check the backend logs for the traces above
5. Look for "✓ Multi-part query detected!" in logs
6. If you see "Retrieved 0 chunks", the audio content isn't matching any documents
""")

print("="*80 + "\n")
