from modules.retriever import update_bm25_index
from modules.models import load_metadata_store

print("Rebuilding BM25 index from current metadata...")

metadata_store = load_metadata_store()
print(f"Found {len(metadata_store)} documents in metadata")

# Trigger rebuild by calling update with a dummy document
# This will rebuild the entire index
if metadata_store:
    # Get the last document to trigger rebuild
    last_chunk_id = list(metadata_store.keys())[-1]
    last_text = metadata_store[last_chunk_id].get("text_excerpt", "")
    update_bm25_index(last_chunk_id, last_text)
    print("BM25 index rebuilt successfully")
else:
    print("No metadata found")