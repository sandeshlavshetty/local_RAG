from modules.models import load_bm25_data

print("Testing BM25 persistence...")
corpus, ids = load_bm25_data()
print(f"Loaded corpus: {len(corpus)} documents")
print(f"Loaded IDs: {len(ids)} documents")
if corpus:
    print(f"First document tokens: {corpus[0]}")
    print(f"First ID: {ids[0]}")
print("BM25 persistence test complete!")