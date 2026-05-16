from modules.retriever import retrieve_with_bm25

print('Testing BM25 retrieval with audio content...')
results = retrieve_with_bm25('pad smashed face', 5)
print(f'Retrieved {len(results)} results')

for i, (text, meta) in enumerate(results):
    print(f'Result {i+1}: {text[:100]}...')
    print(f'Modality: {meta.get("modality")}, Source: {meta.get("source_file")}')
    print()