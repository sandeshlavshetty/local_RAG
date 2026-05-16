#!/usr/bin/env python3
"""
Test script for the enhanced retriever with multiple retrieval methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.retriever import retrieve_chunks, retrieve_answer

def test_retrieval_methods():
    """Test different retrieval methods."""
    test_query = "What is machine learning?"

    print("Testing retrieval methods...\n")

    methods = ["semantic", "bm25", "keyword", "tfidf", "hybrid"]

    for method in methods:
        print(f"\n=== Testing {method.upper()} retrieval ===")
        try:
            results = retrieve_chunks(test_query, top_k=2, method=method)
            print(f"Retrieved {len(results)} chunks")
            if results:
                print(f"First result preview: {results[0][0][:100]}...")
        except Exception as e:
            print(f"Error with {method}: {str(e)}")

def test_full_rag():
    """Test full RAG pipeline with different methods."""
    test_query = "Explain artificial intelligence"

    print("\n\n=== Testing Full RAG Pipeline ===")

    methods = ["semantic", "hybrid"]

    for method in methods:
        print(f"\n--- Testing {method.upper()} RAG ---")
        try:
            answer = retrieve_answer(test_query, top_k=2, retrieval_method=method)
            print(f"Answer length: {len(answer.get('answer', ''))}")
            print(f"Citations: {len(answer.get('citations', []))}")
            print(f"Answer preview: {answer.get('answer', '')[:150]}...")
        except Exception as e:
            print(f"Error with {method} RAG: {str(e)}")

if __name__ == "__main__":
    test_retrieval_methods()
    test_full_rag()