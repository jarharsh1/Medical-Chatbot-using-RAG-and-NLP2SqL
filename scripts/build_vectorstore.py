"""
Build (or refresh) the ChromaDB vector index for CI and local setup.
Safe to run multiple times — incremental indexing skips already-embedded notes.
"""
from backend.rag.vectorstore import populate_vectorstore

if __name__ == "__main__":
    print("Building ChromaDB index...")
    populate_vectorstore()
    print("ChromaDB index ready.")
