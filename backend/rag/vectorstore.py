"""
ChromaDB vector store with incremental indexing.

Features:
  - Persistent storage at backend/chroma_db/
  - Bootstrap: embed all notes on first run
  - Incremental upsert: only embed new/changed notes on subsequent runs
  - Version-aware re-index: full re-index when embedding model changes
  - indexed_notes SQLite table tracks what's been embedded
"""

import hashlib
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from backend.config import (
    CHROMA_PATH,
    DB_PATH,
    EMBED_MODEL_VERSION,
    SEMANTIC_TOP_K,
)
from backend.rag.chunking import get_chunks
from backend.rag.embeddings import embed_texts

logger = logging.getLogger(__name__)

_chroma_client = None
_collection = None


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    """Get or create the clinical_notes collection."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name="clinical_notes",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _init_indexed_notes_table():
    """Create the indexed_notes tracking table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indexed_notes (
            note_id INTEGER PRIMARY KEY,
            content_hash TEXT NOT NULL,
            embed_version TEXT NOT NULL,
            indexed_at TIMESTAMP NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _get_indexed_state() -> Dict[int, Tuple[str, str]]:
    """Return {note_id: (content_hash, embed_version)} from indexed_notes table."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT note_id, content_hash, embed_version FROM indexed_notes")
    state = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    conn.close()
    return state


def _update_indexed_state(note_id: int, content_hash: str, embed_version: str):
    """Upsert a single entry in indexed_notes."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO indexed_notes (note_id, content_hash, embed_version, indexed_at)
        VALUES (?, ?, ?, ?)
    """, (note_id, content_hash, embed_version, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def _remove_indexed_state(note_ids: List[int]):
    """Remove entries from indexed_notes."""
    if not note_ids:
        return
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join(["?"] * len(note_ids))
    conn.execute(f"DELETE FROM indexed_notes WHERE note_id IN ({placeholders})", note_ids)
    conn.commit()
    conn.close()


def populate_vectorstore():
    """
    Main indexing entry point. Called on server startup.

    Strategy:
    1. If collection is empty → full bootstrap
    2. If embed_version changed → full re-index
    3. Otherwise → incremental (new/changed/deleted)
    """
    _init_indexed_notes_table()
    collection = get_collection()
    indexed_state = _get_indexed_state()

    # Check for version change → full re-index
    version_changed = False
    if indexed_state:
        sample_version = next(iter(indexed_state.values()))[1]
        if sample_version != EMBED_MODEL_VERSION:
            logger.warning(
                f"Embedding model version changed ({sample_version} → {EMBED_MODEL_VERSION}). "
                f"Triggering full re-index."
            )
            version_changed = True

    current_count = collection.count()
    needs_full_index = current_count == 0 or version_changed

    # Get chunks from SQLite
    chunks = get_chunks()
    chunk_map = {}  # note_id → Document
    for doc in chunks:
        note_id = doc.metadata.get("note_id")
        if note_id is not None:
            chunk_map[note_id] = doc

    if needs_full_index:
        _full_index(collection, chunks)
    else:
        _incremental_index(collection, chunk_map, indexed_state)


def _full_index(collection, chunks):
    """Embed and store all chunks from scratch."""
    if not chunks:
        logger.info("No chunks to index.")
        return

    # Clear existing data
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Clear indexed_notes table
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM indexed_notes")
    conn.commit()
    conn.close()

    logger.info(f"Full indexing: {len(chunks)} documents...")

    batch_size = 500
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i : i + batch_size]
        texts = [doc.page_content for doc in batch]
        ids = [doc.metadata["doc_id"] for doc in batch]
        metadatas = []
        for doc in batch:
            meta = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
            metadatas.append(meta)

        try:
            vectors = embed_texts(texts)
            collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )

            # Track in indexed_notes
            for doc in batch:
                note_id = doc.metadata.get("note_id")
                if note_id is not None:
                    _update_indexed_state(
                        note_id,
                        doc.metadata["content_hash"],
                        EMBED_MODEL_VERSION,
                    )
        except Exception as e:
            logger.error(f"Error indexing batch {i}: {e}")
            continue

    logger.info(f"Full indexing complete. Collection count: {collection.count()}")


def _incremental_index(collection, chunk_map, indexed_state):
    """Only embed new/changed notes, delete removed ones."""
    current_note_ids = set(chunk_map.keys())
    indexed_note_ids = set(indexed_state.keys())

    # New notes
    new_ids = current_note_ids - indexed_note_ids

    # Changed notes (content_hash differs)
    changed_ids = set()
    for nid in current_note_ids & indexed_note_ids:
        current_hash = chunk_map[nid].metadata["content_hash"]
        indexed_hash = indexed_state[nid][0]
        if current_hash != indexed_hash:
            changed_ids.add(nid)

    # Deleted notes
    deleted_ids = indexed_note_ids - current_note_ids

    to_upsert = new_ids | changed_ids

    if not to_upsert and not deleted_ids:
        logger.info("Vector store is up to date. No changes needed.")
        return

    logger.info(
        f"Incremental index: {len(new_ids)} new, {len(changed_ids)} changed, "
        f"{len(deleted_ids)} deleted"
    )

    # Delete removed notes from ChromaDB
    if deleted_ids:
        delete_doc_ids = [f"note:{nid}" for nid in deleted_ids]
        try:
            collection.delete(ids=delete_doc_ids)
        except Exception as e:
            logger.error(f"Error deleting from ChromaDB: {e}")
        _remove_indexed_state(list(deleted_ids))

    # Upsert new/changed
    if to_upsert:
        docs = [chunk_map[nid] for nid in to_upsert]
        batch_size = 500
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [doc.page_content for doc in batch]
            ids = [doc.metadata["doc_id"] for doc in batch]
            metadatas = []
            for doc in batch:
                meta = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
                metadatas.append(meta)

            try:
                vectors = embed_texts(texts)
                collection.upsert(
                    ids=ids,
                    embeddings=vectors,
                    documents=texts,
                    metadatas=metadatas,
                )
                for doc in batch:
                    note_id = doc.metadata.get("note_id")
                    if note_id is not None:
                        _update_indexed_state(
                            note_id,
                            doc.metadata["content_hash"],
                            EMBED_MODEL_VERSION,
                        )
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")

    logger.info(f"Incremental indexing complete. Collection count: {collection.count()}")


def semantic_search(
    query: str,
    top_k: int = SEMANTIC_TOP_K,
    where_filter: Optional[Dict] = None,
) -> List[Dict]:
    """
    Query ChromaDB for semantically similar documents.

    Returns list of {doc_id, content, metadata, score} dicts.
    """
    from backend.rag.embeddings import embed_query

    collection = get_collection()
    query_vector = embed_query(query)

    kwargs = {
        "query_embeddings": [query_vector],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)

    docs = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns distances; for cosine, similarity = 1 - distance
            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1.0 - distance

            docs.append({
                "doc_id": doc_id,
                "content": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": score,
            })

    return docs
