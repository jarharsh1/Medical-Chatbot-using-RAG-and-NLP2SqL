"""
Chunking module: converts clinical notes from SQLite into Documents with rich metadata.

Two modes:
  - "note": 1 clinical note = 1 chunk (default)
  - "encounter": concatenate all notes from same patient + visit_date

Each document gets:
  - Canonical doc_id: "note:{note_id}"
  - SHA-256 content_hash for dedup
  - Versioning metadata (embedding_model, embed_version, chunk_version)
"""

import hashlib
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from langchain_core.documents import Document

from backend.config import (
    CHUNK_MODE,
    CHUNK_VERSION,
    DB_PATH,
    EMBED_MODEL,
    EMBED_MODEL_VERSION,
)

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """SHA-256 hash of text content for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fetch_notes_with_metadata() -> List[Dict]:
    """
    Fetch all clinical notes enriched with patient and clinic metadata.
    Single SQL JOIN pass for efficiency.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT
            n.note_id,
            n.patient_id,
            n.visit_date,
            n.doctor_name,
            n.diagnosis_code,
            n.condition_name,
            n.note_text,
            p.full_name AS patient_name,
            c.name AS clinic_name
        FROM clinical_notes n
        JOIN patients p ON n.patient_id = p.patient_id
        JOIN clinics c ON p.clinic_id = c.clinic_id
    """
    rows = cur.execute(query).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def _build_metadata(note: Dict, doc_id: str, content_hash: str) -> Dict:
    """Build the full metadata dict for a document."""
    return {
        "doc_id": doc_id,
        "note_id": note["note_id"],
        "patient_id": note["patient_id"],
        "patient_name": note["patient_name"],
        "doctor_name": note["doctor_name"],
        "condition_name": note["condition_name"],
        "diagnosis_code": note["diagnosis_code"],
        "visit_date": note["visit_date"],
        "clinic_name": note["clinic_name"],
        "content_hash": content_hash,
        "embedding_model": EMBED_MODEL,
        "embed_version": EMBED_MODEL_VERSION,
        "chunk_version": CHUNK_VERSION,
        "indexed_at": datetime.utcnow().isoformat() + "Z",
    }


def chunk_notes_note_level(notes: List[Dict]) -> List[Document]:
    """
    Mode 1: Each clinical note = 1 Document.
    Dedup by content_hash (keep latest visit_date).
    """
    seen_hashes: Dict[str, Dict] = {}

    for note in notes:
        text = (note["note_text"] or "").strip()
        if not text:
            continue

        h = _content_hash(text)

        if h in seen_hashes:
            existing = seen_hashes[h]
            if note["visit_date"] > existing["visit_date"]:
                seen_hashes[h] = note
        else:
            seen_hashes[h] = note

    documents = []
    deduped_count = len(notes) - len(seen_hashes)
    if deduped_count > 0:
        logger.info(f"Deduped {deduped_count} duplicate clinical notes")

    for h, note in seen_hashes.items():
        text = note["note_text"].strip()
        doc_id = f"note:{note['note_id']}"
        metadata = _build_metadata(note, doc_id, h)
        documents.append(Document(page_content=text, metadata=metadata))

    logger.info(f"Chunked {len(documents)} notes (note-level mode)")
    return documents


def chunk_notes_encounter_level(notes: List[Dict]) -> List[Document]:
    """
    Mode 2: Group notes by (patient_id, visit_date) = 1 encounter.
    Concatenate with separators, preserve individual note_ids in metadata.
    """
    encounters = defaultdict(list)
    for note in notes:
        key = (note["patient_id"], note["visit_date"])
        encounters[key].append(note)

    documents = []
    for (patient_id, visit_date), enc_notes in encounters.items():
        enc_notes.sort(key=lambda n: n["note_id"])

        texts = []
        note_ids = []
        for n in enc_notes:
            text = (n["note_text"] or "").strip()
            if text:
                texts.append(text)
                note_ids.append(n["note_id"])

        if not texts:
            continue

        combined_text = "\n---\n".join(texts)
        h = _content_hash(combined_text)

        primary_note = enc_notes[0]
        doc_id = f"encounter:{patient_id}:{visit_date}"

        metadata = {
            "doc_id": doc_id,
            "encounter_notes": note_ids,
            "patient_id": patient_id,
            "patient_name": primary_note["patient_name"],
            "doctor_name": primary_note["doctor_name"],
            "condition_name": primary_note["condition_name"],
            "diagnosis_code": primary_note["diagnosis_code"],
            "visit_date": visit_date,
            "clinic_name": primary_note["clinic_name"],
            "content_hash": h,
            "embedding_model": EMBED_MODEL,
            "embed_version": EMBED_MODEL_VERSION,
            "chunk_version": CHUNK_VERSION,
            "indexed_at": datetime.utcnow().isoformat() + "Z",
        }

        documents.append(Document(page_content=combined_text, metadata=metadata))

    logger.info(f"Chunked {len(documents)} encounters (encounter-level mode)")
    return documents


def get_chunks() -> List[Document]:
    """
    Main entry point: fetch notes from SQLite, chunk according to CHUNK_MODE.
    Returns list of LangChain Documents ready for embedding.
    """
    notes = _fetch_notes_with_metadata()
    logger.info(f"Fetched {len(notes)} clinical notes from SQLite")

    if CHUNK_MODE == "encounter":
        return chunk_notes_encounter_level(notes)
    else:
        return chunk_notes_note_level(notes)
