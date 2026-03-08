"""
Standalone database initializer for CI and local setup.
Creates SQLite tables and loads CSV data without importing the FastAPI app.
"""
import csv
import os
import sqlite3

from backend.config import DATA_DIR, DB_PATH


def _load_csv(conn: sqlite3.Connection, filename: str, table: str):
    if not DATA_DIR:
        return
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return
    cursor = conn.cursor()
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if not headers:
            return
        rows = list(reader)
    placeholders = ",".join(["?"] * len(headers))
    cursor.executemany(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})", rows)
    conn.commit()
    print(f"  Loaded {len(rows):,} rows into {table}")


def init_db():
    print(f"Initializing database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS clinics (
            clinic_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT
        );
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            dob TEXT,
            gender TEXT,
            insurance_provider TEXT,
            clinic_id INTEGER,
            FOREIGN KEY(clinic_id) REFERENCES clinics(clinic_id)
        );
        CREATE TABLE IF NOT EXISTS clinical_notes (
            note_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            visit_date TEXT,
            doctor_name TEXT,
            diagnosis_code TEXT,
            condition_name TEXT,
            note_text TEXT,
            doctor_notes TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        );
        CREATE TABLE IF NOT EXISTS prescriptions (
            rx_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            medication_name TEXT,
            dosage TEXT,
            form TEXT,
            drug_class TEXT,
            days_supply INTEGER,
            refills_remaining INTEGER,
            last_filled_date TEXT,
            status TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        );
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            message_count INTEGER DEFAULT 0
        );
    """)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM clinics")
    if cur.fetchone()[0] == 0:
        print("Loading CSV data...")
        _load_csv(conn, "clinics.csv", "clinics")
        _load_csv(conn, "patients.csv", "patients")
        _load_csv(conn, "clinical_notes.csv", "clinical_notes")
        _load_csv(conn, "prescriptions.csv", "prescriptions")
    else:
        print("Database already populated, skipping CSV load.")

    cur.execute("SELECT COUNT(*) FROM patients")
    print(f"Database ready: {cur.fetchone()[0]:,} patients")
    conn.close()


if __name__ == "__main__":
    init_db()
