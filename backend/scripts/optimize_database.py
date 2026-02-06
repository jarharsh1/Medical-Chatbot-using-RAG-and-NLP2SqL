"""
Database Optimization Script

Implements:
1. Indexes on frequently queried columns
2. Materialized views for common aggregations
3. Query analysis and EXPLAIN support

Run: python -m backend.scripts.optimize_database
"""

import sqlite3
import time
from typing import List, Tuple

from backend.config import DB_PATH


# ============================================
# 1. INDEX DEFINITIONS
# ============================================

INDEXES = [
    # Clinical Notes - most queried table
    ("idx_cn_patient_id", "clinical_notes", "patient_id"),
    ("idx_cn_condition", "clinical_notes", "condition_name"),
    ("idx_cn_doctor", "clinical_notes", "doctor_name"),
    ("idx_cn_visit_date", "clinical_notes", "visit_date"),
    ("idx_cn_diagnosis", "clinical_notes", "diagnosis_code"),

    # Patients
    ("idx_patients_clinic", "patients", "clinic_id"),
    ("idx_patients_gender", "patients", "gender"),
    ("idx_patients_insurance", "patients", "insurance_provider"),

    # Prescriptions
    ("idx_rx_patient", "prescriptions", "patient_id"),
    ("idx_rx_medication", "prescriptions", "medication_name"),
    ("idx_rx_status", "prescriptions", "status"),
    ("idx_rx_last_filled", "prescriptions", "last_filled_date"),

    # Clinics (usually small, but for completeness)
    ("idx_clinics_name", "clinics", "name"),
]

# Composite indexes for common JOIN patterns
COMPOSITE_INDEXES = [
    ("idx_cn_patient_condition", "clinical_notes", "patient_id, condition_name"),
    ("idx_cn_patient_doctor", "clinical_notes", "patient_id, doctor_name"),
    ("idx_rx_patient_status", "prescriptions", "patient_id, status"),
]


# ============================================
# 2. MATERIALIZED VIEW DEFINITIONS
# ============================================

MATERIALIZED_VIEWS = {
    "mv_condition_stats": """
        SELECT
            condition_name,
            COUNT(DISTINCT patient_id) as patient_count,
            COUNT(*) as note_count,
            MIN(visit_date) as first_seen,
            MAX(visit_date) as last_seen
        FROM clinical_notes
        WHERE condition_name IS NOT NULL AND condition_name != ''
        GROUP BY condition_name
        ORDER BY patient_count DESC
    """,

    "mv_clinic_stats": """
        SELECT
            c.clinic_id,
            c.name as clinic_name,
            c.location,
            COUNT(DISTINCT p.patient_id) as total_patients,
            COUNT(DISTINCT cn.note_id) as total_notes,
            COUNT(DISTINCT cn.doctor_name) as doctor_count,
            COUNT(DISTINCT cn.condition_name) as conditions_treated
        FROM clinics c
        LEFT JOIN patients p ON c.clinic_id = p.clinic_id
        LEFT JOIN clinical_notes cn ON p.patient_id = cn.patient_id
        GROUP BY c.clinic_id, c.name, c.location
    """,

    "mv_doctor_stats": """
        SELECT
            doctor_name,
            COUNT(DISTINCT patient_id) as patients_seen,
            COUNT(*) as total_visits,
            COUNT(DISTINCT condition_name) as conditions_treated,
            MIN(visit_date) as first_visit,
            MAX(visit_date) as last_visit
        FROM clinical_notes
        WHERE doctor_name IS NOT NULL AND doctor_name != ''
        GROUP BY doctor_name
        ORDER BY patients_seen DESC
    """,

    "mv_medication_stats": """
        SELECT
            medication_name,
            COUNT(*) as prescription_count,
            COUNT(DISTINCT patient_id) as patient_count,
            SUM(CASE WHEN status = 'Active' THEN 1 ELSE 0 END) as active_count,
            AVG(refills_remaining) as avg_refills_remaining
        FROM prescriptions
        WHERE medication_name IS NOT NULL AND medication_name != ''
        GROUP BY medication_name
        ORDER BY prescription_count DESC
    """,

    "mv_patient_summary": """
        SELECT
            p.patient_id,
            p.full_name,
            p.gender,
            c.name as clinic_name,
            COUNT(DISTINCT cn.note_id) as visit_count,
            COUNT(DISTINCT cn.condition_name) as condition_count,
            COUNT(DISTINCT rx.rx_id) as prescription_count,
            MAX(cn.visit_date) as last_visit
        FROM patients p
        LEFT JOIN clinics c ON p.clinic_id = c.clinic_id
        LEFT JOIN clinical_notes cn ON p.patient_id = cn.patient_id
        LEFT JOIN prescriptions rx ON p.patient_id = rx.patient_id
        GROUP BY p.patient_id, p.full_name, p.gender, c.name
    """,

    "mv_daily_stats": """
        SELECT
            DATE(visit_date) as visit_day,
            COUNT(DISTINCT patient_id) as unique_patients,
            COUNT(*) as total_visits,
            COUNT(DISTINCT doctor_name) as doctors_active
        FROM clinical_notes
        WHERE visit_date IS NOT NULL
        GROUP BY DATE(visit_date)
        ORDER BY visit_day DESC
    """,
}


def create_indexes(conn: sqlite3.Connection) -> List[Tuple[str, float]]:
    """Create all indexes and return timing info."""
    cursor = conn.cursor()
    results = []

    print("\n" + "=" * 60)
    print("CREATING INDEXES")
    print("=" * 60)

    # Single-column indexes
    for idx_name, table, column in INDEXES:
        start = time.time()
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
            elapsed = time.time() - start
            results.append((idx_name, elapsed))
            print(f"  [OK] {idx_name} on {table}({column}) - {elapsed:.3f}s")
        except Exception as e:
            print(f"  [FAIL] {idx_name}: {e}")

    # Composite indexes
    for idx_name, table, columns in COMPOSITE_INDEXES:
        start = time.time()
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({columns})")
            elapsed = time.time() - start
            results.append((idx_name, elapsed))
            print(f"  [OK] {idx_name} on {table}({columns}) - {elapsed:.3f}s")
        except Exception as e:
            print(f"  [FAIL] {idx_name}: {e}")

    conn.commit()
    return results


def create_materialized_views(conn: sqlite3.Connection) -> List[Tuple[str, float, int]]:
    """Create materialized views and return timing + row counts."""
    cursor = conn.cursor()
    results = []

    print("\n" + "=" * 60)
    print("CREATING MATERIALIZED VIEWS")
    print("=" * 60)

    for view_name, query in MATERIALIZED_VIEWS.items():
        start = time.time()
        try:
            # Drop existing view/table
            cursor.execute(f"DROP TABLE IF EXISTS {view_name}")

            # Create as table (materialized)
            cursor.execute(f"CREATE TABLE {view_name} AS {query}")

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {view_name}")
            row_count = cursor.fetchone()[0]

            elapsed = time.time() - start
            results.append((view_name, elapsed, row_count))
            print(f"  [OK] {view_name} - {row_count:,} rows in {elapsed:.3f}s")

        except Exception as e:
            print(f"  [FAIL] {view_name}: {e}")

    conn.commit()
    return results


def create_mv_indexes(conn: sqlite3.Connection):
    """Create indexes on materialized views for fast lookups."""
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("INDEXING MATERIALIZED VIEWS")
    print("=" * 60)

    mv_indexes = [
        ("idx_mv_condition_name", "mv_condition_stats", "condition_name"),
        ("idx_mv_condition_count", "mv_condition_stats", "patient_count DESC"),
        ("idx_mv_clinic_id", "mv_clinic_stats", "clinic_id"),
        ("idx_mv_clinic_patients", "mv_clinic_stats", "total_patients DESC"),
        ("idx_mv_doctor_name", "mv_doctor_stats", "doctor_name"),
        ("idx_mv_doctor_patients", "mv_doctor_stats", "patients_seen DESC"),
        ("idx_mv_med_name", "mv_medication_stats", "medication_name"),
        ("idx_mv_med_count", "mv_medication_stats", "prescription_count DESC"),
        ("idx_mv_patient_id", "mv_patient_summary", "patient_id"),
    ]

    for idx_name, table, column in mv_indexes:
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
            print(f"  [OK] {idx_name}")
        except Exception as e:
            print(f"  [SKIP] {idx_name}: {e}")

    conn.commit()


def analyze_tables(conn: sqlite3.Connection):
    """Run ANALYZE to update query planner statistics."""
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("ANALYZING TABLES")
    print("=" * 60)

    cursor.execute("ANALYZE")
    print("  [OK] Statistics updated for query planner")
    conn.commit()


def verify_optimizations(conn: sqlite3.Connection):
    """Run sample queries to verify optimization impact."""
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("VERIFICATION - Sample Query Performance")
    print("=" * 60)

    test_queries = [
        ("Count diabetes patients (indexed)",
         "SELECT COUNT(DISTINCT patient_id) FROM clinical_notes WHERE condition_name LIKE '%Diabetes%'"),

        ("Count diabetes patients (from MV)",
         "SELECT patient_count FROM mv_condition_stats WHERE condition_name LIKE '%Diabetes%'"),

        ("Top clinic by patients (indexed)",
         "SELECT c.name, COUNT(DISTINCT p.patient_id) as cnt FROM clinics c JOIN patients p ON c.clinic_id = p.clinic_id GROUP BY c.name ORDER BY cnt DESC LIMIT 1"),

        ("Top clinic by patients (from MV)",
         "SELECT clinic_name, total_patients FROM mv_clinic_stats ORDER BY total_patients DESC LIMIT 1"),

        ("Doctor with most patients (indexed)",
         "SELECT doctor_name, COUNT(DISTINCT patient_id) as cnt FROM clinical_notes GROUP BY doctor_name ORDER BY cnt DESC LIMIT 1"),

        ("Doctor with most patients (from MV)",
         "SELECT doctor_name, patients_seen FROM mv_doctor_stats ORDER BY patients_seen DESC LIMIT 1"),
    ]

    for description, query in test_queries:
        start = time.time()
        try:
            cursor.execute(query)
            result = cursor.fetchone()
            elapsed = (time.time() - start) * 1000  # ms
            print(f"  {description}")
            print(f"    Result: {result}")
            print(f"    Time: {elapsed:.2f}ms")
        except Exception as e:
            print(f"  {description}: ERROR - {e}")

    print()


def get_db_stats(conn: sqlite3.Connection) -> dict:
    """Get database statistics."""
    cursor = conn.cursor()

    stats = {}

    # Table row counts
    tables = ["patients", "clinics", "clinical_notes", "prescriptions"]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cursor.fetchone()[0]

    # Index count
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
    stats["total_indexes"] = cursor.fetchone()[0]

    # MV count
    mv_count = 0
    for mv_name in MATERIALIZED_VIEWS.keys():
        cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{mv_name}'")
        if cursor.fetchone()[0] > 0:
            mv_count += 1
    stats["materialized_views"] = mv_count

    return stats


def refresh_materialized_views(conn: sqlite3.Connection = None):
    """Refresh all materialized views. Call this after data changes."""
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        should_close = True
    else:
        should_close = False

    print("Refreshing materialized views...")
    create_materialized_views(conn)

    if should_close:
        conn.close()


def run_full_optimization():
    """Run complete database optimization."""
    print("\n" + "=" * 60)
    print("DATABASE OPTIMIZATION - FULL RUN")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    # Get initial stats
    print("\nInitial database stats:")
    stats = get_db_stats(conn)
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    # Run optimizations
    create_indexes(conn)
    create_materialized_views(conn)
    create_mv_indexes(conn)
    analyze_tables(conn)

    # Verify
    verify_optimizations(conn)

    # Final stats
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    final_stats = get_db_stats(conn)
    print(f"  Total indexes: {final_stats['total_indexes']}")
    print(f"  Materialized views: {final_stats['materialized_views']}")
    print("\nNext steps:")
    print("  - Queries using indexed columns will be 10-100x faster")
    print("  - Use mv_* tables for instant aggregations")
    print("  - Run refresh_materialized_views() after bulk data changes")

    conn.close()


if __name__ == "__main__":
    run_full_optimization()
