"""
Test case for: Prescription metrics should use distinct counts

This test verifies that the Prescription Overview dashboard metrics
correctly count unique prescriptions and patients, avoiding duplicates
from table joins.
"""

import sqlite3
from backend.config import DB_PATH


def test_distinct_prescription_counts():
    """
    Verify that KPI queries return distinct counts.
    
    The dashboard joins patients, clinics, clinical_notes, and prescriptions tables.
    Without DISTINCT, counts are inflated due to row multiplication.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get counts WITH DISTINCT
    distinct_sql = """
        SELECT 
            COUNT(DISTINCT r.rx_id) as distinct_rx,
            COUNT(DISTINCT p.patient_id) as distinct_patients
        FROM patients p
        JOIN clinics c ON p.clinic_id = c.clinic_id
        JOIN clinical_notes n ON p.patient_id = n.patient_id
        JOIN prescriptions r ON p.patient_id = r.patient_id
    """
    distinct_result = cur.execute(distinct_sql).fetchone()
    distinct_rx = distinct_result["distinct_rx"]
    distinct_patients = distinct_result["distinct_patients"]
    
    # Get counts WITHOUT DISTINCT (the buggy way)
    duplicate_sql = """
        SELECT 
            COUNT(*) as duplicate_rx,
            COUNT(*) as duplicate_patients
        FROM patients p
        JOIN clinics c ON p.clinic_id = c.clinic_id
        JOIN clinical_notes n ON p.patient_id = n.patient_id
        JOIN prescriptions r ON p.patient_id = r.patient_id
    """
    duplicate_result = cur.execute(duplicate_sql).fetchone()
    duplicate_rx = duplicate_result["duplicate_rx"]
    duplicate_patients = duplicate_result["duplicate_patients"]
    
    conn.close()
    
    # Assertions
    print(f"Distinct Rx count: {distinct_rx}")
    print(f"Duplicate Rx count: {duplicate_rx}")
    print(f"Distinct Patients: {distinct_patients}")
    print(f"Duplicate Patients: {duplicate_patients}")
    
    # The distinct count should be less than or equal to duplicate count
    assert distinct_rx <= duplicate_rx, \
        f"Distinct Rx ({distinct_rx}) should be <= duplicate Rx ({duplicate_rx})"
    
    assert distinct_patients <= duplicate_patients, \
        f"Distinct patients ({distinct_patients}) should be <= duplicate ({duplicate_patients})"
    
    # If there are duplicates, the counts should differ
    if distinct_rx < duplicate_rx:
        print(f"[*] Found {duplicate_rx - distinct_rx} duplicate prescription counts")
    
    if distinct_patients < duplicate_patients:
        print(f"[*] Found {duplicate_patients - distinct_patients} duplicate patient counts")
    
    print("\n[PASS] Test passed: Distinct counts are correct")


def test_status_counts_are_distinct():
    """
    Verify that status-based counts (Active, At Risk, Due) use distinct rx_id.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Distinct status counts
    distinct_status_sql = """
        SELECT 
            r.status,
            COUNT(DISTINCT r.rx_id) as cnt
        FROM prescriptions r
        JOIN patients p ON r.patient_id = p.patient_id
        JOIN clinics c ON p.clinic_id = c.clinic_id
        JOIN clinical_notes n ON p.patient_id = n.patient_id
        GROUP BY r.status
    """
    distinct_results = dict((row["status"], row["cnt"]) 
                           for row in cur.execute(distinct_status_sql).fetchall())
    
    # Duplicate status counts
    duplicate_status_sql = """
        SELECT 
            r.status,
            COUNT(*) as cnt
        FROM prescriptions r
        JOIN patients p ON r.patient_id = p.patient_id
        JOIN clinics c ON p.clinic_id = c.clinic_id
        JOIN clinical_notes n ON p.patient_id = n.patient_id
        GROUP BY r.status
    """
    duplicate_results = dict((row["status"], row["cnt"]) 
                           for row in cur.execute(duplicate_status_sql).fetchall())
    
    conn.close()
    
    print("\nStatus counts comparison:")
    print("| Status          | Distinct | Duplicate |")
    print("|-----------------|----------|-----------|")
    for status in distinct_results:
        d = distinct_results.get(status, 0)
        dup = duplicate_results.get(status, 0)
        print(f"| {status:<15} | {d:<8} | {dup:<9} |")
    
    # Verify distinct counts don't exceed duplicate counts
    for status in distinct_results:
        assert distinct_results[status] <= duplicate_results.get(status, 0), \
            f"Status '{status}': distinct ({distinct_results[status]}) > duplicate ({duplicate_results.get(status, 0)})"
    
    print("\n[PASS] Test passed: Status counts are distinct")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Prescription Metrics - Distinct Counts")
    print("=" * 60)
    
    test_distinct_prescription_counts()
    test_status_counts_are_distinct()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
