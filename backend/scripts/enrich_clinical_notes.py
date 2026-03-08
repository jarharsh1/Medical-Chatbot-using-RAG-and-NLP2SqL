"""
Enrich clinical notes with realistic LLM-generated doctor observations.

Uses Ollama (qwen2.5:14b) to generate authentic clinical narratives including:
- Chief complaint / presenting symptoms
- History of present illness (HPI)
- Physical examination findings
- Vital signs
- Assessment and plan
- Patient-doctor interaction notes

Run: python -m backend.scripts.enrich_clinical_notes
"""

import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from tqdm import tqdm

from backend.config import LLM_MODEL

# Medical context for each condition to help the LLM
CONDITION_CONTEXT: Dict[str, Dict] = {
    "Gout": {
        "typical_symptoms": "severe joint pain (especially big toe), swelling, redness, warmth, tenderness, difficulty walking, morning stiffness",
        "risk_factors": "high purine diet, alcohol, obesity, kidney disease, certain medications, dehydration",
        "common_triggers": "red meat, shellfish, alcohol, dehydration, injury, surgery",
        "exam_focus": "joint inspection, palpation for tophi, range of motion, signs of infection",
    },
    "Hypertension": {
        "typical_symptoms": "often asymptomatic, headaches, dizziness, shortness of breath, chest discomfort, vision changes, fatigue",
        "risk_factors": "obesity, high sodium diet, sedentary lifestyle, stress, family history, age",
        "common_triggers": "salt intake, stress, medication non-compliance, caffeine, sleep apnea",
        "exam_focus": "BP in both arms, fundoscopy, cardiac auscultation, peripheral pulses, edema check",
    },
    "Asthma": {
        "typical_symptoms": "wheezing, shortness of breath, chest tightness, coughing (especially at night), difficulty exercising",
        "risk_factors": "allergies, family history, respiratory infections, environmental irritants, obesity",
        "common_triggers": "allergens, cold air, exercise, smoke, dust, pet dander, respiratory infections",
        "exam_focus": "lung auscultation, respiratory rate, accessory muscle use, peak flow, oxygen saturation",
    },
    "Diabetes": {
        "typical_symptoms": "polyuria, polydipsia, fatigue, blurred vision, slow wound healing, tingling/numbness in feet, weight changes",
        "risk_factors": "obesity, sedentary lifestyle, family history, age, gestational diabetes history",
        "common_triggers": "dietary indiscretion, missed medications, illness, stress, decreased activity",
        "exam_focus": "foot exam (monofilament, pulses), skin inspection, BMI, blood pressure, fundoscopy",
    },
    "Arthritis": {
        "typical_symptoms": "joint pain, stiffness (worse in morning), swelling, decreased range of motion, fatigue, joint warmth",
        "risk_factors": "age, obesity, joint injuries, family history, gender (RA more common in women)",
        "common_triggers": "overuse, weather changes, stress, infection, weight gain",
        "exam_focus": "joint inspection, palpation for synovitis, ROM testing, grip strength, gait assessment",
    },
    "COPD": {
        "typical_symptoms": "chronic cough, sputum production, dyspnea on exertion, wheezing, frequent respiratory infections, fatigue",
        "risk_factors": "smoking history, occupational exposures, alpha-1 antitrypsin deficiency, air pollution",
        "common_triggers": "respiratory infections, air pollution, cold weather, smoke exposure",
        "exam_focus": "lung sounds, breathing pattern, accessory muscle use, cyanosis, clubbing, O2 saturation",
    },
    "Anxiety": {
        "typical_symptoms": "excessive worry, restlessness, fatigue, difficulty concentrating, irritability, sleep disturbance, muscle tension",
        "risk_factors": "family history, trauma, chronic illness, substance use, stressful life events",
        "common_triggers": "stress, caffeine, lack of sleep, major life changes, health concerns",
        "exam_focus": "mental status exam, vital signs (tachycardia), tremor, screening questionnaires (GAD-7)",
    },
    "Depression": {
        "typical_symptoms": "persistent sadness, loss of interest, sleep changes, appetite changes, fatigue, difficulty concentrating, feelings of worthlessness",
        "risk_factors": "family history, trauma, chronic illness, substance use, major life stressors",
        "common_triggers": "loss, relationship problems, financial stress, chronic pain, seasonal changes",
        "exam_focus": "mental status exam, PHQ-9 screening, assess for suicidal ideation, thyroid function",
    },
}

# System prompt for clinical note generation
SYSTEM_PROMPT = """You are an experienced physician writing clinical documentation. Generate realistic, professional clinical notes that would be found in an electronic health record (EHR).

Your notes should:
1. Sound authentic - like a real doctor wrote them
2. Include specific clinical details (vitals, exam findings, symptoms)
3. Use appropriate medical terminology and abbreviations
4. Vary in style and detail level (some visits are routine, some are complex)
5. Include patient quotes where appropriate
6. Document the patient-doctor interaction naturally

Format the note with these sections:
- CC (Chief Complaint): Brief quote from patient
- HPI (History of Present Illness): Detailed narrative
- VITALS: Realistic vital signs
- PHYSICAL EXAM: Relevant findings
- ASSESSMENT: Clinical impression
- PLAN: Treatment plan with specifics

Keep it concise but realistic (200-400 words). Do NOT use placeholder brackets - generate specific realistic values."""


def generate_note_with_llm(
    llm: ChatOllama,
    patient_name: str,
    condition: str,
    current_medications: str,
    visit_date: str,
) -> Optional[str]:
    """Generate a clinical note using the LLM."""

    # Get condition-specific context
    context = CONDITION_CONTEXT.get(condition, {})
    typical_symptoms = context.get("typical_symptoms", "symptoms related to the condition")
    risk_factors = context.get("risk_factors", "various risk factors")
    triggers = context.get("common_triggers", "various triggers")
    exam_focus = context.get("exam_focus", "relevant physical examination")

    # Randomize visit type for variety
    visit_types = [
        "routine follow-up visit",
        "visit for worsening symptoms",
        "new symptom evaluation",
        "medication refill visit",
        "post-hospitalization follow-up",
        "urgent same-day visit",
    ]
    visit_type = random.choice(visit_types)

    # Randomize patient demographics for variety
    age = random.randint(28, 75)
    gender = random.choice(["male", "female"])

    user_prompt = f"""Generate a clinical note for this patient encounter:

PATIENT: {patient_name}
AGE/GENDER: {age} year old {gender}
CONDITION: {condition}
CURRENT MEDICATIONS: {current_medications}
VISIT DATE: {visit_date}
VISIT TYPE: {visit_type}

CONDITION CONTEXT:
- Typical symptoms: {typical_symptoms}
- Risk factors: {risk_factors}
- Common triggers: {triggers}
- Exam focus areas: {exam_focus}

Generate a realistic clinical note for this {visit_type}. Include specific vital signs, exam findings, and a detailed plan. Make it sound like a real physician wrote it."""

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        return (response.content or "").strip()
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return None


def enrich_notes_with_llm(db_path: str, batch_size: int = 50, max_workers: int = 3):
    """Add LLM-generated clinical narratives to all notes."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("PRAGMA table_info(clinical_notes)")
    columns = [col[1] for col in cursor.fetchall()]

    if "doctor_notes" not in columns:
        print("Adding doctor_notes column...")
        cursor.execute("ALTER TABLE clinical_notes ADD COLUMN doctor_notes TEXT")
        conn.commit()

    # Get all notes that need enrichment
    cursor.execute("""
        SELECT note_id, patient_id, condition_name, note_text, visit_date
        FROM clinical_notes
        WHERE doctor_notes IS NULL OR doctor_notes = ''
    """)
    notes = cursor.fetchall()

    if not notes:
        print("All notes already enriched!")
        conn.close()
        return 0

    print(f"Enriching {len(notes)} clinical notes with LLM...")

    # Get patient names
    cursor.execute("SELECT patient_id, full_name FROM patients")
    patient_names = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    # Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)  # Higher temp for variety

    # Process in batches
    updated = 0
    failed = 0
    results = []

    print(f"Processing with {max_workers} workers...")

    for batch_start in tqdm(range(0, len(notes), batch_size), desc="Batches"):
        batch = notes[batch_start:batch_start + batch_size]
        batch_results = []

        for note_id, patient_id, condition, note_text, visit_date in batch:
            patient_name = patient_names.get(patient_id, "Unknown Patient")

            # Extract medications from existing note
            meds = ""
            if note_text:
                if "is on " in note_text:
                    meds = note_text.split("is on ")[-1].rstrip(".")
                elif "with " in note_text:
                    parts = note_text.split("with ")
                    if len(parts) > 1:
                        meds = parts[-1].rstrip(".")

            # Generate note with LLM
            narrative = generate_note_with_llm(
                llm, patient_name, condition, meds or "current regimen", visit_date or "2024-01-15"
            )

            if narrative:
                batch_results.append((narrative, note_id))
                updated += 1
            else:
                failed += 1

            # Small delay to avoid overwhelming Ollama
            time.sleep(0.1)

        # Write batch to database
        if batch_results:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.executemany(
                "UPDATE clinical_notes SET doctor_notes = ? WHERE note_id = ?",
                batch_results
            )
            conn.commit()
            conn.close()

        print(f"  Batch complete: {updated} updated, {failed} failed")

    print(f"\nEnrichment complete! Updated: {updated}, Failed: {failed}")
    return updated


def enrich_sample(db_path: str, sample_size: int = 100):
    """Enrich only a sample of notes (for testing)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("PRAGMA table_info(clinical_notes)")
    columns = [col[1] for col in cursor.fetchall()]

    if "doctor_notes" not in columns:
        print("Adding doctor_notes column...")
        cursor.execute("ALTER TABLE clinical_notes ADD COLUMN doctor_notes TEXT")
        conn.commit()

    # Get sample of notes (prioritize variety of conditions)
    cursor.execute("""
        SELECT note_id, patient_id, condition_name, note_text, visit_date
        FROM clinical_notes
        WHERE doctor_notes IS NULL OR doctor_notes = ''
        ORDER BY RANDOM()
        LIMIT ?
    """, (sample_size,))
    notes = cursor.fetchall()

    print(f"Enriching {len(notes)} sample clinical notes with LLM...")

    # Get patient names
    cursor.execute("SELECT patient_id, full_name FROM patients")
    patient_names = {row[0]: row[1] for row in cursor.fetchall()}

    # Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)

    updated = 0
    for note_id, patient_id, condition, note_text, visit_date in tqdm(notes, desc="Generating notes"):
        patient_name = patient_names.get(patient_id, "Unknown Patient")

        # Extract medications
        meds = ""
        if note_text:
            if "is on " in note_text:
                meds = note_text.split("is on ")[-1].rstrip(".")
            elif "with " in note_text:
                parts = note_text.split("with ")
                if len(parts) > 1:
                    meds = parts[-1].rstrip(".")

        narrative = generate_note_with_llm(
            llm, patient_name, condition, meds or "current regimen", visit_date or "2024-01-15"
        )

        if narrative:
            cursor.execute(
                "UPDATE clinical_notes SET doctor_notes = ? WHERE note_id = ?",
                (narrative, note_id)
            )
            updated += 1

            if updated % 10 == 0:
                conn.commit()

    conn.commit()
    conn.close()

    print(f"Sample enrichment complete! Updated {updated} notes.")
    return updated


def enrich_percentage(db_path: str, percentage: int = 40):
    """
    Enrich only a percentage of notes randomly.

    This is more realistic - not every visit has detailed doctor notes.
    Some visits are quick refills or routine checks with minimal documentation.

    Args:
        db_path: Path to the database
        percentage: Percentage of notes to enrich (default 40%)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("PRAGMA table_info(clinical_notes)")
    columns = [col[1] for col in cursor.fetchall()]

    if "doctor_notes" not in columns:
        print("Adding doctor_notes column...")
        cursor.execute("ALTER TABLE clinical_notes ADD COLUMN doctor_notes TEXT")
        conn.commit()

    # Get total count
    cursor.execute("SELECT COUNT(*) FROM clinical_notes")
    total_notes = cursor.fetchone()[0]

    # Calculate how many to enrich
    target_count = int(total_notes * percentage / 100)

    # Check how many are already enriched
    cursor.execute("SELECT COUNT(*) FROM clinical_notes WHERE doctor_notes IS NOT NULL AND doctor_notes != ''")
    already_enriched = cursor.fetchone()[0]

    to_enrich = max(0, target_count - already_enriched)

    if to_enrich == 0:
        print(f"Already at {percentage}% enrichment ({already_enriched}/{total_notes} notes)")
        conn.close()
        return 0

    print(f"Target: {percentage}% of {total_notes} = {target_count} notes")
    print(f"Already enriched: {already_enriched}")
    print(f"Will enrich: {to_enrich} more notes")
    print()

    # Get random notes to enrich, ensuring variety of conditions
    cursor.execute("""
        SELECT note_id, patient_id, condition_name, note_text, visit_date
        FROM clinical_notes
        WHERE doctor_notes IS NULL OR doctor_notes = ''
        ORDER BY RANDOM()
        LIMIT ?
    """, (to_enrich,))
    notes = cursor.fetchall()

    # Get patient names
    cursor.execute("SELECT patient_id, full_name FROM patients")
    patient_names = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    # Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)

    updated = 0
    failed = 0

    # Process and save in batches
    batch_results = []
    batch_size = 20

    for note_id, patient_id, condition, note_text, visit_date in tqdm(notes, desc="Generating notes"):
        patient_name = patient_names.get(patient_id, "Unknown Patient")

        # Extract medications
        meds = ""
        if note_text:
            if "is on " in note_text:
                meds = note_text.split("is on ")[-1].rstrip(".")
            elif "with " in note_text:
                parts = note_text.split("with ")
                if len(parts) > 1:
                    meds = parts[-1].rstrip(".")

        narrative = generate_note_with_llm(
            llm, patient_name, condition, meds or "current regimen", visit_date or "2024-01-15"
        )

        if narrative:
            batch_results.append((narrative, note_id))
            updated += 1
        else:
            failed += 1

        # Save batch to database
        if len(batch_results) >= batch_size:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.executemany(
                "UPDATE clinical_notes SET doctor_notes = ? WHERE note_id = ?",
                batch_results
            )
            conn.commit()
            conn.close()
            batch_results = []

    # Save remaining
    if batch_results:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE clinical_notes SET doctor_notes = ? WHERE note_id = ?",
            batch_results
        )
        conn.commit()
        conn.close()

    print(f"\nEnrichment complete!")
    print(f"  Updated: {updated}")
    print(f"  Failed: {failed}")
    print(f"  Total enriched: {already_enriched + updated}/{total_notes} ({100*(already_enriched + updated)/total_notes:.1f}%)")

    return updated


def verify_enrichment(db_path: str):
    """Print sample enriched notes for verification."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "="*70)
    print("SAMPLE LLM-GENERATED CLINICAL NOTES:")
    print("="*70)

    cursor.execute("""
        SELECT DISTINCT condition_name FROM clinical_notes
        WHERE doctor_notes IS NOT NULL AND doctor_notes != ''
        LIMIT 5
    """)
    conditions = [row[0] for row in cursor.fetchall()]

    for cond in conditions:
        cursor.execute("""
            SELECT p.full_name, cn.condition_name, cn.doctor_notes
            FROM clinical_notes cn
            JOIN patients p ON cn.patient_id = p.patient_id
            WHERE cn.condition_name = ? AND cn.doctor_notes IS NOT NULL
            LIMIT 1
        """, (cond,))
        row = cursor.fetchone()
        if row:
            print(f"\n{'='*70}")
            print(f"PATIENT: {row[0]} | CONDITION: {row[1]}")
            print("-"*70)
            print(row[2])
            print()

    # Stats
    cursor.execute("SELECT COUNT(*) FROM clinical_notes WHERE doctor_notes IS NOT NULL AND doctor_notes != ''")
    enriched = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM clinical_notes")
    total = cursor.fetchone()[0]
    print(f"\nENRICHMENT STATUS: {enriched}/{total} notes ({100*enriched/total:.1f}%)")

    conn.close()


if __name__ == "__main__":
    import argparse
    from backend.config import DB_PATH

    parser = argparse.ArgumentParser(description="Enrich clinical notes with LLM-generated content")
    parser.add_argument("--sample", type=int, default=0, help="Only enrich N sample notes (for testing)")
    parser.add_argument("--percentage", type=int, default=0, help="Enrich N%% of notes randomly (realistic - not all visits have detailed notes)")
    parser.add_argument("--verify", action="store_true", help="Only verify existing enrichment")
    parser.add_argument("--all", action="store_true", help="Enrich all notes (takes a long time)")
    args = parser.parse_args()

    print("Clinical Notes LLM Enrichment Script")
    print("="*70)
    print(f"Database: {DB_PATH}")
    print(f"Model: {LLM_MODEL} (via Ollama)")
    print("="*70)

    if args.verify:
        verify_enrichment(DB_PATH)
    elif args.sample > 0:
        enrich_sample(DB_PATH, sample_size=args.sample)
        verify_enrichment(DB_PATH)
    elif args.percentage > 0:
        print(f"\nEnriching {args.percentage}% of notes (realistic distribution)")
        enrich_percentage(DB_PATH, percentage=args.percentage)
        verify_enrichment(DB_PATH)
    elif args.all:
        print("\nWARNING: This will enrich ALL notes and may take several hours!")
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() == "yes":
            enrich_notes_with_llm(DB_PATH)
            verify_enrichment(DB_PATH)
    else:
        print("\nUsage:")
        print("  --sample N      Enrich N random notes (for quick testing)")
        print("  --percentage N  Enrich N% of notes randomly (RECOMMENDED - realistic)")
        print("  --all           Enrich all notes (takes hours)")
        print("  --verify        Check enrichment status")
        print("\nRecommended: python -m backend.scripts.enrich_clinical_notes --percentage 40")
        print("This enriches ~40% of notes, which is realistic - not every visit has detailed documentation.")
