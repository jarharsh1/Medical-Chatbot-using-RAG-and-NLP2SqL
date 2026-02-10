import os
import csv
import random
import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict
from faker import Faker
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

NUM_PATIENTS = 35000
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OS_SEED = 42

fake = Faker()
Faker.seed(OS_SEED)
random.seed(OS_SEED)

today = datetime.now()

# --- CACHING FOR LLM-GENERATED CONTENT ---

def load_cached_conditions() -> List[Dict]:
    """Load cached medical conditions or generate new ones using LLM."""
    cache_file = os.path.join(DATA_DIR, "medical_conditions_cache.json")
    
    if os.path.exists(cache_file):
        logger.info("Loading cached medical conditions...")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    logger.info("Generating medical conditions using LLM...")
    conditions = generate_medical_conditions_llm()
    
    # Cache for future use
    with open(cache_file, 'w') as f:
        json.dump(conditions, f, indent=2)
    logger.info(f"Cached {len(conditions)} conditions to {cache_file}")
    
    return conditions


def generate_medical_conditions_llm() -> List[Dict]:
    """Generate realistic medical conditions with medications and templates using LLM."""
    
    prompt = """
Generate 15 diverse medical conditions as a JSON object.
Each condition needs realistic data for patient records.

Return JSON:
{
    "conditions": [
        {
            "condition_name": "Type 2 Diabetes Mellitus",
            "icd10_code": "E11.9",
            "medications": ["Metformin", "Glipizide", "Empagliflozin", "Insulin Glargine", "Glimepiride"],
            "dosage_forms": ["tablet", "tablet", "tablet", "injection", "tablet"],
            "dosage_examples": ["500mg", "5mg", "10mg", "20 units", "2mg"],
            "symptoms": ["increased thirst", "fatigue", "frequent urination", "blurred vision", "slow healing"],
            "chronic": true,
            "note_templates": [
                "Follow-up for diabetes management. HbA1c shows {symptom}. Patient reports {symptom}. Continue current regimen.",
                "Routine diabetic check. {symptom} noted. Discussed diet and exercise. Scheduled lab work.",
                "Diabetes review. {symptom} and {symptom}. Medication adjustment considered. Monitor closely."
            ]
        }
    ]
}

Include: Diabetes, Hypertension, Asthma, COPD, Depression, Anxiety, Hyperlipidemia, Hypothyroidism, 
GERD, Arthritis, Acne, Allergic Rhinitis, Migraine, Anemia, Osteoporosis.

Make it realistic and diverse.
"""
    
    try:
        llm = ChatOllama(model="qwen2.5:14b", temperature=0.8, format="json")
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        data = json.loads(content)
        conditions = data.get("conditions", [])
        
        if not conditions:
            raise ValueError("No conditions generated")
        
        logger.info(f"LLM generated {len(conditions)} conditions")
        return conditions
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise


def generate_note_from_template(patient_name: str, condition: Dict, doctor: str, visit_date: str) -> str:
    """Generate clinical note using templates."""
    templates = condition.get("note_templates", ["Patient presents for routine visit."])
    symptoms = condition.get("symptoms", ["general complaints"])
    
    template = random.choice(templates)
    symptom1 = random.choice(symptoms)
    symptom2 = random.choice([s for s in symptoms if s != symptom1] or symptoms)
    
    note = template.replace("{symptom}", symptom1).replace("{symptom2}", symptom2)
    
    # Add structured sections
    return f"""CHIEF COMPLAINT: {patient_name} presents with {symptom1}.

HPI: This is a {random.choice(['new', 'established'])} patient with {condition['condition_name']} (ICD-10: {condition['icd10_code']}). 
Patient reports {symptom1} and {symptom2} for the past {random.randint(1, 8)} weeks.

MEDICAL HISTORY: {condition['condition_name']}

CURRENT MEDICATIONS: {', '.join(condition['medications'][:2])}

ALLERGIES: No known drug allergies

ASSESSMENT: {condition['condition_name']} - currently {random.choice(['stable', 'improving', 'requiring adjustment'])} on current regimen.

PLAN:
1. Continue current medications
2. Follow-up in {random.choice([4, 6, 8, 12])} weeks
3. Labs ordered: {random.choice(['HbA1c', 'CMP', 'CBC', 'Lipid panel', 'TSH'])}
4. Patient education provided

Visit Date: {visit_date}
Provider: {doctor}"""


def generate_dosage(med_name: str, form: str, example: str) -> str:
    """Generate realistic dosage."""
    dosage_formats = {
        "tablet": ["25mg", "50mg", "100mg", "200mg", "500mg", "10mg once daily", "20mg twice daily"],
        "capsule": ["25mg", "50mg", "100mg", "75mg/25mg", "10/325mg", "25mcg", "50mcg"],
        "liquid": ["5ml", "10ml", "15ml", "5ml twice daily", "10ml once daily"],
        "inhaler": ["90mcg/actuation", "50mcg/actuation", "2 puffs", "1-2 puffs PRN"],
        "cream": ["1% cream", "0.5% ointment", "thin layer", "apply twice daily"],
        "patch": ["25mcg/hr", "50mcg/hr", "100mcg/hr", "apply once weekly"],
        "injection": ["10 units", "20 units", "0.5ml", "1ml IM", "subcutaneous"],
        "drops": ["1-2 drops", "2 drops", "one drop per nostril"],
        "syrup": ["5ml", "10ml", "15ml", "1 tbsp"],
    }
    
    if form in dosage_formats:
        return random.choice(dosage_formats[form])
    return example if example else f"{random.randint(1, 10) * 10}mg"


def generate_bulk_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory: {DATA_DIR}")
    
    # Load/generate medical conditions (cached)
    conditions_db = load_cached_conditions()
    
    # 1. Generate Clinics
    logger.info("Generating Clinics...")
    clinics = []
    clinic_prefixes = ["Downtown", "Westside", "North Hills", "Valley", "City", "Lakeside", "Riverside", 
                       "Central", "Metro", "Community", "Family First", "Prime Care", "Sunrise", "Oakwood"]
    clinic_types = ["Medical Center", "Health Clinic", "Family Practice", "Wellness Center", "Urgent Care", 
                    "Primary Care", "Internal Medicine"]
    locations = ["New York", "Chicago", "San Francisco", "Austin", "Seattle", "Boston", "Miami", 
                 "Denver", "Phoenix", "Los Angeles", "Atlanta", "Portland", "Dallas"]
    
    for i in range(1, 51):
        clinics.append({
            "clinic_id": i,
            "name": f"{random.choice(clinic_prefixes)} {random.choice(clinic_types)}",
            "location": random.choice(locations)
        })
    
    # 2. Generate Patients
    logger.info(f"Generating {NUM_PATIENTS} patients and associated records...")
    
    patients = []
    prescriptions = []
    clinical_notes = []
    
    doctors = [f"Dr. {fake.first_name()} {fake.last_name()}" for _ in range(100)]
    insurances = ["BlueCross BlueShield", "Aetna", "Medicare", "UnitedHealthcare", "Cigna", "Kaiser Permanente", "Humana"]
    
    for i in tqdm(range(1, NUM_PATIENTS + 1), desc="Generating Data"):
        # Patient
        patient = {
            "patient_id": i,
            "full_name": fake.name(),
            "dob": fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y-%m-%d"),
            "gender": random.choice(["Male", "Female"]),
            "insurance_provider": random.choice(insurances),
            "clinic_id": random.randint(1, 50)
        }
        patients.append(patient)
        
        # Assign condition (weighted towards chronic)
        weights = [15 if c.get("chronic", False) else 5 for c in conditions_db]
        condition = random.choices(conditions_db, weights=weights)[0]
        
        visit_date = (today - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        doctor = random.choice(doctors)
        
        # Generate clinical note using template
        note_text = generate_note_from_template(patient["full_name"], condition, doctor, visit_date)
        doctor_notes = random.choice(condition.get("note_templates", ["Routine visit."]))
        
        clinical_notes.append({
            "note_id": i + 100000,
            "patient_id": i,
            "visit_date": visit_date,
            "doctor_name": doctor,
            "diagnosis_code": condition["icd10_code"],
            "condition_name": condition["condition_name"],
            "note_text": note_text,
            "doctor_notes": doctor_notes.replace("{symptom}", random.choice(condition.get("symptoms", ["symptoms"])))
        })
        
        # Generate prescriptions (1-3 per patient)
        is_chronic = condition.get("chronic", False)
        num_rx = random.choices([1, 2, 3], weights=[50, 35, 15] if is_chronic else [60, 30, 10])[0]
        
        meds = condition.get("medications", ["Generic Med"])
        forms = condition.get("dosage_forms", ["tablet"])
        examples = condition.get("dosage_examples", ["100mg"])
        
        # Pick medications
        if len(meds) > num_rx:
            selected_meds = random.sample(meds, num_rx)
        else:
            selected_meds = meds[:num_rx]
        
        for j, med in enumerate(selected_meds):
            form_idx = min(j, len(forms) - 1)
            ex_idx = min(j, len(examples) - 1)
            
            prescriptions.append({
                "rx_id": i * 10 + j + 1,
                "patient_id": i,
                "medication_name": med,
                "dosage": generate_dosage(med, forms[form_idx], examples[ex_idx]),
                "form": forms[form_idx] if form_idx < len(forms) else "tablet",
                "drug_class": condition["condition_name"],
                "days_supply": random.choice([30, 60, 90]),
                "refills_remaining": random.randint(0, 5),
                "last_filled_date": (today - timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d"),
                "status": random.choices(["Active", "Active", "Expired", "Not Purchased"], weights=[65, 15, 15, 5])[0]
            })
    
    # 3. Write to CSVs
    logger.info("Starting CSV export...")
    
    def write_csv(filename, data_list, fieldnames):
        filepath = os.path.join(DATA_DIR, filename)
        logger.info(f"Writing {len(data_list)} rows to {filepath}...")
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_list)
        logger.info(f"Saved: {filename}")
    
    write_csv("clinics.csv", clinics, ["clinic_id", "name", "location"])
    write_csv("patients.csv", patients, ["patient_id", "full_name", "dob", "gender", "insurance_provider", "clinic_id"])
    write_csv("prescriptions.csv", prescriptions, ["rx_id", "patient_id", "medication_name", "dosage", "form", "drug_class", "days_supply", "refills_remaining", "last_filled_date", "status"])
    write_csv("clinical_notes.csv", clinical_notes, ["note_id", "patient_id", "visit_date", "doctor_name", "diagnosis_code", "condition_name", "note_text", "doctor_notes"])
    
    # Summary
    unique_patients = len(set(p["patient_id"] for p in patients))
    avg_rx = len(prescriptions) / unique_patients if unique_patients > 0 else 0
    
    logger.info("=" * 60)
    logger.info("SUCCESS: Generated realistic medical dataset")
    logger.info(f"Patients: {unique_patients}")
    logger.info(f"Prescriptions: {len(prescriptions)} (avg {avg_rx:.2f}/patient)")
    logger.info(f"Clinical Notes: {len(clinical_notes)}")
    logger.info(f"Conditions: {len(conditions_db)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_bulk_data()
