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

NUM_PATIENTS = 65375
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OS_SEED = 42

fake = Faker()
Faker.seed(OS_SEED)
random.seed(OS_SEED)

today = datetime.now()

# --- REALISTIC DOSAGE FORMATS ---

DOSAGE_FORMATS = {
    "tablet": ["25mg", "50mg", "100mg", "200mg", "500mg", "10mg once daily", "20mg twice daily", "325mg", "600mg", "800mg"],
    "capsule": ["25mg", "50mg", "100mg", "75mg/25mg", "10/325mg", "25mcg", "50mcg", "75mcg", "150mg", "200mg"],
    "liquid": ["5ml", "10ml", "15ml", "5ml twice daily", "10ml once daily", "2.5ml", "1 tsp (5ml)", "15ml three times daily"],
    "inhaler": ["90mcg/actuation", "50mcg/actuation", "110mcg/actuation", "2 puffs", "1-2 puffs PRN", "100mcg/dose", "SABA rescue inhaler"],
    "cream": ["1% cream", "0.5% ointment", "2.5% gel", "thin layer", "pea-sized amount", "apply twice daily", "apply to affected area"],
    "patch": ["25mcg/hr", "50mcg/hr", "100mcg/hr", "apply once weekly", "72-hour patch", "apply every 24 hours"],
    "injection": ["10 units", "20 units", "0.5ml IM", "1ml IM", "subcutaneous", "25 units SC", "40 units", "0.1ml intralesional"],
    "drops": ["1-2 drops", "2 drops OU", "1 drop OS", "one drop per nostril", "2 drops in each ear", "ophthalmic drops"],
    "syrup": ["5ml", "10ml", "15ml", "1 tbsp (15ml)", "2 tsp (10ml)", "5ml every 6 hours", "10ml twice daily"],
    "spray": ["1 spray each nostril", "2 sprays per nostril", "nasal spray", "1 spray"],
    "suppository": ["25mg", "50mg", "10mg pediatric", "insert rectally"],
    "pessary": ["100mg", "500mg", "insert vaginally"],
}

def get_dosage(form: str) -> str:
    """Get realistic dosage for medication form."""
    if form in DOSAGE_FORMATS:
        return random.choice(DOSAGE_FORMATS[form])
    return random.choice(DOSAGE_FORMATS.get("tablet", ["100mg"]))

# --- LLM-BASED MEDICAL DATA ---

def load_cached_conditions() -> List[Dict]:
    """Load cached medical conditions or generate new ones using LLM."""
    cache_file = os.path.join(DATA_DIR, "medical_conditions_cache.json")
    
    if os.path.exists(cache_file):
        logger.info("Loading cached medical conditions...")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    logger.info("Generating medical conditions using LLM...")
    conditions = generate_medical_conditions_llm()
    
    with open(cache_file, 'w') as f:
        json.dump(conditions, f, indent=2)
    logger.info(f"Cached {len(conditions)} conditions to {cache_file}")
    
    return conditions


def generate_medical_conditions_llm() -> List[Dict]:
    """Generate realistic medical conditions with medications using LLM."""
    
    prompt = """
Generate 20 diverse medical conditions as JSON. Include chronic diseases and acute conditions.

Return JSON:
{
    "conditions": [
        {
            "condition_name": "Type 2 Diabetes Mellitus",
            "icd10_code": "E11.9",
            "medications": ["Metformin 500mg", "Glipizide 5mg", "Empagliflozin 10mg", "Insulin Glargine 20 units", "Glimepiride 2mg"],
            "medication_forms": ["tablet", "tablet", "tablet", "injection", "tablet"],
            "symptoms": ["increased thirst", "frequent urination", "fatigue", "blurred vision", "slow healing wounds"],
            "chronic": true,
            "severity_levels": ["well-controlled", "fairly controlled", "poorly controlled"]
        }
    ]
}

Required conditions:
1. Type 2 Diabetes Mellitus (E11.9)
2. Hypertension (I10)
3. Asthma (J45.909)
4. COPD (J44.9)
5. Hyperlipidemia (E78.5)
6. Hypothyroidism (E03.9)
7. Depression (F32.9)
8. Anxiety Disorder (F41.9)
9. GERD (K21.9)
10. Osteoarthritis (M19.90)
11. Allergic Rhinitis (J30.9)
12. Migraine (G43.9)
13. Iron Deficiency Anemia (D50.9)
14. Osteoporosis (M81.0)
15. Acne Vulgaris (L70.9)
16. Pneumonia (J18.9)
17. UTI (N39.0)
18. Cellulitis (L03.90)
19. Back Pain (M54.5)
20. Ear Infection (H66.90)

For each condition, provide:
- Realistic medications with dosages
- Appropriate forms (tablet, capsule, inhaler, cream, injection, drops, etc.)
- Common symptoms
- Whether chronic or acute

Make it realistic and medically accurate.
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
        
        # Parse medications to extract name and dosage
        parsed_conditions = []
        for c in conditions:
            meds = []
            forms = c.get("medication_forms", [])
            for i, med_str in enumerate(c.get("medications", [])):
                # Parse "Name Dosage" format
                parts = med_str.rsplit(None, 1)
                if len(parts) == 2:
                    name, dosage = parts
                else:
                    name = med_str
                    dosage = get_dosage(forms[i] if i < len(forms) else "tablet")
                meds.append({
                    "name": name,
                    "dosage": dosage,
                    "form": forms[i] if i < len(forms) else "tablet"
                })
            
            parsed_conditions.append({
                "condition_name": c["condition_name"],
                "icd10_code": c["icd10_code"],
                "medications": meds,
                "symptoms": c.get("symptoms", []),
                "chronic": c.get("chronic", False),
                "severity": c.get("severity_levels", ["stable"])[0]
            })
        
        logger.info(f"LLM generated {len(parsed_conditions)} conditions")
        return parsed_conditions
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise


def generate_clinical_note(patient: Dict, condition: Dict, doctor: str) -> str:
    """Generate realistic clinical note."""
    symptoms = condition.get("symptoms", ["general complaints"])
    severity = condition.get("severity", "stable")
    
    chief_complaint = random.choice(symptoms)
    symptom2 = random.choice([s for s in symptoms if s != chief_complaint] or symptoms)
    
    templates = [
        f"""CHIEF COMPLAINT: {patient['full_name']} presents with {chief_complaint}.

HPI: {patient['gender']} patient, age {random.randint(18, 80)}, presents with {chief_complaint} for {random.randint(1, 8)} weeks. 
Also reports {symptom2}. Symptoms affect daily activities. No known triggers identified.

MEDICAL HISTORY: {condition['condition_name']} - {severity}

MEDICATIONS: {', '.join([m['name'] for m in condition['medications'][:3]])}

ALLERGIES: NKDA

ASSESSMENT: {condition['condition_name']} (ICD-10: {condition['icd10_code']}) - {random.choice(['stable', 'improving', 'requires adjustment'])}

PLAN:
1. Continue current medications
2. Follow-up in {random.choice([2, 4, 6, 8, 12])} weeks
3. Labs/tests: {random.choice(['CBC', 'CMP', 'Lipid panel', 'HbA1c', 'TSH', 'CRP', 'ESR'])}
4. Patient education provided

Provider: {doctor}
Date: {(today - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d')}""",
        
        f"""SUBJECTIVE: {patient['full_name']} here for follow-up of {condition['condition_name']}. 
Reports {chief_complaint} and {symptom2}. 
Symptoms rated {random.randint(1, 10)}/10 severity.

OBJECTIVE:
- BP: {random.randint(110, 160)}/{random.randint(60, 100)}
- HR: {random.randint(60, 100)}
- General: Alert, oriented, no acute distress

ASSESSMENT: {condition['condition_name']} - {severity}. 
Current regimen partially effective.

PLAN:
1. Optimize therapy
2. Monitor symptoms
3. Return if {random.choice(['symptoms worsen', 'no improvement in 2 weeks', 'side effects develop'])}

Signed: {doctor}"""
    ]
    
    return random.choice(templates)


def generate_bulk_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory: {DATA_DIR}")
    
    conditions_db = load_cached_conditions()
    
    # 1. Generate Clinics
    logger.info("Generating Clinics...")
    clinics = []
    clinic_names = ["Downtown Medical", "Westside Health", "North Hills Clinic", "Valley Family", 
                   "City Center", "Lakeside Wellness", "Riverside Primary", "Central Care",
                   "Metro Medical", "Community Health", "Family First", "Prime Care", "Sunrise Medical"]
    locations = ["New York", "Chicago", "San Francisco", "Austin", "Seattle", "Boston", "Miami", 
                 "Denver", "Phoenix", "Los Angeles", "Atlanta", "Portland", "Dallas", "Houston"]
    
    for i in range(1, 51):
        clinics.append({
            "clinic_id": i,
            "name": f"{random.choice(clinic_names)} - {random.choice(['Main', 'North', 'South', 'East', 'West'])}",
            "location": random.choice(locations)
        })
    
    # 2. Generate Patients
    logger.info(f"Generating {NUM_PATIENTS} patients...")
    
    patients = []
    prescriptions = []
    clinical_notes = []
    
    doctors = [f"Dr. {fake.first_name()} {fake.last_name()}" for _ in range(100)]
    insurances = ["BlueCross BlueShield", "Aetna", "Medicare", "UnitedHealthcare", "Cigna", "Kaiser Permanente", "Humana"]
    
    for i in tqdm(range(1, NUM_PATIENTS + 1), desc="Generating Data"):
        patient = {
            "patient_id": i,
            "full_name": fake.name(),
            "dob": fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y-%m-%d"),
            "gender": random.choice(["Male", "Female"]),
            "insurance_provider": random.choice(insurances),
            "clinic_id": random.randint(1, 50)
        }
        patients.append(patient)
        
        # Weighted towards chronic conditions
        weights = [20 if c.get("chronic", False) else 5 for c in conditions_db]
        condition = random.choices(conditions_db, weights=weights)[0]
        
        doctor = random.choice(doctors)
        
        # Clinical note
        note_text = generate_clinical_note(patient, condition, doctor)
        clinical_notes.append({
            "note_id": i + 100000,
            "patient_id": i,
            "visit_date": (today - timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d"),
            "doctor_name": doctor,
            "diagnosis_code": condition["icd10_code"],
            "condition_name": condition["condition_name"],
            "note_text": note_text,
            "doctor_notes": f"Follow-up for {condition['condition_name']}. {random.choice(condition.get('symptoms', ['Symptoms noted']))}."
        })
        
        # Prescriptions (1-4 per patient)
        is_chronic = condition.get("chronic", False)
        num_rx = random.choices([1, 2, 3, 4], 
                               weights=[45, 30, 15, 10] if is_chronic else [60, 25, 10, 5])[0]
        
        meds = condition.get("medications", [{"name": "Medication", "dosage": "100mg", "form": "tablet"}])
        
        # Select medications
        selected_meds = random.sample(meds, min(num_rx, len(meds)))
        
        for j, med in enumerate(selected_meds):
            days_supply = random.choice([30, 60, 90])
            
            status_roll = random.random()
            if status_roll < 0.65:
                status = "Active"
                days_ago = random.randint(1, days_supply - 1)
            elif status_roll < 0.80:
                status = "Active"
                days_ago = days_supply - random.randint(1, 5)
            elif status_roll < 0.95:
                status = "Expired"
                days_ago = days_supply + random.randint(10, 60)
            else:
                status = "Not Purchased"
                days_supply = 0
            
            last_filled = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d") if status != "Not Purchased" else None
            
            prescriptions.append({
                "rx_id": i * 10 + j + 1,
                "patient_id": i,
                "medication_name": med["name"],
                "dosage": med["dosage"],
                "form": med["form"],
                "drug_class": condition["condition_name"],
                "days_supply": days_supply,
                "refills_remaining": random.randint(0, 5) if status == "Active" else 0,
                "last_filled_date": last_filled,
                "status": status
            })
    
    # 3. Write CSVs
    logger.info("Writing CSV files...")
    
    def write_csv(filename, data_list, fieldnames):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_list)
        logger.info(f"Saved: {filename} ({len(data_list)} rows)")
    
    write_csv("clinics.csv", clinics, ["clinic_id", "name", "location"])
    write_csv("patients.csv", patients, ["patient_id", "full_name", "dob", "gender", "insurance_provider", "clinic_id"])
    write_csv("prescriptions.csv", prescriptions, ["rx_id", "patient_id", "medication_name", "dosage", "form", "drug_class", "days_supply", "refills_remaining", "last_filled_date", "status"])
    write_csv("clinical_notes.csv", clinical_notes, ["note_id", "patient_id", "visit_date", "doctor_name", "diagnosis_code", "condition_name", "note_text", "doctor_notes"])
    
    # Summary
    avg_rx = len(prescriptions) / NUM_PATIENTS
    logger.info("=" * 60)
    logger.info("SUCCESS: Realistic medical dataset generated")
    logger.info(f"Patients: {NUM_PATIENTS}")
    logger.info(f"Prescriptions: {len(prescriptions)} (avg {avg_rx:.2f}/patient)")
    logger.info(f"Clinical Notes: {len(clinical_notes)}")
    logger.info(f"Conditions: {len(conditions_db)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_bulk_data()
