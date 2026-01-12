import os
import csv
import random
import json
import time
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, Field
from faker import Faker
from tqdm import tqdm

# Updated Import to fix deprecation warning
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
NUM_PATIENTS = 35000  # Will generate ~105,000 total rows
DATA_DIR = "data"
OS_SEED = 42

fake = Faker()
Faker.seed(OS_SEED)
random.seed(OS_SEED)

# --- PYDANTIC MODELS (Schema Definition) ---

class Clinic(BaseModel):
    clinic_id: int = Field(..., description="Unique identifier for the clinic.")
    name: str = Field(..., description="The name of the medical facility.")
    location: str = Field(..., description="The physical city or region location.")

class Patient(BaseModel):
    patient_id: int = Field(..., description="Unique identifier for the patient.")
    full_name: str = Field(..., description="The full legal name of the patient.")
    dob: str = Field(..., description="Date of birth in YYYY-MM-DD format.")
    gender: str = Field(..., description="Gender of the patient.")
    insurance_provider: str = Field(..., description="Insurance company.")
    clinic_id: int = Field(..., description="Foreign key referencing the primary clinic.")

class ClinicalNote(BaseModel):
    note_id: int = Field(..., description="Unique identifier for the clinical note.")
    patient_id: int = Field(..., description="Foreign key referencing the patient.")
    visit_date: str = Field(..., description="Date of visit YYYY-MM-DD.")
    doctor_name: str = Field(..., description="Name of the physician.")
    diagnosis_code: str = Field(..., description="ICD-10 code.")
    condition_name: str = Field(..., description="Name of the medical condition.")
    note_text: str = Field(..., description="Unstructured clinical text.")

class Prescription(BaseModel):
    rx_id: int = Field(..., description="Unique identifier for the prescription.")
    patient_id: int = Field(..., description="Foreign key referencing the patient.")
    medication_name: str = Field(..., description="Name of the prescribed drug.")
    dosage: str = Field(..., description="Dosage instructions.")
    days_supply: int = Field(..., description="Days supply.")
    refills_remaining: int = Field(..., description="Refills remaining.")
    last_filled_date: str = Field(..., description="Date last filled YYYY-MM-DD.")
    status: str = Field(..., description="Status (Active/Expired).")

# --- OLLAMA SEED GENERATOR ---

class MedicalSeeder:
    """
    Uses Ollama to generate realistic medical templates.
    """
    def __init__(self):
        logger.info("Initializing Ollama (llama3.2) for medical context generation...")
        try:
            self.llm = ChatOllama(model="llama3.2", temperature=0.8, format="json")
        except Exception as e:
            logger.error(f"Error: Ollama initialization failed. {e}")
            exit(1)

    def generate_medical_knowledge(self) -> List[dict]:
        logger.info("Asking AI for diverse medical conditions and treatment plans...")
        
        prompt = """
        You are a medical data generator. Generate a JSON object containing a list of exactly 20 distinct medical conditions.
        
        You must return a JSON object with this exact structure:
        {
            "conditions": [
                {
                    "condition": "Condition Name",
                    "code": "ICD-10 Code",
                    "meds": ["Medication Name Dosage", "Medication Name Dosage"],
                    "templates": ["Clinical note template 1 with {name}", "Clinical note template 2 with {name}"]
                }
            ]
        }

        Requirements for the data:
        1. "condition": Real medical condition name (e.g. Type 2 Diabetes, Hypertension, Asthma).
        2. "code": Valid ICD-10 code.
        3. "meds": A list of 2 common medications used to treat this condition, including dosages.
        4. "templates": A list of 3 distinct, realistic clinical note templates. Use {name} as a placeholder for the patient's name.
        
        DO NOT return an empty list. Populate "conditions" with 20 diverse items.
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            content = response.content.strip()
            
            # --- Robust JSON Cleanup ---
            # Llama often wraps JSON in markdown blocks like ```json ... ```
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            data = json.loads(content)
            conditions = data.get("conditions", [])
            
            if not conditions:
                logger.warning("AI returned valid JSON but empty list. Using fallback.")
                return self.get_fallback_data()
                
            logger.info(f"AI successfully generated {len(conditions)} medical profiles.")
            return conditions

        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Failed: {e}. Content received: {content[:100]}...")
            return self.get_fallback_data()
        except Exception as e:
            logger.error(f"AI Generation Failed: {e}. Using fallback data.")
            return self.get_fallback_data()

    def get_fallback_data(self):
        logger.info("Loading Fallback Medical Data...")
        return [
            {
                "condition": "Hypertension", "code": "I10", 
                "meds": ["Lisinopril 10mg", "Amlodipine 5mg"],
                "templates": ["{name} reports persistent headaches. BP 150/95.", "Follow-up for {name}. BP controlled at 130/85.", "Patient {name} advised to reduce sodium intake."]
            },
            {
                "condition": "Type 2 Diabetes", "code": "E11.9",
                "meds": ["Metformin 500mg", "Glipizide 5mg"],
                "templates": ["{name} showing signs of insulin resistance.", "Routine diabetic check for {name}. HbA1c 7.2%.", "{name} complains of increased thirst and fatigue."]
            },
            {
                "condition": "Asthma", "code": "J45.909",
                "meds": ["Albuterol 90mcg", "Fluticasone 110mcg"],
                "templates": ["{name} reports wheezing after exercise.", "Seasonal allergies triggering asthma for {name}.", "{name} needs refill on rescue inhaler."]
            },
            {
                "condition": "Hyperlipidemia", "code": "E78.5",
                "meds": ["Atorvastatin 20mg", "Simvastatin 40mg"],
                "templates": ["Lipid panel shows elevated LDL for {name}.", "{name} advised on low-cholesterol diet.", "Routine follow-up for high cholesterol for {name}."]
            },
            {
                "condition": "Hypothyroidism", "code": "E03.9",
                "meds": ["Levothyroxine 50mcg", "Levothyroxine 75mcg"],
                "templates": ["TSH levels elevated for {name}.", "{name} reports fatigue and weight gain.", "Adjusting dosage for {name}."]
            }
        ]

# --- BULK GENERATOR ---

def generate_bulk_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory: {DATA_DIR}")

    # 1. Get Seeds from AI
    seeder = MedicalSeeder()
    conditions_db = seeder.generate_medical_knowledge()
    
    # CRITICAL FIX: Ensure we never proceed with empty conditions
    if not conditions_db:
        logger.warning("Conditions DB is empty despite fallback attempts. Forcing hardcoded fallback.")
        conditions_db = seeder.get_fallback_data()

    # 2. Generate Clinics (Static 50)
    logger.info("Generating Clinics...")
    clinics = []
    clinic_names = ["Downtown", "Westside", "North Hills", "Valley", "City", "Lakeside", "Riverside", "Central"]
    clinic_types = ["Health", "Clinic", "Family Practice", "Wellness Center", "Urgent Care"]
    locations = ["New York", "Chicago", "San Francisco", "Austin", "Seattle", "Boston", "Miami"]
    
    for i in range(1, 51):
        clinics.append(Clinic(
            clinic_id=i,
            name=f"{random.choice(clinic_names)} {random.choice(clinic_types)}",
            location=random.choice(locations)
        ))

    # 3. Generate Patients, Notes, Rx
    logger.info(f"Generating {NUM_PATIENTS} patients and associated records...")
    
    patients = []
    notes = []
    prescriptions = []
    
    doctors = [f"Dr. {fake.last_name()}" for _ in range(100)]
    insurances = ["BlueCross", "Aetna", "Medicare", "UnitedHealth", "Cigna", "Kaiser"]
    
    today = datetime.now()
    
    for i in tqdm(range(1, NUM_PATIENTS + 1), desc="Generating Data"):
        # Patient
        p_clinic = random.randint(1, 50)
        p = Patient(
            patient_id=i,
            full_name=fake.name(),
            dob=fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y-%m-%d"),
            gender=random.choice(["Male", "Female"]),
            insurance_provider=random.choice(insurances),
            clinic_id=p_clinic
        )
        patients.append(p)

        # Assign a random medical condition from the AI DB
        # SAFETY CHECK: Random.choice will crash if list is empty, but we handled that above.
        cond = random.choice(conditions_db)
        
        # Clinical Note
        visit_delta = random.randint(0, 365)
        visit_date = (today - timedelta(days=visit_delta)).strftime("%Y-%m-%d")
        
        # Handle templates safely
        templates = cond.get("templates", ["Patient {name} visited for routine checkup."])
        if not templates: templates = ["Patient {name} visited for routine checkup."]
        note_template = random.choice(templates)
        
        n = ClinicalNote(
            note_id=i + 1000,
            patient_id=i,
            visit_date=visit_date,
            doctor_name=random.choice(doctors),
            diagnosis_code=cond.get("code", "R69"),
            condition_name=cond.get("condition", "General"),
            note_text=note_template.replace("{name}", p.full_name)
        )
        notes.append(n)

        # Prescription
        meds_list = cond.get("meds", ["Vitamin D 1000IU"])
        if not meds_list: meds_list = ["Vitamin D 1000IU"]
        med_str = random.choice(meds_list)
        
        parts = med_str.split(" ")
        med_name = parts[0]
        dosage = " ".join(parts[1:]) if len(parts) > 1 else "Standard"
        
        days_supply = random.choice([30, 60, 90])
        status_roll = random.random()
        
        if status_roll < 0.7: 
            # Good
            days_ago = random.randint(1, days_supply - 5)
            status = "Active"
        elif status_roll < 0.9:
            # Due Soon
            days_ago = days_supply - 2
            status = "Active"
        else:
            # Overdue
            days_ago = days_supply + random.randint(10, 50)
            status = "Expired"

        last_filled = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        rx = Prescription(
            rx_id=i + 5000,
            patient_id=i,
            medication_name=med_name,
            dosage=dosage,
            days_supply=days_supply,
            refills_remaining=random.randint(0, 5),
            last_filled_date=last_filled,
            status=status
        )
        prescriptions.append(rx)

    # 4. Write to CSVs
    logger.info("Starting CSV export...")
    
    def write_csv(filename, data_list, model_class):
        filepath = os.path.join(DATA_DIR, filename)
        logger.info(f"Writing {len(data_list)} rows to {filepath}...")
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(model_class.model_fields.keys())
            for item in data_list:
                writer.writerow(item.model_dump().values())
        logger.info(f"Successfully saved {filename}")

    write_csv("clinics.csv", clinics, Clinic)
    write_csv("patients.csv", patients, Patient)
    write_csv("clinical_notes.csv", notes, ClinicalNote)
    write_csv("prescriptions.csv", prescriptions, Prescription)

    logger.info("SUCCESS: Generated massive dataset.")
    logger.info(f"Total Rows: {len(clinics) + len(patients) + len(notes) + len(prescriptions)}")

if __name__ == "__main__":
    generate_bulk_data()