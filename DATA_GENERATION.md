# Medical Dataset Generation

This document describes how the realistic medical dataset is generated for the Medical AI Analytics Platform.

## Overview

The dataset is generated using LLM (qwen2.5:14b) to create diverse, medically-accurate synthetic data including:
- **65,375 patients** with realistic demographics
- **116,101 prescriptions** with proper medication-dosage-form relationships
- **65,375 clinical notes** in SOAP format
- **50 clinics** across multiple locations

## Dataset Statistics

| File | Records | Description |
|------|---------|-------------|
| `patients.csv` | 65,375 | Patient demographics |
| `prescriptions.csv` | 116,101 | Prescription records (avg 1.78/patient) |
| `clinical_notes.csv` | 65,375 | SOAP-format clinical documentation |
| `clinics.csv` | 50 | Healthcare facility locations |

## Medical Conditions Generated

The LLM generates 20 diverse medical conditions including:

### Chronic Conditions
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
11. Osteoporosis (M81.0)

### Acute Conditions
12. Pneumonia (J18.9)
13. UTI (N39.0)
14. Cellulitis (L03.90)
15. Back Pain (M54.5)
16. Ear Infection (H66.90)

### Other
17. Allergic Rhinitis (J30.9)
18. Migraine (G43.9)
19. Iron Deficiency Anemia (D50.9)
20. Acne Vulgaris (L70.9)

## Dosage Formats

Realistic dosage formats based on medication form:

| Form | Examples |
|------|----------|
| Tablet | 25mg, 50mg, 100mg, 200mg, 500mg, 10mg once daily |
| Capsule | 25mg, 50mg, 100mg, 75mg/25mg, 25mcg, 50mcg |
| Inhaler | 90mcg/actuation, 50mcg/actuation, 2 puffs, 1-2 puffs PRN |
| Liquid | 5ml, 10ml, 15ml, 5ml twice daily |
| Cream | 1% cream, 0.5% ointment, thin layer |
| Injection | 10 units, 20 units, 0.5ml IM, subcutaneous |
| Drops | 1-2 drops, 2 drops OU, one drop per nostril |
| Syrup | 5ml, 10ml, 15ml, 1 tbsp |
| Spray | 1 spray each nostril, 2 sprays per nostril |
| Patch | 25mcg/hr, 50mcg/hr, apply once weekly |

## Regenerating the Dataset

### Prerequisites

1. Python 3.10+
2. Ollama running with qwen2.5:14b model
3. Required packages: `langchain-ollama`, `faker`, `tqdm`

### Steps

```bash
# Navigate to project root
cd Medical-Chatbot-using-RAG-and-NLP2SqL

# Delete old data files
rm -f data/*.csv
rm -f backend/medical_records.db
rm -rf backend/chroma_db

# Generate new dataset
python scripts/generate_data.py
```

### Output

The script will:
1. Call LLM to generate 20 medical conditions (cached to `data/medical_conditions_cache.json`)
2. Generate patients with realistic demographics
3. Assign conditions weighted towards chronic diseases
4. Generate prescriptions with proper medication-form relationships
5. Create SOAP-format clinical notes
6. Write all CSV files

### First Run vs Subsequent Runs

- **First run**: LLM generates conditions (takes ~30 seconds)
- **Subsequent runs**: Uses cached conditions (instant)

## Clinical Note Format

Clinical notes use SOAP format:

```text
SUBJECTIVE: Patient presents with [chief complaint].
Reports [symptom1] and [symptom2].

OBJECTIVE:
- BP: [vital signs]
- HR: [vitals]
- General: [observation]

ASSESSMENT: [Condition] - [severity]

PLAN:
1. [Action item]
2. [Follow-up]
3. [Patient education]

Signed: Dr. [Doctor Name]
Date: YYYY-MM-DD
```

## Prescriptions Schema

| Column | Description |
|--------|-------------|
| rx_id | Unique prescription identifier |
| patient_id | Reference to patient |
| medication_name | Drug name |
| dosage | Realistic dosage (e.g., "500mg", "90mcg/actuation") |
| form | Medication form (tablet, capsule, inhaler, etc.) |
| drug_class | Medical condition treated |
| days_supply | 30, 60, or 90 days |
| refills_remaining | 0-5 |
| last_filled_date | Date of last fill |
| status | Active, Expired, or Not Purchased |

## Customization

### Modifying Conditions

Edit the LLM prompt in `scripts/generate_data.py` to add/remove conditions:

```python
prompt = """
Generate 20 diverse medical conditions...
Required conditions:
1. [Your condition 1]
2. [Your condition 2]
...
"""
```

### Adjusting Patient Count

Change `NUM_PATIENTS` in `scripts/generate_data.py`:

```python
NUM_PATIENTS = 100000  # For larger dataset
```

### Prescription Distribution

Modify the prescription count weights:

```python
# Chronic patients
num_rx = random.choices([1, 2, 3, 4], weights=[45, 30, 15, 10])[0]
```

## License

The generated dataset is synthetic and for demonstration purposes. Ensure compliance with HIPAA/GDPR when using real patient data.
