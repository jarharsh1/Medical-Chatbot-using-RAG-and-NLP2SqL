"""
Visualization script using actual project data from data/ directory
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load actual data
patients = pd.read_csv('data/patients.csv')
prescriptions = pd.read_csv('data/prescriptions.csv')
clinical_notes = pd.read_csv('data/clinical_notes.csv')
clinics = pd.read_csv('data/clinics.csv')

# Convert dates
prescriptions['last_filled_date'] = pd.to_datetime(prescriptions['last_filled_date'])
patients['dob'] = pd.to_datetime(patients['dob'])

print("Data loaded successfully!")
print(f"Patients: {len(patients)} records")
print(f"Prescriptions: {len(prescriptions)} records")
print(f"Clinical Notes: {len(clinical_notes)} records")
print(f"Clinics: {len(clinics)} records")

# ===== CHART 1: Prescriptions over time (Trend Line) =====
fig, ax = plt.subplots(figsize=(12, 6))

# Group by month
prescriptions['month'] = prescriptions['last_filled_date'].dt.to_period('M')
monthly_rx = prescriptions.groupby('month').size()

# Plot trend
ax.plot(monthly_rx.index.astype(str), monthly_rx.values, 'b-o', linewidth=2, markersize=6, label='Prescriptions')

# Add trend line
x_numeric = np.arange(len(monthly_rx))
z = np.polyfit(x_numeric, monthly_rx.values, 1)
p = np.poly1d(z)
ax.plot(monthly_rx.index.astype(str), p(x_numeric), 'r--', linewidth=2, label='Trend Line')

ax.set_title('Prescription Trends Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Prescriptions', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prescriptions_trend.png', dpi=150)
plt.close()
print("[OK] Saved: prescriptions_trend.png")

# ===== CHART 2: Gender Distribution =====
fig, ax = plt.subplots(figsize=(8, 8))
gender_counts = patients['gender'].value_counts()
colors = ['#FF6B6B', '#4ECDC4']
ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
       colors=colors, startangle=90, explode=[0.02]*len(gender_counts))
ax.set_title('Patient Gender Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gender_distribution.png', dpi=150)
plt.close()
print("[OK] Saved: gender_distribution.png")

# ===== CHART 3: Insurance Provider Breakdown =====
fig, ax = plt.subplots(figsize=(10, 6))
insurance_counts = patients['insurance_provider'].value_counts()
bars = ax.barh(insurance_counts.index, insurance_counts.values, color='#45B7D1')
ax.set_xlabel('Number of Patients', fontsize=12)
ax.set_title('Patients by Insurance Provider', fontsize=14, fontweight='bold')
for bar, val in zip(bars, insurance_counts.values):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va='center')
plt.tight_layout()
plt.savefig('insurance_breakdown.png', dpi=150)
plt.close()
print("[OK] Saved: insurance_breakdown.png")

# ===== CHART 4: Top Medications =====
fig, ax = plt.subplots(figsize=(10, 6))
top_meds = prescriptions['medication_name'].value_counts().head(10)
bars = ax.barh(top_meds.index[::-1], top_meds.values[::-1], color='#96CEB4')
ax.set_xlabel('Prescription Count', fontsize=12)
ax.set_title('Top 10 Most Prescribed Medications', fontsize=14, fontweight='bold')
for bar, val in zip(bars, top_meds.values[::-1]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), va='center')
plt.tight_layout()
plt.savefig('top_medications.png', dpi=150)
plt.close()
print("[OK] Saved: top_medications.png")

# ===== CHART 5: Prescription Status =====
fig, ax = plt.subplots(figsize=(8, 6))
status_counts = prescriptions['status'].value_counts()
colors = ['#2ECC71', '#E74C3C', '#F39C12']
bars = ax.bar(status_counts.index, status_counts.values, color=colors)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Prescription Status', fontsize=14, fontweight='bold')
for bar, val in zip(bars, status_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val), ha='center')
plt.tight_layout()
plt.savefig('prescription_status.png', dpi=150)
plt.close()
print("[OK] Saved: prescription_status.png")

# ===== CHART 6: Drug Class Distribution =====
fig, ax = plt.subplots(figsize=(10, 6))
drug_class_counts = prescriptions['drug_class'].value_counts().head(10)
bars = ax.bar(drug_class_counts.index, drug_class_counts.values, color='#FFEAA7')
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Top Drug Classes', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('drug_classes.png', dpi=150)
plt.close()
print("[OK] Saved: drug_classes.png")

# ===== CHART 7: Patients per Clinic =====
fig, ax = plt.subplots(figsize=(10, 6))
clinic_patients = patients['clinic_id'].value_counts().head(10)
bars = ax.bar(clinic_patients.index.astype(str), clinic_patients.values, color='#DDA0DD')
ax.set_xlabel('Clinic ID', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('Top 10 Clinics by Patient Count', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('clinics_patients.png', dpi=150)
plt.close()
print("[OK] Saved: clinics_patients.png")

# ===== CHART 8: Age Distribution =====
fig, ax = plt.subplots(figsize=(10, 6))
patients['age'] = (pd.Timestamp('2026-02-14') - patients['dob']).dt.days // 365
age_bins = [0, 18, 30, 45, 60, 75, 100]
age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
patients['age_group'] = pd.cut(patients['age'], bins=age_bins, labels=age_labels)
age_counts = patients['age_group'].value_counts().sort_index()
bars = ax.bar(age_counts.index, age_counts.values, color='#87CEEB')
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('Patient Age Distribution', fontsize=14, fontweight='bold')
for bar, val in zip(bars, age_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val), ha='center')
plt.tight_layout()
plt.savefig('age_distribution.png', dpi=150)
plt.close()
print("[OK] Saved: age_distribution.png")

print("\nAll charts generated from your actual project data!")
