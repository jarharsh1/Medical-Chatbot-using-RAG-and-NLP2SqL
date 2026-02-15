"""
On-demand visualization service for the Medical Chatbot
Generates charts dynamically when users request them
"""

import os
import uuid
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from backend.config import DATA_DIR, PROJECT_ROOT


def load_data() -> Dict[str, pd.DataFrame]:
    """Load all data files"""
    return {
        'patients': pd.read_csv(os.path.join(DATA_DIR, 'patients.csv')),
        'prescriptions': pd.read_csv(os.path.join(DATA_DIR, 'prescriptions.csv')),
        'clinical_notes': pd.read_csv(os.path.join(DATA_DIR, 'clinical_notes.csv')),
        'clinics': pd.read_csv(os.path.join(DATA_DIR, 'clinics.csv')),
    }


def generate_chart(chart_type: str, data: Optional[Dict] = None) -> str:
    """
    Generate a chart on-demand and return the file path.
    
    Args:
        chart_type: Type of chart to generate
        data: Optional custom data dict (for SQL query results)
    
    Returns:
        Path to generated chart image
    """
    # Ensure output directory exists
    output_dir = os.path.join(PROJECT_ROOT, 'static', 'charts')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename
    filename = f"{chart_type}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Load data if not provided
    if data is None:
        data = load_data()
        # Convert date columns
        if 'prescriptions' in data:
            data['prescriptions']['last_filled_date'] = pd.to_datetime(
                data['prescriptions']['last_filled_date']
            )
        if 'patients' in data:
            data['patients']['dob'] = pd.to_datetime(data['patients']['dob'])
    
    # Generate the requested chart
    chart_functions = {
        'prescriptions_trend': _chart_prescriptions_trend,
        'gender_distribution': _chart_gender_distribution,
        'insurance_breakdown': _chart_insurance_breakdown,
        'top_medications': _chart_top_medications,
        'prescription_status': _chart_prescription_status,
        'drug_classes': _chart_drug_classes,
        'clinics_patients': _chart_clinics_patients,
        'age_distribution': _chart_age_distribution,
        'custom_data': _chart_custom_data,
    }
    
    chart_func = chart_functions.get(chart_type, _chart_prescriptions_trend)
    chart_func(data, filepath)
    
    return filepath


def _chart_prescriptions_trend(data: Dict, filepath: str):
    """Generate prescriptions trend line chart"""
    prescriptions = data['prescriptions']
    
    # Group by month
    prescriptions['month'] = prescriptions['last_filled_date'].dt.to_period('M')
    monthly_rx = prescriptions.groupby('month').size()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual data
    ax.plot(monthly_rx.index.astype(str), monthly_rx.values, 'b-o', 
            linewidth=2, markersize=6, label='Prescriptions')
    
    # Add trend line
    x_numeric = np.arange(len(monthly_rx))
    z = np.polyfit(x_numeric, monthly_rx.values, 1)
    p = np.poly1d(z)
    ax.plot(monthly_rx.index.astype(str), p(x_numeric), 'r--', 
            linewidth=2, label='Trend Line')
    
    ax.set_title('Prescription Trends Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Number of Prescriptions', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def _chart_gender_distribution(data: Dict, filepath: str):
    """Generate gender distribution pie chart"""
    patients = data['patients']
    gender_counts = patients['gender'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#FF6B6B', '#4ECDC4']
    ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
           colors=colors, startangle=90, explode=[0.02]*len(gender_counts))
    ax.set_title('Patient Gender Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def _chart_insurance_breakdown(data: Dict, filepath: str):
    """Generate insurance provider breakdown chart"""
    patients = data['patients']
    insurance_counts = patients['insurance_provider'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(insurance_counts.index, insurance_counts.values, color='#45B7D1')
    ax.set_xlabel('Number of Patients', fontsize=12)
    ax.set_title('Patients by Insurance Provider', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, insurance_counts.values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va='center')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def _chart_top_medications(data: Dict, filepath: str):
    """Generate top medications chart"""
    prescriptions = data['prescriptions']
    top_meds = prescriptions['medication_name'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_meds.index[::-1], top_meds.values[::-1], color='#96CEB4')
    ax.set_xlabel('Prescription Count', fontsize=12)
    ax.set_title('Top 10 Most Prescribed Medications', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, top_meds.values[::-1]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), va='center')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def _chart_prescription_status(data: Dict, filepath: str):
    """Generate prescription status chart"""
    prescriptions = data['prescriptions']
    status_counts = prescriptions['status'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ECC71', '#E74C3C', '#F39C12']
    bars = ax.bar(status_counts.index, status_counts.values, color=colors)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Prescription Status', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, status_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val), ha='center')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def _chart_drug_classes(data: Dict, filepath: str):
    """Generate drug classes chart"""
    prescriptions = data['prescriptions']
    drug_class_counts = prescriptions['drug_class'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(drug_class_counts.index, drug_class_counts.values, color='#FFEAA7')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Top Drug Classes', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def _chart_clinics_patients(data: Dict, filepath: str):
    """Generate clinics patient count chart"""
    patients = data['patients']
    clinic_patients = patients['clinic_id'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(clinic_patients.index.astype(str), clinic_patients.values, color='#DDA0DD')
    ax.set_xlabel('Clinic ID', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Top 10 Clinics by Patient Count', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def _chart_age_distribution(data: Dict, filepath: str):
    """Generate age distribution chart"""
    patients = data['patients']
    patients['age'] = (pd.Timestamp('2026-02-14') - patients['dob']).dt.days // 365
    
    age_bins = [0, 18, 30, 45, 60, 75, 100]
    age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
    patients['age_group'] = pd.cut(patients['age'], bins=age_bins, labels=age_labels)
    age_counts = patients['age_group'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(age_counts.index, age_counts.values, color='#87CEEB')
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Patient Age Distribution', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, age_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val), ha='center')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def _chart_custom_data(data: Dict, filepath: str):
    """Generate chart from custom data (e.g., SQL query results)"""
    if 'custom' not in data:
        # Default fallback
        _chart_prescriptions_trend(data, filepath)
        return
    
    custom_data = data['custom']
    labels = custom_data.get('labels', [])
    values = custom_data.get('values', [])
    chart_title = custom_data.get('title', 'Custom Data')
    chart_style = custom_data.get('style', 'bar')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_style == 'pie':
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    elif chart_style == 'line':
        ax.plot(labels, values, 'b-o', linewidth=2)
    else:  # bar
        ax.bar(labels, values, color='#45B7D1')
    
    ax.set_title(chart_title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


# Map user queries to chart types
QUERY_TO_CHART = {
    'prescription trend': 'prescriptions_trend',
    'prescription trends': 'prescriptions_trend',
    'prescriptions over time': 'prescriptions_trend',
    'medication trend': 'prescriptions_trend',
    'gender': 'gender_distribution',
    'gender distribution': 'gender_distribution',
    'male female': 'gender_distribution',
    'insurance': 'insurance_breakdown',
    'insurance providers': 'insurance_breakdown',
    'top medications': 'top_medications',
    'most prescribed': 'top_medications',
    'prescription status': 'prescription_status',
    'active prescriptions': 'prescription_status',
    'drug classes': 'drug_classes',
    'drug categories': 'drug_classes',
    'clinics': 'clinics_patients',
    'clinics patients': 'clinics_patients',
    'age distribution': 'age_distribution',
    'patient ages': 'age_distribution',
}


def detect_chart_request(user_query: str) -> Optional[str]:
    """
    Detect if user is asking for a visualization and return chart type.
    
    Args:
        user_query: The user's question
    
    Returns:
        Chart type string if visualization requested, None otherwise
    """
    query_lower = user_query.lower()
    
    for keywords, chart_type in QUERY_TO_CHART.items():
        if keywords in query_lower:
            return chart_type
    
    return None
