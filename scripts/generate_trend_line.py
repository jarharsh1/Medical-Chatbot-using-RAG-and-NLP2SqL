"""
Example script to generate trend line visualizations
This demonstrates image generation capability for data analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example: Generate trend line for patient visits over time
def generate_trend_line():
    # Sample data - in real usage, this would come from your database
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
    visits = np.cumsum(np.random.randint(5, 20, len(dates)))  # Cumulative visits with random growth
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual data
    ax.scatter(dates, visits, alpha=0.5, label='Actual Visits', color='blue')
    
    # Calculate and plot trend line (linear regression)
    x_numeric = np.arange(len(dates))
    z = np.polyfit(x_numeric, visits, 1)
    p = np.poly1d(z)
    ax.plot(dates, p(x_numeric), 'r--', linewidth=2, label='Trend Line')
    
    # Formatting
    ax.set_title('Patient Visits Trend - 2024', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Visits', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trend_line_example.png', dpi=150)
    print("Trend line chart saved as 'trend_line_example.png'")
    plt.close()

# Example: Generate bar chart for conditions
def generate_condition_chart():
    conditions = ['Hypertension', 'Diabetes', 'Asthma', 'Arthritis', 'Migraine']
    counts = [145, 132, 98, 76, 54]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(conditions, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    
    ax.set_title('Top Medical Conditions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Patient Count', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('conditions_chart.png', dpi=150)
    print("Conditions chart saved as 'conditions_chart.png'")
    plt.close()

if __name__ == '__main__':
    generate_trend_line()
    generate_condition_chart()
    print("All charts generated successfully!")
