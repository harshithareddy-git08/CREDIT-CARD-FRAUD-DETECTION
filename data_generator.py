import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=10000, fraud_ratio=0.002):
    print(f"Generating {num_samples} samples with fraud ratio {fraud_ratio}...")
    
    # Time: Increasing sequence
    time = np.arange(num_samples)
    
    # V1-V28: Normally distributed features
    v_features = np.random.randn(num_samples, 28)
    
    # Amount: Log-normal distribution
    amount = np.random.lognormal(mean=2, sigma=1, size=num_samples)
    
    # Class: Imbalanced classes
    num_fraud = int(num_samples * fraud_ratio)
    classes = np.zeros(num_samples)
    fraud_indices = np.random.choice(num_samples, num_fraud, replace=False)
    classes[fraud_indices] = 1
    
    # Create DataFrame
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    data = np.column_stack((time, v_features, amount, classes))
    df = pd.DataFrame(data, columns=columns)
    
    # Cast Class to int
    df['Class'] = df['Class'].astype(int)
    
    # Save to CSV
    output_path = 'creditcard.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully saved synthetic data to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
