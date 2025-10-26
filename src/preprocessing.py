#src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    # Try to auto-detect separator (; or ,)
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','

    # Load the dataset
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip()

    # Remove unnamed or empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(how='all', axis=1)

    # Check for target column name
    possible_targets = [c for c in df.columns if 'CO' in c and '(GT)' in c]
    if not possible_targets:
        raise ValueError("❌ Could not find 'CO(GT)' or similar target column.")
    target = possible_targets[0]

    # Replace invalid readings
    df = df.replace(-200, np.nan)

    # Drop rows missing target or too many NaNs
    df = df.dropna(subset=[target])
    df = df.dropna()

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Split into features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Ensure we have data
    if len(X) == 0:
        raise ValueError("❌ No valid rows left after cleaning. Check your CSV content.")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("✅ Data loaded and preprocessed successfully!")
    print(f"Separator used: '{sep}'")
    print(f"Target column: {target}")
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print("Sample feature names:", list(X.columns[:5]))

    return X, X_scaled, y, scaler

if __name__ == "__main__":
    X, X_scaled, y, scaler = load_and_preprocess_data("data/AirQualityUCI.csv")
