# src/feature_engineering.py
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits features and target into train/test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print("✅ Data split successful!")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from preprocessing import load_and_preprocess_data
    X, X_scaled, y, scaler = load_and_preprocess_data("data/AirQualityUCI.csv")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
