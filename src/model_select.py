# src/model_select.py

import os
import shutil
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import load_and_preprocess_data
from feature_engineering import split_data

# ✅ Import model builder functions
from model_mlp import create_mlp_model
from model_lstm import create_lstm_model

# =======================
# STEP 1: Compare Models
# =======================
if __name__ == "__main__":
    X, X_scaled, y, scaler = load_and_preprocess_data("data/AirQualityUCI.csv")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Load trained models
    lstm_loaded = load_model("models/lstm_model.keras")
    mlp_loaded = load_model("models/mlp_model.keras")

    # Evaluate
    y_pred_lstm = lstm_loaded.predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).flatten()
    y_pred_mlp = mlp_loaded.predict(X_test).flatten()

    mse_lstm = mean_squared_error(y_test, y_pred_lstm)
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)

    print(f"LSTM MSE: {mse_lstm:.4f}")
    print(f"MLP  MSE: {mse_mlp:.4f}")

    # Select best model
    best_model_name = "LSTM" if mse_lstm < mse_mlp else "MLP"
    print(f"🏆 Best model: {best_model_name}")

    best_model_path = f"models/{'lstm_model.keras' if best_model_name == 'LSTM' else 'mlp_model.keras'}"
    shutil.copy(best_model_path, "models/best_model.keras")
    print("💾 Saved best model as models/best_model.keras")

    with open("models/metrics.txt", "w") as f:
        f.write(f"LSTM MSE: {mse_lstm:.4f}\nMLP MSE: {mse_mlp:.4f}\nBest: {best_model_name}")

    # =====================================
    # STEP 2: Retrain best model on TOP 6 FEATURES
    # =====================================
    top_features = ["C6H6(GT)", "PT08.S1(CO)", "PT08.S2(NMHC)",
                    "NOx(GT)", "T", "RH"]  # ✅ Use 'T' instead of 'Temperature (°C)' as in your dataset

    print("\n🔁 Retraining best model on top 6 features...")

    available_features = [f for f in top_features if f in X.columns]
    missing_features = [f for f in top_features if f not in X.columns]
    if missing_features:
        print(f"⚠️ Warning: These features were not found and will be skipped: {missing_features}")

    X_top = X[available_features]

    # New scaler for selected features
    scaler_top = StandardScaler()
    X_top_scaled = scaler_top.fit_transform(X_top)

    # Split again
    X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
        X_top_scaled, y, test_size=0.2, random_state=42
    )

    # Retrain from scratch
    if best_model_name == "LSTM":
        X_train_top = np.expand_dims(X_train_top, axis=2)
        X_test_top = np.expand_dims(X_test_top, axis=2)
        model = create_lstm_model(input_shape=(X_train_top.shape[1], 1))
    else:
        model = create_mlp_model(input_dim=X_train_top.shape[1])

    model.fit(X_train_top, y_train_top, epochs=30, batch_size=16, verbose=0)
    mse = model.evaluate(X_test_top, y_test_top, verbose=0)
    print(f"🔁 Retrained {best_model_name} on top features MSE: {mse:.4f}")

    # Save final retrained model
    model.save("models/final_best_model.keras")
    joblib.dump(scaler_top, "models/final_scaler.pkl")
    print("💾 Saved retrained model as models/final_best_model.keras and scaler as final_scaler.pkl")
