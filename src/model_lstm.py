# src/model_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from preprocessing import load_and_preprocess_data
from feature_engineering import split_data

def reshape_for_lstm(X):
    # reshape (samples, timesteps, features)
    return X.reshape((X.shape[0], 1, X.shape[1]))

def create_lstm_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import tensorflow as tf
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model


if __name__ == "__main__":
    X, X_scaled, y, scaler = load_and_preprocess_data("data/AirQualityUCI.csv")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    X_train = reshape_for_lstm(X_train)
    X_test = reshape_for_lstm(X_test)

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        validation_split=0.2, callbacks=[es], verbose=1)

    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"📊 LSTM MSE: {mse:.4f}")
    print(f"📈 LSTM R²: {r2:.4f}")


    import os
    os.makedirs("models", exist_ok=True)

    model.save("models/lstm_model.keras")
    joblib.dump(scaler, "models/scaler.pkl")
    print("💾 LSTM model saved to models/lstm_model.keras")

