import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import os
import pandas as pd

# -------------------------------
# ⚙️ Paths
# -------------------------------
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "final_scaler.pkl")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "final_best_model.keras")
CSV_PATH = "recent_predictions.csv"

# -------------------------------
# 🧠 Load model and scaler
# -------------------------------
st.sidebar.title("⚙️ Model Configuration")

try:
    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    st.sidebar.success("✅ Model and scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# 🌫️ App Header
# -------------------------------
st.title("🌫️ Air Quality — Predict CO(GT) Concentration")
st.markdown("Enter air sensor readings below to predict **CO concentration (mg/m³)**.")

# -------------------------------
# 📊 Input features (Top 6)
# -------------------------------
top_features = [
    "C6H6(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "T",
    "RH"
]

st.subheader("📥 Enter Sensor Readings")
cols = st.columns(3)
inputs = []

for i, f in enumerate(top_features):
    with cols[i % 3]:
        val = st.number_input(f, step=0.1, format="%.2f")
        inputs.append(val)

# Convert and scale
input_array = np.array([inputs])
scaled_input = scaler.transform(input_array)

# -------------------------------
# 🔮 Prediction
# -------------------------------
if st.button("🔍 Predict CO(GT)"):
    scaled_input = np.array([inputs])
    scaled_input = scaler.transform(scaled_input)

    # Handle LSTM input shape correctly
    if len(model.input_shape) == 3:
        scaled_input = np.expand_dims(scaled_input, axis=2)  # (1, 6, 1)

    pred = model.predict(scaled_input)
    co_value = float(pred[0][0])

    st.subheader(f"Predicted CO(GT): **{co_value:.2f} mg/m³**")

    # Interpretation
    if co_value < 5:
        quality = "Good"
        msg = "✅ Air quality is **Good** — Safe CO levels."
        st.success(msg)
    elif co_value < 10:
        quality = "Moderate"
        msg = "⚠️ Moderate CO levels — Sensitive groups should be cautious."
        st.warning(msg)
    else:
        quality = "Dangerous"
        msg = "☠️ Dangerous CO levels — Harmful to human health!"
        st.error(msg)

    #st.subheader(f"Predicted CO(GT): **{co_value:.2f} mg/m³**")

    # -------------------------------
    # 💾 Save to CSV
    # -------------------------------
    entry = {f: inputs[i] for i, f in enumerate(top_features)}
    entry["Predicted CO(GT)"] = co_value
    entry["Air Quality"] = quality

    # Append or create CSV
    df_entry = pd.DataFrame([entry])
    if os.path.exists(CSV_PATH):
        df_entry.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df_entry.to_csv(CSV_PATH, index=False)

    #st.success("📝 Entry saved to recent predictions!")

# -------------------------------
# 📄 Show and download saved entries
# -------------------------------
if os.path.exists(CSV_PATH):
    st.markdown("---")
    st.subheader("🕒 Recent Predictions")

    df = pd.read_csv(CSV_PATH)
    st.dataframe(df.tail(10), use_container_width=True)

    # Download button
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download All Predictions (CSV)",
        data=csv_data,
        file_name="recent_predictions.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("🔁 Automatically uses the best performing model (MLP or LSTM) trained on top 6 features.")


