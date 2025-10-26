# 🌫️ Air Quality Prediction App  

![App Preview](https://github.com/swards05/AirQualityPredictionApp/assets/your-image-id/app-preview.png)  
*Smart Air Quality Prediction using Machine Learning & Streamlit*

---

## 🚀 Overview  

This project predicts **Carbon Monoxide (CO)** concentration in the air using real-world sensor data.  
By analyzing environmental and gas sensor readings, it estimates air quality levels and provides a simple **web-based interface** for real-time predictions.  

It is built using **Python, Streamlit**, and **machine learning (LSTM & Random Forest)** for intelligent analysis.

---

## ✨ Key Features  

✅ Predicts **CO(GT)** (mg/m³) from top air quality features  
✅ Displays **real-time prediction & interpretation** (Good / Moderate / Caution)  
✅ Stores recent user inputs & predictions in a **CSV file**  
✅ Allows **CSV download** directly from the app  
✅ Retrained model using **top 6 features** for optimized performance  
✅ Simple, fast, and elegant Streamlit UI  

---

## 🧠 ML Workflow  

1. **Data Preprocessing:**  
   - Cleaning, handling missing values, and scaling using `StandardScaler`.

2. **Feature Selection:**  
   - Top 6 important features chosen for best accuracy:
     ```
     C6H6(GT), PT08.S1(CO), PT08.S2(NMHC), NOx(GT), T, RH
     ```

3. **Model Training:**  
   - Both **LSTM** and **Random Forest** models trained.  
   - Evaluated on RMSE — best model retrained for deployment.

4. **Deployment:**  
   - Web app developed with **Streamlit** for real-time interaction.

---

## ⚙️ Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/swards05/AirQualityPredictionApp.git
cd AirQualityPredictionApp
```

## 2️⃣ Install dependencies
```
pip install -r requirements.txt
```
## 3️⃣ Run the Streamlit app
```
streamlit run app.py
```
## 📊 Example Input  

| Feature        | Example Value |
|----------------|----------------|
| C6H6(GT)       | 5.5            |
| PT08.S1(CO)    | 1100           |
| PT08.S2(NMHC)  | 900            |
| NOx(GT)        | 120            |
| T              | 18.5           |
| RH             | 45             |

## 🌍 Example Output
``` Predicted CO(GT): 4.32 mg/m³
✅ Air quality is Good — Safe CO levels.
```

## 💾 Data Logging

All prediction inputs and outputs are automatically stored in:
```
recent_predictions.csv
```
You can download it directly using the 📥 Download CSV button inside the app.

## 🧩 Tech Stack
Python 3.12+

TensorFlow / Keras

scikit-learn

pandas, numpy, matplotlib

Streamlit

## 🌟 Acknowledgements

Dataset: Air Quality UCI Dataset (UCI ML Repository)

Libraries: TensorFlow, scikit-learn, Streamlit
