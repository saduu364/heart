import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("🫀 Heart Attack Risk Prediction App")
st.markdown("Enter your medical details to assess the risk of a heart attack.")

# Input form
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=90, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=60, max_value=130, value=80)
chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
exang = st.selectbox("Exercise Induced Angina", [0, 1])
thalachh = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia Type", [1, 2, 3])

# Create feature vector in same order
input_data = np.array([[age, sex, systolic_bp, diastolic_bp, chol, fbs, cp, exang,
                        thalachh, oldpeak, slope, restecg, ca, thal]])

# Prediction
if st.button("Predict Heart Attack Risk"):
    # Scale input
    input_scaled = scaler.transform(input_data)
    probabilities = model.predict_proba(input_scaled)[0]  # probabilities for [No Heart Attack, Heart Attack]
    prediction = model.predict(input_scaled)[0]
    
    # Show full probabilities for debugging
    st.write(f"Prediction Probabilities: No Heart Attack = {probabilities[0]:.2f}, Heart Attack = {probabilities[1]:.2f}")

    # Output
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack\n🔢 Probability: {probabilities[1]:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Attack\n🔢 Probability: {probabilities[1]:.2f}")
