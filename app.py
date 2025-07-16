import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("heart_attack_model.pkl", "rb") as f:
    model = pickle.load(f)

# App UI
st.title("❤️ Heart Attack Prediction App")
st.markdown("Enter patient data to predict the risk of heart attack.")

# Form input fields
age = st.number_input("Age", min_value=0, max_value=120, value=45)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
thalachh = st.number_input("Maximum Heart Rate Achieved (thalachh)", min_value=0, value=150)
chol = st.number_input("Cholesterol (chol)", min_value=0, value=200)
oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, value=1.0, step=0.1)
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=50, max_value=250, value=120)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])

# Convert input to DataFrame
input_data = pd.DataFrame([{
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": cp,
    "thalachh": thalachh,
    "chol": chol,
    "oldpeak": oldpeak,
    "ca": ca,
    "thal": thal,
    "restecg": restecg,
    "fbs": fbs,
    "systolic_bp": systolic_bp,
    "slope": slope
}])

# Predict
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Attack (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Attack (Probability: {prob:.2f})")
