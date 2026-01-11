import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

# --- Load models ---
risk_ml = joblib.load("models/risk_ml_pipeline.pkl")
risk_dl = tf.keras.models.load_model("models/risk_dl_model.h5")
risk_preprocessor = joblib.load("models/risk_preprocessor.pkl")

los_ml = joblib.load("models/los_ml_pipeline.pkl")
los_dl = tf.keras.models.load_model("models/los_dl_model.h5")
los_preprocessor = joblib.load("models/los_preprocessor.pkl")

# --- Streamlit UI ---
st.set_page_config(page_title="HealthAI Suite", layout="wide")
st.title("🏥 HealthAI Suite")
st.write("Interactive predictions using **ML** and **DL** models")

# Sidebar navigation
task = st.sidebar.radio("Choose Prediction Task", ["Patient Risk", "Length of Stay"])

# --- Patient Risk ---
if task == "Patient Risk":
    st.header("🔎 Patient Risk Prediction")

    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    insurance = st.selectbox("Health Insurance", ["Yes", "No"])
    state = st.text_input("State Name", "Tamil Nadu")
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=175)
    systolic_bp = st.number_input("Systolic_BP", min_value=80, max_value=200, value=120)

    if st.button("Predict Risk"):
        # ✅ Wrap dictionary in DataFrame
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Health_Insurance": [insurance],
            "State_Name": [state],
            "Cholesterol_Level": [chol],
            "Systolic_BP": [systolic_bp],
        })

        # ML prediction
        ml_pred = risk_ml.predict(input_df)[0]
        ml_proba = risk_ml.predict_proba(input_df)[0][1]

        # DL prediction
        X_dl = risk_preprocessor.transform(input_df).toarray()
        dl_proba = risk_dl.predict(X_dl).ravel()[0]
        dl_pred = int(dl_proba > 0.5)

        st.subheader("Results")
        col1, col2 = st.columns(2)
        col1.metric("ML Prediction", f"{ml_pred}", f"Prob: {ml_proba:.2f}")
        col2.metric("DL Prediction", f"{dl_pred}", f"Prob: {dl_proba:.2f}")

# --- Length of Stay ---
elif task == "Length of Stay":
    st.header("⏱️ Length of Stay Prediction")

    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    diagnosis = st.text_input("Diagnosis Code", "D123")
    severity = st.selectbox("Severity", ["Low", "Medium", "High"])
    prev_adm = st.number_input("Previous Admissions", min_value=0, max_value=20, value=1)

    # ✅ Add missing columns
    location = st.text_input("Location", "Chennai")
    beds = st.number_input("Hospital_Beds", min_value=10, max_value=1000, value=250)
    mri_units = st.number_input("MRI_Units", min_value=0, max_value=20, value=2)
    ct_scanners = st.number_input("CT_Scanners", min_value=0, max_value=20, value=3)
    time = st.number_input("Time", min_value=0, max_value=24, value=12)

    if st.button("Predict LOS"):
        input_df = pd.DataFrame({
            "Age": [age],
            "Diagnosis": [diagnosis],
            "Severity": [severity],
            "Prev_Admissions": [prev_adm],
            "Location": [location],
            "Hospital_Beds": [beds],
            "MRI_Units": [mri_units],
            "CT_Scanners": [ct_scanners],
            "Time": [time],
        })

        # ML prediction
        ml_pred = los_ml.predict(input_df)[0]

        # DL prediction
        X_dl = los_preprocessor.transform(input_df).toarray()
        dl_pred = los_dl.predict(X_dl).ravel()[0]

        st.subheader("Results")
        col1, col2 = st.columns(2)
        col1.metric("ML Prediction", f"{ml_pred:.2f} days")
        col2.metric("DL Prediction", f"{dl_pred:.2f} days")   
        
         # ML prediction
        ml_pred = los_ml.predict(input_df)[0]

        # DL prediction
        X_dl = los_preprocessor.transform(input_df).toarray()
        dl_pred = los_dl.predict(X_dl).ravel()[0]

        