import os
import joblib
import numpy as np
import pandas as pd
import logging

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- PATH SETUP ----------------
# utils.py is inside Webapp/
# Models folder is one level above
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "Models", "stroke_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Models", "scaler.pkl")

# ---------------- LOAD MODEL & SCALER ----------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("✅ Model and scaler loaded successfully.")

except Exception as e:
    logging.error(f"❌ Error loading model or scaler: {e}")
    model, scaler = None, None


# ---------------- PREPROCESS INPUT ----------------
def preprocess_input(data):
    """
    Preprocess input data with correct feature order and scaling
    Feature order MUST match training exactly
    """
    try:
        if scaler is None:
            raise ValueError("❌ Scaler not loaded!")

        feature_names = [
            'gender',
            'age',
            'hypertension',
            'heart_disease',
            'ever_married',
            'work_type',
            'Residence_type',
            'avg_glucose_level',
            'bmi',
            'smoking_status'
        ]

        # Convert input list to DataFrame
        data_df = pd.DataFrame([data], columns=feature_names)

        # Apply same scaler used during training
        scaled_data = scaler.transform(data_df)

        return scaled_data

    except Exception as e:
        logging.error(f"❌ Error in preprocessing: {e}")
        raise


# ---------------- PREDICTION ----------------
def predict_stroke(input_data):
    """
    Predict stroke risk with probability and reasons
    """
    try:
        if model is None:
            raise ValueError("❌ Model not loaded!")

        processed_data = preprocess_input(input_data)

        # Probability for stroke (class 1)
        proba = model.predict_proba(processed_data)[0][1]
        probability = round(proba * 100, 2)

        prediction = 1 if proba >= 0.3 else 0

        # ---------- REASON EXTRACTION (EXPLAINABLE & SAFE) ----------
        feature_names = [
            'gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type',
            'avg_glucose_level', 'bmi', 'smoking_status'
        ]

        raw = dict(zip(feature_names, input_data))
        reasons = []

        if raw['age'] > 60:
            reasons.append("Advanced age (> 60 years)")
        if raw['hypertension'] == 1:
            reasons.append("History of hypertension")
        if raw['heart_disease'] == 1:
            reasons.append("Existing heart disease")
        if raw['avg_glucose_level'] > 200:
            reasons.append("High average glucose level")
        if raw['bmi'] > 30:
            reasons.append("High BMI (obesity)")
        if raw['smoking_status'] == 3:
            reasons.append("Active smoking habit")

        if not reasons:
            reasons.append("No major clinical risk factors detected")

        logging.info(f"✅ Probability: {probability}%")
        logging.info(f"✅ Reasons: {reasons}")

        return prediction, probability, reasons

    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return -1, 0.0, []
