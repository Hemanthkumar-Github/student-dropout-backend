import joblib
import numpy as np

model = joblib.load("dropout_model.pkl")

def predict_dropout(features):
    prob = model.predict_proba([features])[0][1]
    prediction = "High Risk" if prob > 0.5 else "Low Risk"
    return prediction, round(prob*100, 2)


import joblib
import os

MODEL_PATH = "dropout_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run train.py first.")

model = joblib.load(MODEL_PATH)
