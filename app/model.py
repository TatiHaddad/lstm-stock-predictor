
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model_lstm.h5")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "scaler.pkl")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_next_price(closing_prices: list) -> float:
    last_60 = np.array(closing_prices).reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60)
    X = np.reshape(last_60_scaled, (1, 60, 1))
    prediction_scaled = model.predict(X)[0][0]
    prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]
    return round(float(prediction), 2)

