import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# Caminhos absolutos relativos à pasta atual
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "..", "model", "model_lstm.h5")
scaler_path = os.path.join(base_dir, "..", "model", "scaler.pkl")

# Carrega modelo e scaler na inicialização
model = load_model(model_path)
scaler = joblib.load(scaler_path)

def predict_next_price(prices: list) -> float:
    try:
        # Normaliza os preços
        prices_array = np.array(prices).reshape(-1, 1)
        scaled = scaler.transform(prices_array)

        # Prepara a entrada
        X = np.reshape(scaled, (1, len(prices), 1))

        # Faz a predição
        prediction_scaled = model.predict(X)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        return round(float(prediction), 2)
    except Exception as e:
        raise RuntimeError(f"Erro ao prever: {str(e)}")
