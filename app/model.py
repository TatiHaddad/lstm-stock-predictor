import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Diretório base do projeto (nível raiz)
base_dir = os.path.dirname(__file__)
model_dir = os.path.abspath(os.path.join(base_dir, "..", "model"))

# Recupera valores do .env ou usa fallback para arquivos *_latest
model_filename = os.getenv("MODEL_PATH", os.path.join("model", "model_lstm_latest.h5"))
scaler_filename = os.getenv("SCALER_PATH", os.path.join("model", "scaler_latest.pkl"))

# Caminhos absolutos
model_path = os.path.abspath(os.path.join(base_dir, "..", model_filename))
scaler_path = os.path.abspath(os.path.join(base_dir, "..", scaler_filename))

# Carrega modelo e scaler
try:
    model = load_model(model_path)
except Exception as e:
    raise FileNotFoundError(f"Erro ao carregar o modelo: {model_path}. Detalhes: {str(e)}")

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    raise FileNotFoundError(f"Erro ao carregar o scaler: {scaler_path}. Detalhes: {str(e)}")

def predict_next_price(prices: list) -> float:
    try:
        # Normaliza os preços
        prices_array = np.array(prices).reshape(-1, 1)
        scaled = scaler.transform(prices_array)

        # Prepara a entrada para o modelo
        X = np.reshape(scaled, (1, len(prices), 1))

        # Faz a predição
        prediction_scaled = model.predict(X)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        return round(float(prediction), 2)

    except Exception as e:
        raise RuntimeError(f"Erro ao prever: {str(e)}")
