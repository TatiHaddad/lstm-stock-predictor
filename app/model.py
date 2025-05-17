import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler

# Caminhos dos arquivos salvos
MODEL_PATH = "model/model_lstm.h5"
SCALER_PATH = "model/scaler.npy"

def load_scaler():
    """
    Carrega o MinMaxScaler salvo como dicionário (com .npy + .item()).
    """
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler não encontrado em: {SCALER_PATH}")
    
    # Carregando o scaler salvo com np.save (dicionário)
    scaler_dict = np.load(SCALER_PATH, allow_pickle=True).item()

    scaler = MinMaxScaler()
    scaler.min_ = scaler_dict['min_']
    scaler.scale_ = scaler_dict['scale_']
    scaler.data_min_ = scaler_dict.get('data_min_', None)
    scaler.data_max_ = scaler_dict.get('data_max_', None)
    scaler.data_range_ = scaler_dict.get('data_range_', None)
    scaler.feature_range = (0, 1)
    
    return scaler

def predict_price(input_sequence):
    """
    Faz a predição com base na sequência histórica de preços normalizada.
    """
    if len(input_sequence) < 1:
        raise ValueError("A entrada precisa conter ao menos 1 valor.")

    model = load_model(MODEL_PATH)
    scaler = load_scaler()

    # Formata a sequência de entrada
    input_data = np.array(input_sequence).reshape(-1, 1)
    scaled_data = scaler.transform(input_data)
    X = scaled_data.reshape(1, scaled_data.shape[0], 1)

    # Predição
    prediction = model.predict(X)
    predicted_price = scaler.inverse_transform(prediction)

    return float(predicted_price[0][0])
