import numpy as np
import pandas as pd
import datetime
import joblib
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import mlflow

# Configura diretório de modelo
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Carrega dados
data = pd.read_csv("data/raw/BBAS3.SA.csv")
prices = data["Close"].values.reshape(-1, 1)

# Preprocessamento
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

# Cria sequências de treino
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_prices)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Ajusta para (samples, timesteps, features)

# Modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento com MLflow
with mlflow.start_run():
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    # Versionamento por timestamp
    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{MODEL_DIR}/model_lstm_{version}.h5"
    scaler_path = f"{MODEL_DIR}/scaler_{version}.pkl"

    # Salva artefatos com versão
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # Salva também como "latest"
    model.save(f"{MODEL_DIR}/model_lstm_latest.h5")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_latest.pkl")

    # Log no MLflow
    mlflow.log_param("epochs", 50)
    mlflow.log_param("version", version)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(scaler_path)
    mlflow.log_artifact(f"{MODEL_DIR}/model_lstm_latest.h5")
    mlflow.log_artifact(f"{MODEL_DIR}/scaler_latest.pkl")

# Atualiza .env.dev com a versão treinada
with open(".env.dev", "w") as f:
    f.write(f"MODEL_VERSION=model_lstm_{version}\n")
    f.write(f"SCALER_VERSION=scaler_{version}\n")
