import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df: pd.DataFrame, look_back: int = 60):
    """
    Pré-processa os dados de fechamento de ações:
    - Remove valores nulos
    - Normaliza os dados com MinMaxScaler
    - Cria sequências de entrada (X) e saída (y) com base no parâmetro look_back
    - Retorna X, y, scaler e dados normalizados
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna 'Close'
        look_back (int): Número de passos de tempo para olhar para trás
    
    Retorna:
        X (np.array): Sequências de entrada
        y (np.array): Valores alvo
        scaler (MinMaxScaler): Scaler usado para normalizar os dados
        df_scaled (np.array): Série de preços normalizada
    """
    df = df[['Close']].dropna()

    # Escalar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(look_back, len(df_scaled)):
        X.append(df_scaled[i - look_back:i, 0])
        y.append(df_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)

    # Adiciona uma dimensão extra para LSTM [amostras, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler, df_scaled
