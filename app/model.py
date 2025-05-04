### Predição com o modelo salvo: Carrega o modelo salvo e faz a previsão

import os
import numpy as np
from tensorflow.keras.models import load_model  # Keras - load_model para carregar o modelo salvo
from tensorflow.keras.models import Sequential  # Keras - Sequential para construir o modelo de rede Neural LSTM    
from sklearn.preprocessing import MinMaxScaler  # MinMaxScaler para normalizar os dados
import yfinance as yf   

### Carrega bibliotecas para inferência, incluindo o modelo salvo.


MODEL_PATH = "model/model_lstm.h5"
SCALER_PATH = "model/scaler.npy"


def load_scaler():
    
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler não encontrado em: {SCALER_PATH}")
    

    data_max = np.load(SCALER_PATH)
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = 0, 1 / data_max
    return scaler

### Reconstrói o MinMaxScaler com os parâmetros salvos (máximo dos dados), para manter a mesma 
# escala usada no treinamento.



###Carrega o modelo treinado e o scaler.
#
# O modelo é um modelo LSTM treinado para prever preços de ações com base em dados históricos.
# O scaler é usado para normalizar os dados de entrada antes de fazer previsões e para reverter a normalização após a previsão.


def predict_next_price(symbol: str, lookback: int = 60):
    try:
        # Verifica se o modelo existe
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
        
        
        # Carrega o modelo e o scaler apenas quando a função for chamada
        model = load_model(MODEL_PATH)
        scaler = load_scaler()

        # Baixa os dados
        df = yf.download(symbol, period=f'{lookback + 1}d')
        if df.empty or len(df) < lookback + 1:
            raise ValueError(f"Dados insuficientes para o símbolo '{symbol}'")
        

        close_prices = df['Close'].values.reshape(-1, 1)

### Baixa os preços recentes para fazer previsão.
# lookback+1 garante dados suficientes.
        #Normaliza os dados de entrada
        scaled_input = scaler.transform(close_prices)
        last_sequence = scaled_input[-lookback:]
        X = np.reshape(last_sequence, (1, lookback, 1))


        # Faz a predição
        prediction = model.predict(X)
        predicted_price = scaler.inverse_transform(prediction)



        return float(predicted_price[0][0])

    except Exception as e:
        return {"erro": str(e)}

### Prepara a entrada no formato esperado.
# Faz a predição e retorna o valor desnormalizado (preço real).

