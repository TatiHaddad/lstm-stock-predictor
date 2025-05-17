### Importação das bibliotecas necessárias
import yfinance as yf # yfinance para baixar os dados do Yahoo Finance
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # MinMaxScaler para normalizar os dados
from tensorflow.keras.models import Sequential, load_model # Keras - Sequential para construir o modelo de rede Neural LSTM e load_model
from tensorflow.keras.layers import LSTM, Dense # Keras - LSTM e Dense para construir o modelo de rede Neural LSTM
import os # Para salvar e manipular os diretórios e salvar os arquivos (o modelo treinado )
import json
from evaluate_model import avaliar_modelo



### Função para baixar os dados históricos de ações do Yahoo Finance
def download_data(symbol='AAPL', start='2018-01-01', end='2024-01-01'):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Close']].dropna()
    return df

### Baixa os dados históricos da ação, no caso selecionada a Apple,
#- retorando apenas a coluna de fechamento (close)
#- e removendo os valores nulos (dropna)


### Constrói o dataset para o modelo LSTM
def prepare_data(df, lookback=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

### Na construção do dataset, o modelo LSTM espera uma entrada tridimensional (samples, time steps, features)
# e com a opção de lookback, o modelo LSTM pode prever o preço de fechamento da ação
# com base nos preços de fechamento dos últimos 60 dias (lookback=60)
# - O MinMaxScaler é usado para normalizar os dados entre 0 e 1, o que é importante para o desempenho do LSTM.
# - O loop for percorre os dados escalonados e cria as sequências de entrada (X) e os rótulos (y).   
# - A entrada X é então remodelada para ter a forma (n_samples, n_time_steps, n_features), onde:
# - n_samples é o número de amostras, 
# - n_time_steps é o número de passos de tempo (lookback) e 
# - n_features é o número de recursos (1 neste caso, pois estamos usando apenas o preço de fechamento).



### Cria o Modelo LSTM
def train_and_save_model(symbol='AAPL'):
    df = download_data(symbol)
    X, y, scaler = prepare_data(df)

    #Separando treino e teste (80% treino, 20% teste)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]


    # Cria o modelo LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

### Cria o modelo LSTM com 2 camadas LSTM e uma camada densa final para prever o próximo valor.
# return_sequences=True é necessário para empilhar LSTMs.
#    O modelo LSTM é construído usando a API Sequential do Keras.
# - A primeira camada LSTM tem 50 unidades e retorna sequências para a próxima camada LSTM.
# - A segunda camada LSTM também tem 50 unidades, mas não retorna sequências, pois é a última camada LSTM.
#
# - A camada de saída é uma camada densa com uma unidade, que prevê o preço de fechamento da ação.
# - O modelo é compilado usando o otimizador Adam e a função de perda de erro quadrático médio  (mean squared error).
# - O modelo é treinado por 10 épocas com um tamanho de lote de 32.
#
# - Após o treinamento, o modelo é salvo em um diretório chamado "model" e o scaler é salvo como um arquivo numpy.
# - O diretório "model" é criado se não existir, usando os métodos os.makedirs e exist_ok=True.
# 
# - O scaler é salvo como um arquivo numpy para que possa ser carregado posteriormente para normalizar os dados 
# de entrada para previsões futuras.
# - O scaler é usado para reverter a normalização dos dados previstos, transformando os valores previstos de 
# volta para o intervalo original.

    model.compile(optimizer='adam', loss='mean_squared_error')

    #Treinamento
    model.fit(X_train, y_train, epochs=10, batch_size=32)


### Compila o modelo com otimizador Adam e erro quadrático médio.
# Treina com 10 épocas e batch size de 32.

    # Faz previsões
    y_pred = model.predict(X_test)

    # Avaliação com gráfico
    avaliar_modelo(y_test, y_pred, scaler=scaler, salvar_grafico=True)


    #Cria pasta se não existir 
    os.makedirs("model", exist_ok=True)
    
    # Salva o modelo treinado como .h5 
    model.save("model/model_lstm.h5")

    #SAlva o scaler como .npye
    np.save("model/scaler.npy", scaler, allow_pickle=True)


    # Reverter normalização para salvar resultados reais
    y_test_orig = y_test * scaler.data_max_[0]
    y_pred_orig = y_pred.flatten() * scaler.data_max_[0]

    # CSV com valores reais e previstos
    df_results = pd.DataFrame({
        'Real': y_test_orig,
        'Previsto': y_pred_orig
    })
    df_results.to_csv("model/resultados.csv", index=False)

    # Exportar JSON para API usar no Swagger
    resultados_json = df_results.to_dict(orient="records")
    with open("model/resultados.json", "w") as f:
        json.dump(resultados_json, f, indent=2)


### Salva o modelo treinado e o "máximo dos dados" do scaler para usar depois na inferência.


if __name__ == "__main__":
    train_and_save_model()
