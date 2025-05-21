import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import joblib
import os

# =============================
# Configurações
# =============================
ticker = "BBAS3.SA"
n_steps = 60

# =============================
# Função para carregar e limpar o CSV bagunçado
# =============================
def load_and_clean_csv(filepath):
    # Pula as 2 primeiras linhas
    df_raw = pd.read_csv(filepath, skiprows=2, header=None)

    # Define manualmente as colunas que queremos
    df_raw.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Remove linhas com NaN na coluna 'Date'
    df = df_raw.dropna(subset=['Date'])

    # Converte 'Date' para datetime com formato explícito
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

    # Remove datas mal formatadas
    df = df.dropna(subset=['Date'])

    # Define como índice
    df.set_index('Date', inplace=True)

    return df

# =============================
# Carrega e limpa os dados
# =============================
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", f"{ticker}.csv")
df = load_and_clean_csv(data_path)

# Seleciona a coluna Close e remove valores nulos
df = df[['Close']].dropna()

# =============================
# Normaliza os dados
# =============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# =============================
# Prepara sequência para LSTM
# =============================
X, y = [], []
for i in range(n_steps, len(scaled_data)):
    X.append(scaled_data[i - n_steps:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# =============================
# Modelo LSTM
# =============================
model = Sequential()
model.add(Input(shape=(X.shape[1], 1)))  # substitui input_shape=
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# =============================
# Salva modelo e scaler
# =============================
model_path = os.path.join(os.path.dirname(__file__), "model_lstm.keras")  # novo formato
model.save(model_path)

scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
joblib.dump(scaler, scaler_path)

print("Treinamento concluído e arquivos salvos.")
print(f"Modelo salvo em {model_path}")
print(f"Scaler salvo em {scaler_path}")
