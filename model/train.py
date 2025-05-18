import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import json
from evaluate_model import avaliar_modelo

def download_data(symbol='AAPL', start='2018-01-01', end='2024-01-01'):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Close']].dropna()
    return df

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

def train_and_save_model(symbol='AAPL'):
    df = download_data(symbol)
    X, y, scaler = prepare_data(df)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    y_pred = model.predict(X_test)
    avaliar_modelo(y_test, y_pred, scaler=scaler, salvar_grafico=True)

    os.makedirs("model", exist_ok=True)
    model.save("model/model_lstm.h5")

    # Salva o scaler completo
    np.save("model/scaler.npy", scaler, allow_pickle=True)

    # Reverte normalização
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred).flatten()

    df_results = pd.DataFrame({
        'Real': y_test_orig,
        'Previsto': y_pred_orig
    })
    df_results.to_csv("model/resultados.csv", index=False)

    resultados_json = df_results.to_dict(orient="records")
    with open("model/resultados.json", "w") as f:
        json.dump(resultados_json, f, indent=2)

if __name__ == "__main__":
    train_and_save_model()
