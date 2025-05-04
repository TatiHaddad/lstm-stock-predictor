from fastapi import FastAPI, Query
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from typing import Optional
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
from fastapi import HTTPException

#Inicializa o app FastAPI e importa o modelo e schemas
app = FastAPI(
    title="Modelo preditivo de ações com LSTM", #Título da API
    description="API para prever preços de ações usando um modelo LSTM treinado com dados históricos.",  # Descrição da API
    version="1.0.0",  # Versão da API
    docs_url="/docs",  # URL para acessar o Swagger UI
    redoc_url="/redoc",  # URL para acessar o ReDoc
)

# Carrega o modelo e scaler
model = load_model("model/model_lstm.h5")
data_max = np.load("model/scaler.npy")  # Carregando data_max_ salvo
scaler = MinMaxScaler()

#Reconstrução do scaler
scaler.min_ = 0  # O valor mínimo é 0
scaler.scale_ = 1 / data_max  # O valor de scale_ é o inverso do valor carregado

# Modelo de entrada (opcional)
class PredictRequest(BaseModel):
    symbol: str = "AAPL"
    lookback: int = 60
    start: Optional[str] = "2023-01-01"
    end: Optional[str] = "2024-01-01"
    incluir_grafico: Optional[bool] = True

# Função auxiliar de avaliação
def avaliar_modelo(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"rmse": rmse, "mae": mae, "mape": mape}

def gerar_grafico(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Real')
    plt.plot(y_pred, label='Previsto')
    plt.legend()
    plt.title("Preço Real vs Previsto")
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# Endpoint para previsão de preços
@app.post("/predict/")
def predict(request: PredictRequest):
    
    try:
        # 1. Download dados
        df = yf.download(request.symbol, start=request.start, end=request.end)[['Close']].dropna()

        if df.empty:
            raise HTTPException(status_code=400, detail=f"Nenhum dado encontrado para o símbolo '{request.symbol}' no período informado.")

        if len(df) <= request.lookback:
            raise HTTPException(
                status_code=400,
                detail=f"Não há dados suficientes para a janela de lookback={request.lookback}. O dataset retornou apenas {len(df)} registros."
            )

        # 2. Prepara os dados
        scaled_data = scaler.transform(df)  # Use 'transform' em vez de 'fit_transform' aqui
        X, y = [], []
        for i in range(request.lookback, len(scaled_data)):
            X.append(scaled_data[i - request.lookback:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Adicionando a dimensão de features (1)
        y = np.array(y)


        # 3. Faz predição
        y_pred = model.predict(X)
        y_pred = y_pred.reshape(-1, 1)

        # 4. Reverte a escala
        y_true_original = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        y_pred_original = scaler.inverse_transform(y_pred).flatten()

        # 5. Avalia
        metrics = avaliar_modelo(y_true_original, y_pred_original)

        # 6. Retorno com ou sem gráfico
        result = {
            "symbol": request.symbol,
            "metrics": metrics
        }

        if request.incluir_grafico:
            graph_b64 = gerar_grafico(y_true_original, y_pred_original)
            result["grafico_base64"] = graph_b64

        return result

    except HTTPException as e:
        raise e  # já estruturada corretamente

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erro inesperado: {str(e)}"}
        )
