from fastapi import FastAPI, HTTPException
from app.schema import PredictRequest, PredictResponse
from app.model import predict_next_price
import logging
import os

from dotenv import load_dotenv
load_dotenv()

# Detecta o ambiente atual
ENV = os.getenv("ENVIRONMENT", "dev").lower()

# Configura o nível de log baseado no ambiente
log_level = logging.DEBUG if ENV == "dev" else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Iniciando API no ambiente: {ENV}")

# Inicialização do app FastAPI
app = FastAPI(
    title="Stock Predictor LSTM API",
    description="API para previsão de preço de fechamento de ações usando LSTM",
    version="1.0.0"
)

@app.get("/", tags=["Health Check"])
def read_root():
    """Verifica se a API está online"""
    return {"message": "API de Previsão de Ações está no ar!"}

@app.post("/predict", response_model=PredictResponse, tags=["Previsão"])
def predict(request: PredictRequest):
    """Recebe os últimos 60 preços de fechamento e retorna a previsão do próximo valor"""
    logging.info("Requisição recebida para previsão.")

    if len(request.closing_prices) < 60:
        logging.warning("Menos de 60 preços enviados.")
        raise HTTPException(status_code=400, detail="Envie ao menos 60 preços de fechamento.")

    try:
        prediction = predict_next_price(request.closing_prices)
        logging.info(f"Previsão gerada: {prediction}")
        return PredictResponse(prediction=prediction)
    except Exception as e:
        logging.exception("Erro ao realizar previsão")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a previsão.")
