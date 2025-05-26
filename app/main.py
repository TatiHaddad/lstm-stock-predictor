import os
import time
import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from app.schema import PredictRequest, PredictResponse
from app.model import predict_next_price

# Prometheus Instrumentation
from prometheus_fastapi_instrumentator import Instrumentator

# Carrega variáveis de ambiente do .env
load_dotenv()

# Inicializa o app
app = FastAPI(
    title="Stock Predictor LSTM API",
    description="API para previsão de preço de fechamento de ações usando LSTM",
    version="1.0.0"
)

# Instrumenta e expõe métricas Prometheus
Instrumentator().instrument(app).expose(app)

# Configura logging com base na variável de ambiente
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    log_level = "INFO"

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info(f"API inicializada em ambiente: {os.getenv('ENVIRONMENT', 'dev')}")

@app.get("/", tags=["Health Check"])
def read_root():
    """Verifica se a API está online"""
    try:
        environment = os.getenv('ENVIRONMENT', 'dev')
        return {"message": f"API de Previsão de Ações está no ar! Ambiente: {environment}"}
    except Exception as e:
        logger.exception("Erro no endpoint de health check")
        raise HTTPException(status_code=500, detail="Erro interno")


@app.get("/predict", tags=["Instruções"])
def predict_get_info():
    return {
        "message": "Use o método POST neste endpoint para enviar os preços e obter uma previsão.",
        "exemplo": {
            "closing_prices": [100.0, 101.5, 102.3, "... mínimo 60 valores"]
        }
    }


@app.post("/predict", response_model=PredictResponse, tags=["Previsão"])
def predict(request: PredictRequest):
    start_time = time.time()
    logger.info("Requisição recebida para previsão.")

    if len(request.closing_prices) < 60:
        logger.warning("Menos de 60 preços enviados.")
        raise HTTPException(status_code=400, detail="Envie ao menos 60 preços de fechamento.")

    try:
        prediction = predict_next_price(request.closing_prices)
        duration = time.time() - start_time
        logger.info(f"Previsão gerada em {duration:.4f} segundos: {prediction}")
        return PredictResponse(prediction=prediction)
    except Exception as e:
        logger.exception("Erro ao realizar previsão")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a previsão.")
