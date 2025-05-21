from fastapi import FastAPI, HTTPException, Body
from app.schema import PredictRequest, PredictResponse
from app.model import predict_next_price
import logging

app = FastAPI(
    title="Stock Predictor LSTM API",
    description="API para previsão de preço de fechamento de ações usando LSTM",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)

@app.get("/")
def read_root():
    return {"message": "API de Previsão de Ações está no ar!"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest = Body(..., example={"closing_prices": [10.5] * 60})):
    logging.info("Recebida requisição de previsão.")

    if len(request.closing_prices) < 60:
        logging.warning("Requisição com menos de 60 preços.")
        raise HTTPException(status_code=400, detail="Envie ao menos 60 preços de fechamento.")

    try:
        prediction = predict_next_price(request.closing_prices)
        logging.info(f"Previsão realizada com sucesso: {prediction}")
        return PredictResponse(prediction=prediction)
    except Exception as e:
        logging.error(f"Erro na previsão: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")
