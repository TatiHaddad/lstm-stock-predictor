# monitor/monitor.py

import time
import logging
from datetime import datetime
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Configura logger
logging.basicConfig(filename="monitor/log.txt", level=logging.INFO)

def monitor_prediction():
    # Simula dados de entrada (últimos 60 preços fictícios ou reais)
    sample_input = {
        "closes": [31.22, 31.50, 31.88, 32.10, 32.00, 32.25] * 10  # total 60 valores
    }

    start_time = time.time()
    response = client.post("/predict", json=sample_input)
    elapsed = time.time() - start_time

    if response.status_code == 200:
        pred = response.json()["next_close_price"]
        log_msg = f"{datetime.now()} | Tempo: {elapsed:.4f}s | Previsão: R${pred:.2f}"
    else:
        log_msg = f"{datetime.now()} | ERRO {response.status_code}"

    logging.info(log_msg)
    print(log_msg)

if __name__ == "__main__":
    monitor_prediction()
