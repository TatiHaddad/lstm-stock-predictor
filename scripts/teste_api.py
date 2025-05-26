from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_prediction_request_schema():
    response = client.post("/predict", json={
        "symbol": "PETR4.SA",
        "start_date": "2022-01-01",
        "end_date": "2023-01-01",
        "window_size": 60
    })
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "prediction_dates" in data
    assert "predicted_prices" in data
    assert isinstance(data["predicted_prices"], list)
