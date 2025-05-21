from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    closing_prices: List[float] = Field(
        ..., 
        title="Últimos preços de fechamento", 
        description="Lista com os últimos 60 preços de fechamento. Deve conter pelo menos 60 valores."
    )

class PredictResponse(BaseModel):
    prediction: float = Field(
        ..., 
        title="Valor previsto", 
        description="Preço de fechamento previsto (desnormalizado)"
    )
