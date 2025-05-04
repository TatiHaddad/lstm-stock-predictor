### Define o schema da requisição e resposta  da API
from pydantic import BaseModel # Pydantic para validação de dados
from typing import Optional # Optional para definir campos opcionais


### Usa o Pydantic para validar o corpo das requisições:
# - O modelo StockRequest define o corpo da requisição ( o que o usuário envia ), que deve conter um campo "symbol" (str).
# - O modelo StockResponse define o corpo da resposta (o que a API retorna), que deve conter um campo "predicted_price" (float).
# - O modelo StockResponse também pode conter um campo "error" (str) para mensagens de erro.

# Modelo de Entrada (Request) para requisição da previsão
class StockRequest(BaseModel):
    symbol: str # O símbolo da ação para a qual será feita a previsão

    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",  # Exemplo de símbolo de ação
            }
        }


# Modelo de Saída (Response) para resposta da previsão
class StockResponse(BaseModel):
    predicted_price: float  # Preço previsto para a ação

    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 150.25,  # Exemplo de preço previsto
            }
        }
