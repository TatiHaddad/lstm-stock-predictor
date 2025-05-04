### Define o schema da requisição e resposta  da API
from pydantic import BaseModel, Field  # Pydantic para validação de dados e Filed para definir campos
from typing import Optional # Optional para definir campos opcionais


### Usa o Pydantic para validar o corpo das requisições:
# - O modelo StockRequest define o corpo da requisição ( o que o usuário envia ), que deve conter um campo "symbol" (str).
# - O modelo StockResponse define o corpo da resposta (o que a API retorna), que deve conter um campo "predicted_price" (float).
# - O modelo StockResponse também pode conter um campo "error" (str) para mensagens de erro.

# Modelo de Entrada (Request) para requisição da previsão
class StockRequest(BaseModel):
    symbol: str = Field(..., title="Símbolo da Ação", description="Símbolo da ação para a qual será feita a previsão, como 'AAPL' ou 'GOOG'.")


    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",  # Exemplo de símbolo de ação
            }
        }


# Modelo de Saída (Response) para resposta da previsão
class StockResponse(BaseModel):
    predicted_price: float = Field(..., title="Preço Previsto", description="O preço previsto para a ação com base no modelo.")
    error: Optional[str] = Field(None, title="Mensagem de Erro", description="Mensagem de erro caso o modelo não consiga realizar a previsão.")

    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 150.75,  # Exemplo de preço previsto para a ação
                "error": None,  # Se não houver erro, o campo pode ser None
            }
        }
