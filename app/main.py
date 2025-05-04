### Cria a API com FastAPI
from fastapi import FastAPI # FastAPI para criar a API
from fastapi.responses import JSONResponse

from app.schema import StockRequest, StockResponse  # Importa os modelos de requisição e resposta
from app.model import predict_next_price # Importa a função de predição do modelo treinado


#Inicializa o app FastAPI e importa o modelo e schemas
app = FastAPI(
    title="Modelo preditivo de ações com LSTM", #Título da API
    description="API para prever preços de ações usando um modelo LSTM treinado com dados históricos.",  # Descrição da API
    version="1.0.0",  # Versão da API
    docs_url="/docs",  # URL para acessar o Swagger UI
    redoc_url="/redoc",  # URL para acessar o ReDoc
)




@app.post("/predict", response_model=StockResponse, responses={400: {"description": "Erro de predição"}})
def get_prediction(data: StockRequest):
    result = predict_next_price(data.symbol)

    # Verifica se houve erro na predição (retorno como dict com chave de "erro")
    if isinstance(result, dict) and "erro" in result:
        return JSONResponse(status_code=400, content={"erro": result["erro"]})

    # Retorno normal com o preço previsto
    return StockResponse(predicted_price=result)


### Cria o endpoint /predict que recebe uma requisição POST com o símbolo da ação e retorna o preço previsto.
# # - O endpoint é definido com o decorador @app.post("/predict").
# # - A função get_prediction recebe um objeto data do tipo StockRequest (definido no schema) 
        #  e retorna um objeto do tipo StockResponse (também definido no schema).        
# # - A função chama a função predict_next_price com o símbolo da ação e retorna o preço previsto.
# # - Se ocorrer um erro durante a predição, retorna uma mensagem de erro.  
#

# ### O FastAPI cuida automaticamente da serialização e validação dos dados de entrada e saída, 
# tornando o código mais limpo e fácil de entender.   

# # - O FastAPI também gera automaticamente a documentação da API, que pode ser acessada em /docs ou /redoc.
# # - A documentação é gerada com base nos modelos Pydantic usados para validar os dados de entrada e saída.

# # - O FastAPI é uma excelente escolha para criar APIs RESTful de forma rápida e eficiente, com suporte a 
#   validação de dados e documentação automática.
# # - O FastAPI é baseado no Starlette e no Pydantic, o que o torna rápido e fácil de usar.
