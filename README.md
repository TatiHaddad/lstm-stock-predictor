# Modelo Preditivo de Preços de Ações com LSTM

## Descrição do Projeto

Este projeto faz parte do Tech Challenge da fase 4 da Pós Graduação de Engenharia de Machine Learning da FIAP.


Deep Learning e IA 
O desafio é criar um Modelo preditivo de Redes Neurais **Long Short Term Memory (LSTM)** para predizer o valor de fechamento da bolsa de valores de uma empresa à sua escolha e realizar toda a pipeline de desenvolvimento, 
desde a criação do modelo preditivo até o deploy do modelo em uma API que permita a previsão de preços de ações. 
 
 A previsão é realizada com base em dados históricos de preços de ações, coletados da plataforma **Yahoo Finance**. O modelo é então servido por uma API **FastAPI** que permite que os usuários forneçam dados históricos e recebam previsões dos preços futuros.



## Requisitos

Antes de rodar o projeto, certifique-se de ter os seguintes requisitos instalados:

- Python 3.x
- Docker (opcional, para contêinerização e deploy)


## Estrutura de Diretórios

stock_predictor_lstm/
├── app/
│   ├── main.py               # API FastAPI para previsões
│   ├── model.py              # Código para carregar e usar o modelo treinado
│   ├── preprocessing.py      # Funções de limpeza e transformação dos dados
│   ├── schema.py             # Modelos Pydantic (validação de entrada)
│
├── data/                     # Dados brutos ou pré-processados
│   └── raw/                  # CSVs originais baixados do Yahoo Finance
│
├── model/                    # Modelos salvos 
│   ├── train.py              # Treinamento do modelo LSTM
│   └── model_lstm.h5         # Modelo treinado (será gerado)
│   └── scaler.npy            # Parâmetros de normalização
├── notebooks/
│   └── exploracao.ipynb      # Notebook de EDA (exploração de dados)
│
├── requirements.txt          # Dependências do projeto
├── Dockerfile                # Dockerfile para deploy da API
├── .gitignore                # Arquivos ignorados pelo Git
├── .env                      # Variáveis de ambiente
└── README.md                 # Documentação do projeto



# Instalação
Clone este repositório:

git clone https://github.com/TatiHaddad/lstm-stock-predictor.git
cd lstm-stock-predictor


# Crie um ambiente virtual e instale as dependências:

python3 -m venv venv
source venv/bin/activate  # No Windows use venv\Scripts\activate
pip install -r requirements.txt



Como Rodar
# Treinamento do Modelo
Para treinar o modelo LSTM com os dados de ações, execute o seguinte comando:

python model/train.py

Essa etapa irá baixar os dados históricos de preços da ação APPLE (usando a biblioteca yfinance), pré-processá-los e treinar o modelo LSTM. Após o treinamento, o modelo será salvo como model/model_lstm.h5 e o scaler como model/scaler.npy.



# Iniciar a API
A API está desenvolvida com FastAPI e pode ser executada com o comando abaixo:

uvicorn app.main:app --reload
A API estará disponível em http://127.0.0.1:8000 e a documentação da API pode ser acessada em http://127.0.0.1:8000/docs e http://127.0.0.1:8000/redoc



# Fazer Previsões
A API possui um endpoint POST /predict/ onde você pode enviar uma solicitação para obter previsões de preços de ações. Exemplo de corpo de solicitação:

json
{
  "symbol": "AAPL",
  "lookback": 60,
  "start": "2020-01-01",
  "end": "2024-01-01",
  "incluir_grafico": true
}


# Docker
Se preferir, você pode executar o projeto dentro de um contêiner Docker. Para isso, use o comando:

docker build -t stock_predictor_lstm .
docker run -p 8000:8000 stock_predictor_lstm
A API será disponibilizada em http://127.0.0.1:8000.


# Dependências
As dependências estão listadas em requirements.txt. Para instalar:

pip install -r requirements.txt


# Modelos e Métricas
O modelo utiliza uma rede LSTM para capturar padrões temporais nos dados de preços das ações.

O desempenho do modelo é avaliado com as seguintes métricas:

* MAE (Erro Médio Absoluto)

* RMSE (Raiz do Erro Quadrático Médio)

* R² (Coeficiente de Determinação)

A avaliação do modelo pode ser feita diretamente na API, que retornará as métricas de desempenho juntamente com a previsão.



# Escalabilidade e Deploy
Este projeto está preparado para ser escalável com o uso de contêineres Docker, permitindo fácil deploy em ambientes de nuvem como AWS, Google Cloud, ou Azure. O deploy em nuvem permitirá o uso da API em produção, acessível via URL pública.
