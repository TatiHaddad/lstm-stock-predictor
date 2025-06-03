# Previsão de Preços de Ações BBAS3 com LSTM – API FastAPI com Deploy em Docker
Modelo Preditivo de Preços de Ações com LSTM

## Descrição do Projeto

Este projeto faz parte do Tech Challenge da fase 4 da Pós Graduação de Engenharia de Machine Learning da FIAP.

Deep Learning e IA 

O desafio é criar um Modelo preditivo de Redes Neurais **Long Short Term Memory (LSTM)** para predizer o valor de fechamento da bolsa de valores de uma empresa à sua escolha e realizar toda a pipeline de desenvolvimento, desde a criação do modelo preditivo até o deploy do modelo em uma API que permita a previsão de preços de ações. 
 
A previsão é realizada com base em dados históricos de preços de ações, coletados da plataforma **Yahoo Finance**. O modelo é então servido por uma API **FastAPI** que permite que os usuários forneçam dados históricos e recebam previsões dos preços futuros.


## Funcionalidades:
	- Treinamento de modelo LSTM com dados históricos de ações
	- Servidor FastAPI com endpoint para previsão
	- Pré-processamento e normalização automática dos dados
	- Container Docker pronto para produção


## Requisitos

Antes de rodar o projeto, certifique-se de ter os seguintes requisitos instalados:

	- Python 3.x
	- Docker (opcional, mas recomendado para conteinerização e deploy)


		stock_predictor_lstm/
		├── app/                            # Código principal da API e scripts auxiliares
		│   ├── __init__.py                 # Permite importar a pasta como módulo
		│   ├── exploracao_preprocess.ipynb # Notebook de pré-processamento e exploração
		│   ├── fetch_data.py               # Script para buscar dados de fontes externas
		│   ├── main.py                     # Arquivo principal da API (FastAPI)
		│   ├── model.py                    # Funções de inferência usando o modelo treinado
		│   ├── preprocessing.py            # Funções de transformação de dados (scaling etc.)
		│   ├── schema.py                   # Modelos Pydantic para validação de entrada/saída
		│   └── start.sh                    # Script de inicialização da API no Docker
		│
		├── data/                           # Armazenamento de dados brutos ou processados
		│   ├── raw/                        # Dados crus utilizados no projeto
		│   │   ├── BBAS3.SA.csv            # Base de dados principal de ações BBAS3
		│   │   └── novos_dados.csv         # Arquivo de dados novos para testes
		│   ├── download_data.py            # Script para baixar dados de forma programada
		│   └── teste.py                    # Script de testes locais com dados
		│
		├── model/                          # Treinamento e artefatos do modelo
		│   ├── __init__.py
		│   ├── download_data.py           # Script auxiliar para coleta de dados para treino
		│   ├── model_lstm.h5              # Arquivo com o modelo LSTM treinado
		│   ├── scaler.npy                 # Scaler salvo em formato NumPy
		│   ├── scaler.pkl                 # Scaler salvo com joblib (preferido para produção)
		│   ├── scaler.save                # Versão alternativa do scaler
		│   ├── resultados.csv             # Métricas e previsões geradas
		│   ├── resultados.json            # Métricas em formato JSON
		│   └── train.py                   # Script principal de treinamento do modelo
		│
		├── monitor/                       # Scripts de monitoramento do modelo em produção
		│   └── monitor.py                 # Verifica performance e saúde do modelo
		│
		├── notebooks/                     # Notebooks exploratórios
		│   └── exploracao.ipynb           # Análise exploratória dos dados
		│
		├── scripts/                       # Scripts diversos para testes e automações
		│   ├── predict.py                 # Testes manuais de previsão usando o modelo
		│   └── teste_api.py               # Script para testar os endpoints da API
		│
		├── .dockerignore                  # Ignora arquivos/pastas no contexto do Docker
		├── .env.dev                       # Variáveis de ambiente para ambiente de desenvolvimento
		├── .env.example                   # Exemplo de template de variáveis de ambiente
		├── .env.prod                      # Variáveis de ambiente para ambiente de produção
		├── .gitignore                     # Arquivos/pastas ignoradas pelo Git
		├── docker-compose.yml             # Docker Compose base (usado em dev)
		├── docker-compose.override.yml    # Override para o Compose local
		├── docker-compose.prod.yml        # Configuração de Docker Compose para produção
		├── Dockerfile                     # Dockerfile base para dev
		├── Dockerfile.prod                # Dockerfile otimizado para produção
		├── grafico_avaliacao.png          # Imagem com avaliação do modelo
		├── README.md                      # Documentação do projeto
		├── requirements.txt               # Lista de dependências do projeto
		└── run_all.py                     # Script para rodar toda a pipeline de forma sequencial



## Como rodar o projeto:
# Instalação
Clone este repositório:

	git clone https://github.com/TatiHaddad/lstm-stock-predictor.git
	cd lstm-stock-predictor



# Crie um ambiente virtual e instale as dependências:
	python3 -m venv venv
	source venv/bin/activate  # No Windows use venv\Scripts\activate
	pip install -r requirements.txt


## Como Rodar
Este projeto pode ser executado de três formas principais:

### 1. Ambiente de Desenvolvimento com Docker
Pré-requisitos:
	Docker e Docker Compose instalados
	Arquivo .env.dev configurado (ou copie de .env.example)


# Suba o ambiente em modo desenvolvimento
	docker-compose up --build
 
A API estará disponível em: http://localhost:8000
Swagger docs: http://localhost:8000/docs


### 2. Ambiente de Produção com Docker
Pré-requisitos:
	Docker instalado
	Arquivo .env.prod configurado


# Suba o ambiente de produção
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d

A API estará disponível em: http://<seu_ip>:8000


### 3. Rodar Localmente (sem Docker)
Pré-requisitos:
	Python 3.10+
	Ambiente virtual ativo

Instalar dependências:
	
	pip install -r requirements.txt


Executar API:
		
	uvicorn app.main:app --reload

  
Rodar Pipeline Completa (coleta, treino, predição)

# Executa todo o processo fim a fim
	python run_all.py

Testar a API com script

# Envia uma requisição POST para /predict
	python scripts/teste_api.py


Verificar se a API está online
Acesse no navegador ou via curl:
	curl http://localhost:8000/


# Treinamento do Modelo
Para treinar ou re-treinar o modelo LSTM com os dados históricos:

	python model/train.py

 Isso irá:
		Carregar os dados do Yahoo Finance (se necessário)
		Normalizar e preparar os dados
		Treinar o modelo LSTM
		Salvar os arquivos: model_lstm.h5 e scaler.pkl
*Essa etapa irá baixar os dados históricos de preços da ação BBAS3 (usando a biblioteca yfinance), pré-processá-los e treinar o modelo LSTM. 


# Monitorar a Performance do Modelo
Para executar a checagem e monitoramento de desempenho com novos dados:
	python monitor/monitor.py
	Esse script realiza:
		Avaliação da performance do modelo com dados recentes
		Comparação entre previsão e valor real
		Geração de métricas ou alertas (a depender da lógica implementada)



## API - Endpoints e Testes

### Endpoint principal

# Iniciar a API
A API está desenvolvida com FastAPI e pode ser executada com o comando abaixo:

	uvicorn app.main:app --reload

A API estará disponível em http://127.0.0.1:8000 e a documentação da API pode ser acessada em http://127.0.0.1:8000/docs e http://127.0.0.1:8000/redoc

Como testar a API
Você pode usar o Swagger UI para fazer chamadas ou usar o curl/Postman com o seguinte JSON:

**POST /predict**

# Fazer Previsões
A API possui um endpoint POST /predict/ onde você pode enviar uma solicitação para obter previsões de preços de ações. Exemplo de corpo de solicitação:

**Exemplo de payload**:
JSON

	{
	  "closing_prices": [12.4, 12.6, 10.5, 13.5, 12.4, 12.4, 12.4,  10.5,  10.5,  10.5,  10.5, 10.5, 10.5, 12.6, 12.6 , 12.6, 12.6, 12.6, 12.6, 13.5 , 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 12.4, 12.6, 10.5, 13.5, 12.4, 12.4, 12.6, 10.5, 13.5, 12.4, 12.4, 12.6, 10.5, 13.5, 12.4, 12.4, 12.4,  10.5,  10.5,  10.5,  10.5, 10.5, 10.5, 12.6, 12.6 , 12.6, 12.6, 12.6]
	}
 

A resposta será:
	
	{
	  "prediction": 12.57
	}


# Fazer Previsões
A API possui um endpoint POST /predict/ onde você pode enviar uma solicitação para obter previsões de preços de ações. Exemplo de corpo de solicitação:



## Como rodar com o Docker
# Docker
Se preferir, você pode executar o projeto dentro de um contêiner Docker. Para isso, use o comando:
## Builda da Imagem:
	
	docker build -t stock_predictor_lstm .

## Rode o container
	
	docker run -p 8000:8000 stock_predictor_lstm

A API será disponibilizada em http://127.0.0.1:8000.

# Endpoints
Método: POST
Endpoint: /predict
Descrição: Faz a previsão do próximo valor com base na sequência de entrada

Exemplo de payload:

	{
	  "closing_prices": [135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4, 135.6, 136.2, 137.8, 138.9, 139.4] 
	}



# Dependências
As dependências estão listadas em requirements.txt. Para instalar:
	
	pip install -r requirements.txt



## Modelo e Decisões Técnicas
Usamos LSTM (Long Short-Term Memory) por sua eficácia em séries temporais financeiras.

A entrada da rede é composta por janelas deslizantes de 60 dias de preços de fechamento normalizados.

Normalização com MinMaxScaler para que os dados fiquem na faixa [0, 1].

Arquitetura da rede:

LSTM (50 unidades) → LSTM (50 unidades) → Dense(1)

Função de perda: mean_squared_error, Otimizador: adam

Salvo com TensorFlow .h5 e joblib (.pkl) para reutilização.

---

###  Explicação do modelo LSTM
- **Motivação**: O problema trata de uma série temporal financeira. LSTMs são mais adequadas para capturar dependências de longo prazo do que redes feedforward simples.
  
- **Arquitetura escolhida**:
  - 2 camadas LSTM com 50 unidades cada
  - 1 camada final `Dense(1)` para prever o próximo valor
  - Epochs: 10 (ajustável), batch_size: 32

- **Entradas do modelo**:
  - 60 valores consecutivos de preços de fechamento normalizados.
  - Normalização feita com `MinMaxScaler`.

- **Saída**:
  - Um único valor contínuo: **o próximo preço de fechamento previsto** (já desnormalizado para facilitar interpretação).

- **Salvamento**:
  - `.h5`: modelo treinado (TensorFlow)
  - `.pkl`: scaler usado para normalizar dados, necessário para preparar futuras entradas

---

# Modelos e Métricas
O modelo utiliza uma rede LSTM para capturar padrões temporais nos dados de preços das ações.

O desempenho do modelo é avaliado com as seguintes métricas:

* MAE (Erro Médio Absoluto)

* RMSE (Raiz do Erro Quadrático Médio)

* R² (Coeficiente de Determinação)

A avaliação do modelo pode ser feita diretamente na API, que retornará as métricas de desempenho juntamente com a previsão.


## Autores

- Tatiana M. Haddad – [@TatiHaddad](https://github.com/TatiHaddad)
- Victor Santos 
