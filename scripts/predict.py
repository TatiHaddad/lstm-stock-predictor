import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys


# Adiciona o diretório 'app' ao path para importar o preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from preprocessing import preparar_novos_dados

# Caminhos
CAMINHO_MODELO = 'model/model_lstm.h5'
CAMINHO_SCALER = 'model/scaler.npy'
CAMINHO_DADOS = 'data/raw/novos_dados.csv'  

# Carregar novos dados
df = pd.read_csv(CAMINHO_DADOS)
print("Dados carregados:", df.shape)

# Pré-processamento (gera apenas X para predição)
X_novo, scaler = preparar_novos_dados(df, coluna_alvo='Close', n_passos=60, scaler_path=CAMINHO_SCALER)

# Carregar modelo treinado
modelo = load_model(CAMINHO_MODELO)
print("Modelo carregado com sucesso.")

# Fazer predição
y_pred = modelo.predict(X_novo)
y_pred_original = scaler.inverse_transform(y_pred)


# Exibir previsão
print("Previsão para o próximo valor de fechamento:")
print(y_pred_original.flatten()[0])

# Exibir gráfico da sequência recente com a previsão
precos_reais = df['Close'].dropna().values[-60:]  # últimos 60 reais
plt.figure(figsize=(12, 6))
plt.plot(range(60), precos_reais, label='Últimos 60 Preços Reais')
plt.plot(60, y_pred_original.flatten()[0], 'ro', label='Próxima Previsão')
plt.title('Histórico + Previsão')
plt.xlabel('Índice Temporal')
plt.ylabel('Preço de Fechamento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
