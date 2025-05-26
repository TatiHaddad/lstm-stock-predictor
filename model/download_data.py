import yfinance as yf
import os

def download_data(ticker="BBAS3.SA", filepath="../data/raw/BBAS3.SA.csv"):
    # Garante que o diretório existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Baixa os dados do Yahoo Finance
    df = yf.download(ticker, start="2018-01-01")

    # Reseta índice para colocar 'Date' como coluna
    df.reset_index(inplace=True)

    # Seleciona só as colunas que vamos usar
    df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

    # Salva em CSV
    df.to_csv(filepath, index=False)
    print(f"Dados baixados e salvos em: {filepath}")

if __name__ == "__main__":
    download_data()
