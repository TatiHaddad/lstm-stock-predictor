import yfinance as yf
import os

# Define o ticker e o período de coleta
ticker = "BBAS3.SA"
start_date = "2018-01-01"
end_date = "2024-12-31"

# Cria diretório se não existir
output_dir = os.path.join("data", "raw")
os.makedirs(output_dir, exist_ok=True)

# Caminho de saída do arquivo CSV
output_path = os.path.join(output_dir, f"{ticker}.csv")

# Faz o download dos dados com yfinance
print(f"Baixando dados de {ticker}...")
df = yf.download(ticker, start=start_date, end=end_date)

# Verifica se retornou dados
if df.empty:
    raise ValueError("Nenhum dado retornado. Verifique o ticker ou a conexão.")

# Salva o CSV
df.to_csv(output_path)
print(f"Dados salvos em: {output_path}")
