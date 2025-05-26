# data/download_data.py
import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start, end, path="data/raw/data.csv"):
    df = yf.download(symbol, start=start, end=end)
    df.to_csv(path)
    return df

if __name__ == "__main__":
    get_stock_data("DIS", "2018-01-01", "2024-07-20")
