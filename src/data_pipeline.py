import yfinance as yf
import pandas as pd

def fetch_binance_data(symbol='BTC-USD', interval='1d', start_date='2023-01-01', end_date='2023-03-01'):
    """
    Fetch historical Bitcoin data using yfinance.
    """
    df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    df = df[['Close', 'Volume']].droplevel(1, axis=1).rename(columns={'Close': 'close', 'Volume': 'volume'})
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'timestamp'}, inplace=True)
    return df

if __name__ == '__main__':
    df = fetch_binance_data()
    df.to_csv('data/bitcoin_data.csv')
    print("Data fetched and saved.")