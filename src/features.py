import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    """
    Add engineered features to the dataframe.
    """
    # Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_3d'] = df['close'].pct_change(3)
    df['return_7d'] = df['close'].pct_change(7)
    df['return_30d'] = df['close'].pct_change(30)
    
    # EMA
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volatility
    df['volatility_30d'] = df['close'].rolling(30).std()
    
    # Volume change
    df['volume_change_1d'] = df['volume'].pct_change(1)
    
    # Target for regression
    df['next_return'] = df['close'].pct_change(1).shift(-1)
    
    # Target for classification
    df['direction'] = (df['next_return'] > 0).astype(int)
    
    # Drop NaN
    df.dropna(inplace=True)
    
    return df

if __name__ == '__main__':
    df = pd.read_csv('data/bitcoin_data.csv', index_col='timestamp', parse_dates=True)
    df_featured = add_features(df)
    df_featured.to_csv('data/bitcoin_featured.csv')
    print("Features added and saved.")