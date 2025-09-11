# strategies.py
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

def add_technical_indicators(df):
    """
    Add RSI and MACD columns to OHLC DataFrame.
    Assumes df has: 'Close', 'High', 'Low', 'Open', 'Volume'
    """
    if len(df) < 26:
        return df  # MACD needs at least 26 periods

    # RSI (14-day standard)
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()

    # MACD (12,26,9 standard)
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # Generate Signals
    df['RSI_Signal'] = df['RSI'].apply(
        lambda x: 'OVERBOUGHT' if x > 70 else ('OVERSOLD' if x < 30 else 'NEUTRAL')
    )

    df['MACD_Signal_Cross'] = 'HOLD'
    for i in range(1, len(df)):
        if df['MACD'].iloc[i-1] < df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
            df.loc[df.index[i], 'MACD_Signal_Cross'] = 'BUY'
        elif df['MACD'].iloc[i-1] > df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]:
            df.loc[df.index[i], 'MACD_Signal_Cross'] = 'SELL'

    return df

def get_buy_recommendations(df, top_n=5):
    return df[df["Change %"] > 1.5].head(top_n)

def get_sell_recommendations(df, top_n=5):
    return df[df["Change %"] < -1.5].head(top_n)