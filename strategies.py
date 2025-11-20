# strategies.py
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands  # <-- NEW IMPORT

def add_technical_indicators(df):
    """
    Add RSI, MACD, and Bollinger Bands to OHLC DataFrame.
    Assumes df has: 'Close', 'High', 'Low', 'Open', 'Volume'
    """
    if len(df) < 26:
        return df  # Need enough data for MACD & BB

    # ===== RSI (14-day) =====
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()

    # ===== MACD (12,26,9) =====
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # ===== Bollinger Bands (20,2) =====
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()

    # ===== Generate Signals =====
    # RSI Signal
    df['RSI_Signal'] = df['RSI'].apply(
        lambda x: 'OVERBOUGHT' if x > 70 else ('OVERSOLD' if x < 30 else 'NEUTRAL')
    )

    # MACD Crossover Signal
    df['MACD_Signal_Cross'] = 'HOLD'
    for i in range(1, len(df)):
        prev_macd = df['MACD'].iloc[i-1]
        curr_macd = df['MACD'].iloc[i]
        prev_signal = df['MACD_Signal'].iloc[i-1]
        curr_signal = df['MACD_Signal'].iloc[i]
        
        if prev_macd < prev_signal and curr_macd > curr_signal:
            df.loc[df.index[i], 'MACD_Signal_Cross'] = 'BUY'
        elif prev_macd > prev_signal and curr_macd < curr_signal:
            df.loc[df.index[i], 'MACD_Signal_Cross'] = 'SELL'

    # Bollinger Band Position
    df['BB_Position'] = 'MIDDLE'
    df.loc[df['Close'] > df['BB_High'], 'BB_Position'] = 'ABOVE_UPPER'
    df.loc[df['Close'] < df['BB_Low'], 'BB_Position'] = 'BELOW_LOWER'

    return df


def get_buy_recommendations(df, top_n=5):
    """Recommend stocks with strong positive momentum."""
    return df[df["Change %"] > 1.5].head(top_n)


def get_sell_recommendations(df, top_n=5):
    """Recommend stocks with strong negative momentum."""
    return df[df["Change %"] < -1.5].head(top_n)
