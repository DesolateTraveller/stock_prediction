# data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# List of popular NSE stocks (Yahoo Finance uses .NS suffix)
NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "BAJFINANCE.NS", "ITC.NS",
    "HINDUNILVR.NS", "ASIANPAINT.NS", "MARUTI.NS", "WIPRO.NS", "AXISBANK.NS"
]

def fetch_nse_data():
    """
    Fetch today's intraday or latest daily data for NSE stocks.
    For intraday, you can use period="1d", interval="5m"
    For daily summary (close, high, low, etc.), use period="1d", interval="1d"
    """
    data_list = []

    end_date = datetime.today()
    start_date = end_date - timedelta(days=5)  # Fetch last 5 days for ML features

    for symbol in NSE_STOCKS:
        try:
            # Download last 5 days of daily data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, interval="1d")

            if len(hist) < 2:
                continue  # Skip if insufficient data

            # Use latest day's data
            latest = hist.iloc[-1]
            prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest['Close']

            open_price = float(latest['Open'])
            high = float(latest['High'])
            low = float(latest['Low'])
            close = float(latest['Close'])
            volume = int(latest['Volume'])
            change_pct = ((close - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0

            data_list.append({
                "Symbol": symbol,
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
                "Change %": round(change_pct, 2),
                "Prev Close": prev_close,
                "Date": latest.name.strftime("%Y-%m-%d")
            })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue

    df = pd.DataFrame(data_list)
    df = df.sort_values("Change %", ascending=False).reset_index(drop=True)
    return df

def fetch_stock_history(symbol, period="3mo", interval="1d"):
    """
    Fetch historical data for candlestick chart and ML training
    Supports: 5m, 15m, 1h, 1d, 1wk
    Note: For 5m/15m, max period is 60d (yfinance limit)
    """
    ticker = yf.Ticker(symbol)

    # Adjust period for intraday data
    if interval in ["5m", "15m", "1h"]:
        period = "60d"  # yfinance limit for intraday

    hist = ticker.history(period=period, interval=interval)
    hist.reset_index(inplace=True)
    hist['Date'] = pd.to_datetime(hist['Date'])
    return hist