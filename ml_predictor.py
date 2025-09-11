# ml_predictor.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def train_and_predict_next_close(df_hist):
    """
    Simple Linear Regression to predict next day's Close
    Features: Open, High, Low, Volume
    Target: Close
    """
    if len(df_hist) < 10:
        return None, "Insufficient data"

    # Prepare features and target
    df = df_hist.copy()
    df['Next_Close'] = df['Close'].shift(-1)
    df = df.dropna()

    if len(df) < 5:
        return None, "Not enough valid rows after shift"

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Next_Close']

    # Train model on all but last day
    model = LinearRegression()
    model.fit(X[:-1], y[:-1])

    # Predict next close (for last day's features)
    last_features = X.iloc[[-1]]
    next_close_pred = model.predict(last_features)[0]

    current_close = df.iloc[-1]['Close']
    expected_change_pct = ((next_close_pred - current_close) / current_close) * 100

    return {
        "Predicted_Next_Close": round(next_close_pred, 2),
        "Current_Close": round(current_close, 2),
        "Expected_Change_%": round(expected_change_pct, 2),
        "Signal": "BUY" if expected_change_pct > 1.5 else "SELL" if expected_change_pct < -1.5 else "HOLD"
    }, None