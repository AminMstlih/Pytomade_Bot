import pandas as pd
import numpy as np
import requests
import time
import xgboost as xgb
from ta.trend import MACD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
import hmac
import hashlib
import base64
import json
import datetime
import os

warnings.filterwarnings("ignore")

# Global simulation state
position = None
entry_price = 0

# Config
symbol = 'BTC-USDT'
timeframe = '5m'
limit = 200
loop_interval = 300  # 5 minutes

# OKX API credentials for real trading
API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_API_SECRET")
API_PASS = os.getenv("OKX_API_PASS")

# Fetch historical OHLCV data from OKX
def fetch_data():
    url = f'https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}'
    response = requests.get(url)
    data = response.json()['data']
    df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'vol_usd', 'confirm'])
    df = df[['ts', 'o', 'h', 'l', 'c', 'vol']].astype(float)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# Add indicators
def add_indicators(df):
    df['ma15'] = df['close'].rolling(window=15).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df

# Generate features for ML model
def generate_features(df):
    df['return'] = df['close'].pct_change()
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna()
    if df.empty:
        return None, None
    X = df[['ma15', 'ma21', 'macd', 'macd_signal', 'return']]
    y = df['target']
    return X, y

# Train simple XGBoost model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"[Model] Accuracy: {accuracy_score(y_test, preds):.2f}")
    return model

# ARIMA prediction
def arima_prediction(df):
    series = df['close']
    try:
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast.values[0]
    except Exception as e:
        print(f"[ARIMA] Forecasting failed: {e}")
        return series.iloc[-1]  # fallback: return last known price

# Generate signal from model and indicators
def generate_signal(df, model):
    latest = df.iloc[-1:]
    features = latest[['ma15', 'ma21', 'macd', 'macd_signal', 'return']]
    pred = model.predict(features)[0]
    ma_cross = latest['ma15'].values[0] > latest['ma21'].values[0]
    macd_cross = latest['macd'].values[0] > latest['macd_signal'].values[0]
    arima_forecast = arima_prediction(df)
    price_now = latest['close'].values[0]

    print(f"[ARIMA] Forecast: {arima_forecast:.2f}, Current Price: {price_now:.2f}")

    if ma_cross and macd_cross and pred == 1 and arima_forecast > price_now:
        return 'BUY'
    elif not ma_cross and not macd_cross and pred == 0 and arima_forecast < price_now:
        return 'SELL'
    else:
        return 'HOLD'

# Logging function
def log_to_file(content):
    with open("trades_log.txt", "a") as f:
        f.write(content + "\n")

# Simulate trade execution
def simulate_trade(signal, price):
    global position, entry_price
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    if signal == 'BUY' and position is None:
        position = 'LONG'
        entry_price = price
        log = f"{timestamp} | BUY at {price:.2f}"
        print(f"[TRADE] {log}")
        log_to_file(log)
    elif signal == 'SELL' and position == 'LONG':
        pnl = price - entry_price
        log = f"{timestamp} | SELL at {price:.2f} | PnL: {pnl:.2f} USDT"
        print(f"[TRADE] {log}")
        log_to_file(log)
        position = None
        entry_price = 0
    else:
        print(f"[TRADE] No action. Current Position: {position}")

# Backtesting
def backtest():
    print("\n[BACKTEST] Running backtest...")
    df = fetch_data()
    df = add_indicators(df)
    X, y = generate_features(df)
    if X is None or y is None:
        print("[WARNING] Not enough data to generate features.")
        return

    model = train_model(X, y)
    balance = 1000  # initial capital
    in_position = False
    entry_price = 0

    for i in range(50, len(df)):
        sample = df.iloc[:i].copy()
        signal = generate_signal(sample, model)
        price = sample['close'].iloc[-1]

        if signal == 'BUY' and not in_position:
            in_position = True
            entry_price = price
            print(f"[BT] BUY at {price:.2f}")
        elif signal == 'SELL' and in_position:
            pnl = price - entry_price
            balance += pnl
            print(f"[BT] SELL at {price:.2f} | PnL: {pnl:.2f} | Balance: {balance:.2f}")
            in_position = False

    print(f"[BT] Final Balance: {balance:.2f}")

if __name__ == "__main__":
    backtest()
