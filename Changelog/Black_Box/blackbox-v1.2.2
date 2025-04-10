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
loop_interval = 60  # 5 minutes

# OKX API credentials for real trading
API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_SECRET_KEY")
API_PASS = os.getenv("OKX_PASSPHRASE")

# Config for risk management
STOP_LOSS_PERCENTAGE = 0.02  # 2% loss
TAKE_PROFIT_PERCENTAGE = 0.05  # 5% profit

# Fetch historical OHLCV data from OKX
def fetch_data(timeframe='5m'):
    url = f'https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}'
    response = requests.get(url)
    data = response.json()['data']
    df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'vol_usd', 'confirm'])
    df = df[['ts', 'o', 'h', 'l', 'c', 'vol']].astype(float)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# Add indicators and generate features
def add_indicators(df):
    df['ma15'] = df['close'].rolling(window=15).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # Adding return feature after the indicators are added
    df['return'] = df['close'].pct_change()
    df = df.dropna()  # Drop any NaN values
    return df

# Generate features for ML model
def generate_features(df):
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

# Sentiment analysis function (simulated for now)
def sentiment_analysis():
    sentiment_score = np.random.uniform(-1, 1)  # -1: negative, 1: positive
    return sentiment_score

# Generate signal from model and indicators
def generate_signal(df, model):
    if df.empty:
        print("DataFrame is empty, cannot generate signal.")
        return 'HOLD'  # or return an appropriate signal, like 'HOLD'

    latest = df.iloc[-1:]
    
    # Check if the necessary columns exist before accessing them
    required_columns = ['ma15', 'ma21', 'macd', 'macd_signal', 'return']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing columns: {set(required_columns) - set(df.columns)}")
        return 'HOLD'

    features = latest[required_columns]
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

# Generate signal with sentiment analysis
def generate_signal_with_sentiment(df, model):
    if df.empty:
        return 'HOLD'

    sentiment_score = sentiment_analysis()
    signal = generate_signal(df, model)

    if sentiment_score > 0.5:  # Strong positive sentiment
        if signal == 'SELL':
            return 'HOLD'  # Don't sell during positive sentiment
    elif sentiment_score < -0.5:  # Strong negative sentiment
        if signal == 'BUY':
            return 'HOLD'  # Don't buy during negative sentiment
    return signal

# Logging function
def log_to_file(content):
    with open("trades_log.txt", "a") as f:
        f.write(content + "\n")

# Simulate trade execution with risk management
def simulate_trade_with_risk_management(signal, price):
    global position, entry_price
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    if signal == 'BUY' and position is None:
        position = 'LONG'
        entry_price = price
        stop_loss = entry_price * (1 - STOP_LOSS_PERCENTAGE)
        take_profit = entry_price * (1 + TAKE_PROFIT_PERCENTAGE)
        log = f"{timestamp} | BUY at {price:.2f} | Stop Loss: {stop_loss:.2f} | Take Profit: {take_profit:.2f}"
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
        
    # Check for Stop Loss or Take Profit condition
    if position == 'LONG':
        if price <= stop_loss:
            log = f"{timestamp} | STOP LOSS hit at {price:.2f}"
            print(f"[TRADE] {log}")
            log_to_file(log)
            position = None
            entry_price = 0
        elif price >= take_profit:
            log = f"{timestamp} | TAKE PROFIT hit at {price:.2f}"
            print(f"[TRADE] {log}")
            log_to_file(log)
            position = None
            entry_price = 0

# Fetch multiple timeframe data
def fetch_mtf_data():
    df_5m = fetch_data()  # 5-minute data
    df_1m = fetch_data('1m')  # 1-hour data
    return df_5m, df_1m

# Generate signal from multiple timeframes
def generate_mtf_signal(df_5m, df_1m, model):
    signal_5m = generate_signal_with_sentiment(df_5m, model)
    signal_1m = generate_signal_with_sentiment(df_1m, model)
    
    if signal_5m == 'BUY' and signal_1m == 'BUY':
        return 'BUY'
    elif signal_5m == 'SELL' and signal_1m == 'SELL':
        return 'SELL'
    else:
        return 'HOLD'

# Function to send buy/sell order to OKX
def send_order_to_okx(signal, price):
    if signal == 'BUY':
        order = {
            "instId": symbol,
            "tdMode": "cash",
            "side": "buy",
            "ordType": "market",
            "sz": 1  # Trade size
        }
    elif signal == 'SELL':
        order = {
            "instId": symbol,
            "tdMode": "cash",
            "side": "sell",
            "ordType": "market",
            "sz": 1  # Trade size
        }
    else:
        return

    url = "https://www.okx.com/api/v5/trade/order"
    headers = {
        'Content-Type': 'application/json',
        'OK-API-API-KEY': API_KEY,
        'OK-API-SECRET-KEY': API_SECRET,
        'OK-API-PASSPHRASE': API_PASS
    }

    response = requests.post(url, headers=headers, data=json.dumps(order))
    print(f"Order response: {response.json()}")

# Main trading logic
def main():
    df_5m, df_1m = fetch_mtf_data()
    df_5m = add_indicators(df_5m)
    df_1m = add_indicators(df_1m)
    X_5m, y_5m = generate_features(df_5m)
    
    if X_5m is None or y_5m is None:
        return

    model = train_model(X_5m, y_5m)
    
    signal = generate_mtf_signal(df_5m, df_1m, model)
    price = df_5m['close'].iloc[-1]
    simulate_trade_with_risk_management(signal, price)
    
    send_order_to_okx(signal, price)

if __name__ == "__main__":
    while True:
        main()
        time.sleep(loop_interval)
