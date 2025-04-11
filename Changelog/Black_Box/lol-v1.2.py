import pandas as pd
import numpy as np
import requests
import time
import xgboost as xgb
from ta.trend import MACD
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import warnings
import hmac
import hashlib
import base64
import json
import datetime
import os
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Global trading state
position = None
entry_price = 0
stop_loss = 0
take_profit = 0

# Configurations
BASE_URL = "https://www.okx.com"
symbol = 'BTC-USDT-SWAP'
SYMBOL = 'BTC-USDT-SWAP'
timeframe = '5m'
LEVERAGE = 15
ORDER_SIZE = 10
limit = 200
loop_interval = 30  # in seconds

# API Credentials (set these as environment variables)
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

STOP_LOSS_PERCENTAGE = 0.02
TAKE_PROFIT_PERCENTAGE = 0.05

# ------------------- API Utility Functions -------------------

def get_server_time():
    """
    Retrieve the OKX server time for API signature.
    """
    endpoint = "/api/v5/public/time"
    response = requests.get(BASE_URL + endpoint)
    response.raise_for_status()
    return str(float(response.json()["data"][0]["ts"]) / 1000.0)

def generate_signature(timestamp, method, request_path, body=""):
    """
    Generate an HMAC SHA256 signature for API authentication.
    """
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def send_request(method, endpoint, body=None):
    """
    Send an authenticated request to the OKX API.
    """
    try:
        timestamp = get_server_time()
        body_json = json.dumps(body) if body else ""
        headers = {
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": generate_signature(timestamp, method, endpoint, body_json),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/json",
            "x-simulated-trading": "1"  # Demo trading mode
        }
        url = BASE_URL + endpoint
        response = requests.request(method, url, headers=headers, data=body_json)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {"error": str(e)}

def set_leverage(leverage=LEVERAGE):
    """
    Set the account leverage.
    """
    endpoint = "/api/v5/account/set-leverage"
    body = {
        "instId": SYMBOL,
        "lever": str(leverage),
        "mgnMode": "cross"
    }
    response = send_request("POST", endpoint, body)
    logging.info(f"Leverage Response: {response}")
    return response

def get_realtime_price():
    """
    Get the current ticker price.
    """
    endpoint = f"/api/v5/market/ticker?instId={SYMBOL}"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        return float(response["data"][0]["last"])
    logging.warning("Failed to retrieve real-time price.")
    return None

def get_prices():
    """
    Get one-minute candlestick price data.
    """
    endpoint = f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit={limit}"
    response = send_request("GET", endpoint)
    if "data" in response:
        return [float(candle[4]) for candle in response["data"]]
    logging.warning("Failed to retrieve historical prices.")
    return []

def place_order(side, pos_side, order_size=ORDER_SIZE):
    """
    Place a market order.
    """
    order_data = {
        "instId": SYMBOL,
        "tdMode": "cross",
        "side": side,
        "posSide": pos_side,
        "ordType": "market",
        "sz": str(order_size * LEVERAGE)
    }
    response = send_request("POST", "/api/v5/trade/order", body=order_data)
    logging.info(f"Order Response: {response}")
    return response

def check_open_positions():
    """
    Check if there are any open positions.
    """
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        return True
    return False

def close_all_positions():
    """
    Close all open positions.
    """
    if check_open_positions():
        logging.info("Closing all positions...")
        place_order("buy", "short")   # If short positions exist, close them
        place_order("sell", "long")   # If long positions exist, close them
    else:
        logging.info("No open positions to close.")

# ------------------- Data & Feature Engineering -------------------

def fetch_data(timeframe='5m'):
    """
    Fetch historical candlestick data.
    """
    url = f'https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}'
    response = requests.get(url)
    data = response.json()['data']
    df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'vol_usd', 'confirm'])
    df = df[['ts', 'o', 'h', 'l', 'c', 'vol']].astype(float)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.sort_values('timestamp').reset_index(drop=True)

def add_indicators(df):
    """
    Compute technical indicators.
    """
    df['ma15'] = df['close'].rolling(window=15).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['return'] = df['close'].pct_change()
    return df.dropna()

def generate_features(df):
    """
    Generate features and target for the ML model.
    """
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna()
    if df.empty:
        return None, None
    X = df[['ma15', 'ma21', 'macd', 'macd_signal', 'return']]
    y = df['target']
    return X, y

def train_model(X, y):
    """
    Train an XGBoost classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# ------------------- Forecasting & Sentiment -------------------

def arima_prediction(df):
    try:
        model = ARIMA(df['close'], order=(2, 1, 2))
        model_fit = model.fit()
        return model_fit.forecast()[0]
    except:
        return df['close'].iloc[-1]

def real_sentiment():
    """
    Get a simple sentiment score from CoinGecko trending search.
    """
    try:
        url = 'https://api.coingecko.com/api/v3/search/trending'
        response = requests.get(url).json()
        btc_trending = any('bitcoin' in coin['item']['id'] for coin in response['coins'])
        return 0.8 if btc_trending else -0.8
    except Exception as e:
        logging.error(f"Real sentiment error: {e}")
        return 0

# ------------------- Trading Signal & Order Execution -------------------

def generate_signal(df, model):
    """
    Generate a trade signal using technical indicators, ML, ARIMA, and sentiment.
    """
    latest = df.iloc[-1:]
    X_input = latest[['ma15', 'ma21', 'macd', 'macd_signal', 'return']]
    prediction = model.predict(X_input)[0]
    ma_cross = latest['ma15'].values[0] > latest['ma21'].values[0]
    macd_cross = latest['macd'].values[0] > latest['macd_signal'].values[0]
    arima_pred = arima_prediction(df)
    current_price = latest['close'].values[0]
    sentiment_score = real_sentiment()

    logging.info(f"[ARIMA] {arima_pred:.2f}, Price: {current_price:.2f}, Sentiment: {sentiment_score:.2f}")

    if ma_cross and macd_cross and prediction == 1 and arima_pred > current_price and sentiment_score > 0:
        return 'BUY'
    elif not ma_cross and not macd_cross and prediction == 0 and arima_pred < current_price and sentiment_score < 0:
        return 'SELL'
    return 'HOLD'

def execute_trade(signal, current_price):
    """
    Execute a real trade (in demo mode) using OKX API.
    This function places market orders based on the provided signal,
    updates global trade state, and monitors for stop-loss/take-profit hits.
    """
    global position, entry_price, stop_loss, take_profit

    set_leverage(LEVERAGE)  # Set leverage on each cycle

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if signal == 'BUY' and position != "long":
        close_all_positions()
        logging.info("Placing REAL market order: OPEN LONG position")
        response = place_order("buy", "long", ORDER_SIZE)
        position = "long"
        entry_price = current_price
        stop_loss = entry_price * (1 - STOP_LOSS_PERCENTAGE)
        take_profit = entry_price * (1 + TAKE_PROFIT_PERCENTAGE)
        logging.info(f"Long order executed: Entry: {entry_price:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
    elif signal == 'SELL' and position != "short":
        close_all_positions()
        logging.info("Placing REAL market order: OPEN SHORT position")
        response = place_order("sell", "short", ORDER_SIZE)
        position = "short"
        entry_price = current_price
        stop_loss = entry_price * (1 + STOP_LOSS_PERCENTAGE)
        take_profit = entry_price * (1 - TAKE_PROFIT_PERCENTAGE)
        logging.info(f"Short order executed: Entry: {entry_price:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
    else:
        logging.info("No change in position; monitoring the existing order.")

    # Monitor the current position for stop loss or take profit conditions:
    if position:
        price = get_realtime_price()
        if price is None:
            logging.warning("Cannot retrieve real-time price for monitoring.")
            return
        if position == "long":
            if price <= stop_loss:
                logging.info(f"{now} | LONG STOP LOSS hit at {price:.2f} -- closing position")
                close_all_positions()
                position = None
            elif price >= take_profit:
                logging.info(f"{now} | LONG TAKE PROFIT hit at {price:.2f} -- closing position")
                close_all_positions()
                position = None
        elif position == "short":
            if price >= stop_loss:
                logging.info(f"{now} | SHORT STOP LOSS hit at {price:.2f} -- closing position")
                close_all_positions()
                position = None
            elif price <= take_profit:
                logging.info(f"{now} | SHORT TAKE PROFIT hit at {price:.2f} -- closing position")
                close_all_positions()
                position = None

# ------------------- Main Trading Loop -------------------

def main():
    df = fetch_data()
    df = add_indicators(df)
    X, y = generate_features(df)
    if X is None:
        logging.error("Insufficient data for feature generation.")
        return
    model = train_model(X, y)
    signal = generate_signal(df, model)
    current_price = df['close'].iloc[-1]
    execute_trade(signal, current_price)

if __name__ == "__main__":
    while True:
        main()
        time.sleep(loop_interval)
