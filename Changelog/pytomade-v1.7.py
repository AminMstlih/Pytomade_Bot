# last night 30/03/2025
# Import necessary libraries
import time
import hmac
import hashlib
import base64
import requests
import json
import os
from dotenv import load_dotenv
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input  # Import Input here
from keras.models import Model

# Load API credentials from .env file
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Configuration variables
BASE_URL = "https://www.okx.com"
SYMBOL = "BTC-USDT-SWAP"
LEVERAGE = 10
ORDER_SIZE = 1
CHECK_INTERVAL = 60
DATA_LIMIT = 100

# Logging setup
logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Telegram message function
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logging.info(f"Telegram Message Sent: {message}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send Telegram message: {e}")

# OKX API functions
def get_server_time():
    endpoint = "/api/v5/public/time"
    try:
        response = requests.get(BASE_URL + endpoint)
        response.raise_for_status()
        return str(float(response.json()["data"][0]["ts"]) / 1000.0)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to get server time: {e}")
        return None

def generate_signature(timestamp, method, request_path, body=""):
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def send_request(method, endpoint, body=None):
    try:
        timestamp = get_server_time()
        if timestamp is None:
            return {"error": "Failed to get server time"}
        body_json = json.dumps(body) if body else ""
        headers = {
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": generate_signature(timestamp, method, endpoint, body_json),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/json",
            "x-simulated-trading": "1"  # Simulated trading mode
        }
        url = BASE_URL + endpoint
        response = requests.request(method, url, headers=headers, data=body_json)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {"error": str(e)}

# Account management
def set_leverage(leverage=LEVERAGE):
    endpoint = "/api/v5/account/set-leverage"
    body = {
        "instId": SYMBOL,
        "lever": str(leverage),
        "mgnMode": "cross"
    }
    response = send_request("POST", endpoint, body)
    logging.info(f"Leverage Response: {response}")
    return response

# Data fetching functions
def get_realtime_price():
    endpoint = f"/api/v5/market/ticker?instId={SYMBOL}"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        return float(response["data"][0]["last"])
    logging.warning("Failed to get real-time price.")
    return None

def get_candlestick_data():
    endpoint = f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit={DATA_LIMIT}"
    response = send_request("GET", endpoint)
    if "data" in response:
        candles = response["data"]
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "confirm", "oi"])
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')  # Fixed FutureWarning
        df['close'] = df['close'].astype(float)
        return df
    logging.warning("Failed to get candlestick data.")
    return pd.DataFrame()

# Trading functions
def place_order(side, pos_side, order_size=ORDER_SIZE):
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
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        return True
    return False

def close_all_positions():
    if check_open_positions():
        logging.info("Closing all positions...")
        place_order("buy", "short")
        place_order("sell", "long")
    else:
        logging.info("No open positions to close.")

# Position and account monitoring
def get_open_positions():
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        positions = []
        for position in response["data"]:
            symbol = position.get("instId")
            side = position.get("posSide")
            size = float(position.get("pos", 0))
            pnl = float(position.get("upl", 0))
            positions.append({
                "symbol": symbol,
                "side": side,
                "size": size,
                "floating_pnl": pnl
            })
            logging.info(f"Open Position - Symbol: {symbol}, Side: {side}, Size: {size}, Floating PnL: {pnl:.4f}")
            send_telegram_message(f"Open Position - Symbol: {symbol}, Side: {side}, Size: {size}, Floating PnL: {pnl:.4f}")
        return positions
    logging.warning("No open positions.")
    return []

def get_account_summary():
    endpoint = "/api/v5/account/balance"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        for asset in response["data"][0]["details"]:
            if asset["ccy"] == "USDT":
                equity = float(asset.get("eq", 0))
                realized_pnl = float(asset.get("upl", 0))
                unrealized_pnl = float(asset.get("upl", 0))
                logging.info(f"Account Summary - Equity: {equity:.4f} USDT, Realized PnL: {realized_pnl:.4f} USDT, Unrealized PnL: {unrealized_pnl:.4f} USDT")
                send_telegram_message(f"Account Summary - Equity: {equity:.4f} USDT, Realized PnL: {realized_pnl:.4f} USDT, Unrealized PnL: {unrealized_pnl:.4f} USDT")
                return {
                    "equity": equity,
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl
                }
    logging.warning("Failed to get account summary.")
    return None

# Machine Learning Models
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_regressor = RandomForestRegressor(n_estimators=100)
gb_classifier = GradientBoostingClassifier(n_estimators=100)
gb_regressor = GradientBoostingRegressor(n_estimators=100)

def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(units=50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(units=1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Dataset preparation
def prepare_dataset(df):
    if len(df) < 2:
        raise ValueError("Insufficient data")
    X = df[['close']].values
    y_class = [1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0 for i in range(1, len(df))]
    y_reg = df['close'].iloc[1:].values
    return X[:-1], np.array(y_class), y_reg

def prepare_lstm_data(df, look_back=10):
    X, y = [], []
    for i in range(len(df) - look_back - 1):
        X.append(df[['close']].iloc[i:i+look_back].values)
        y.append(df['close'].iloc[i + look_back])
    return np.array(X), np.array(y)

# Model training
def train_random_forest(df):
    X, y_class, y_reg = prepare_dataset(df)
    rf_classifier.fit(X, y_class)
    rf_regressor.fit(X, y_reg)
    logging.info("Random Forest trained successfully")

def train_gradient_boosting(df):
    X, y_class, y_reg = prepare_dataset(df)
    gb_classifier.fit(X, y_class)
    gb_regressor.fit(X, y_reg)
    logging.info("Gradient Boosting trained successfully")

def train_lstm(df):
    look_back = 10
    X, y = prepare_lstm_data(df, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = build_lstm_model((look_back, 1))
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    return model

# Trading strategy variables
successful_trades = 0
failed_trades = 0
entry_price = None
position = None

def rf_gb_lstm_strategy():
    global successful_trades, failed_trades, entry_price, position
    set_leverage(LEVERAGE)
    lstm_model = None
    while True:
        try:
            candlestick_data = get_candlestick_data()
            if candlestick_data.empty:
                logging.warning("No data, retrying...")
                time.sleep(CHECK_INTERVAL)
                continue
            if len(candlestick_data) >= DATA_LIMIT:
                train_random_forest(candlestick_data)
                train_gradient_boosting(candlestick_data)
                lstm_model = train_lstm(candlestick_data)
            latest_close = candlestick_data['close'].iloc[-1]
            latest_features = np.array([[latest_close]])
            # Random Forest predictions
            rf_dir = rf_classifier.predict(latest_features)[0]
            rf_price = rf_regressor.predict(latest_features)[0]
            # Gradient Boosting predictions
            gb_dir = gb_classifier.predict(latest_features)[0]
            gb_price = gb_regressor.predict(latest_features)[0]
            # LSTM predictions
            if lstm_model:
                lstm_input = candlestick_data['close'].iloc[-10:].values.reshape(1, 10, 1)
                lstm_price = lstm_model.predict(lstm_input)[0][0]
                lstm_dir = 1 if lstm_price > latest_close else 0
            else:
                lstm_dir = 0
                lstm_price = 0
            # Combine signals
            avg_dir = (rf_dir + gb_dir + lstm_dir) / 3
            signal = 1 if avg_dir >= 0.5 else 0
            logging.info(f"Combined Signal: {signal} (RF: {rf_dir}, GB: {gb_dir}, LSTM: {lstm_dir})")

            # Execute trade
            if signal == 1 and position != "long":
                close_all_positions()
                response = place_order("buy", "long")
                if response.get("code") == "0":
                    position = "long"
                    entry_price = latest_close
                    successful_trades += 1
                    logging.info(f"LONG opened at {entry_price}")
                    send_telegram_message(f"LONG OPENED at {entry_price}")
                else:
                    failed_trades += 1
                    logging.error(f"Failed to open LONG position: {response}")
            elif signal == 0 and position != "short":
                close_all_positions()
                response = place_order("sell", "short")
                if response.get("code") == "0":
                    position = "short"
                    entry_price = latest_close
                    successful_trades += 1
                    logging.info(f"SHORT opened at {entry_price}")
                    send_telegram_message(f"SHORT OPENED at {entry_price}")
                else:
                    failed_trades += 1
                    logging.error(f"Failed to open SHORT position: {response}")

            # Close position if the signal changes
            if position:
                current_price = get_realtime_price() or latest_close
                if position == "long" and signal == 0:
                    close_all_positions()
                    logging.info(f"LONG closed at {current_price}")
                    send_telegram_message(f"LONG CLOSED at {current_price}")
                elif position == "short" and signal == 1:
                    close_all_positions()
                    logging.info(f"SHORT closed at {current_price}")
                    send_telegram_message(f"SHORT CLOSED at {current_price}")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            time.sleep(CHECK_INTERVAL)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    logging.info("Starting bot in SIMULATED mode")
    send_telegram_message("Bot started in SIMULATED mode")
    try:
        rf_gb_lstm_strategy()
    except KeyboardInterrupt:
        logging.info("Stopping bot...")
        close_all_positions()
