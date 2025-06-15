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
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from scipy.stats import linregress
import math

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

BASE_URL = "https://www.okx.com"
SYMBOL = "DOGE-USDT-SWAP"
LEVERAGE = 15
INITIAL_CAPITAL = 150000  # in USDT
RISK_PER_TRADE = 0.01   # 1% of capital per trade
STOP_LOSS_PCT = 0.02    # 2%
TAKE_PROFIT_PCT = 0.04  # 4%
CONTRACT_SIZE = 10      # for DOGE-USDT-SWAP
MAX_DRAWDOWN = 0.2      # 20%

# Logging setup
logging.basicConfig(
    filename="bot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- UTILITY FUNCTIONS ---
def get_server_time():
    endpoint = "/api/v5/public/time"
    response = requests.get(BASE_URL + endpoint)
    response.raise_for_status()
    return str(float(response.json()["data"][0]["ts"]) / 1000.0)

def generate_signature(timestamp, method, request_path, body=""):
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def send_request(method, endpoint, body=None):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            timestamp = get_server_time()
            body_json = json.dumps(body) if body else ""
            headers = {
                "OK-ACCESS-KEY": API_KEY,
                "OK-ACCESS-SIGN": generate_signature(timestamp, method, endpoint, body_json),
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": PASSPHRASE,
                "Content-Type": "application/json",
                "x-simulated-trading": "1"  # Uncomment for simulated trading
            }
            url = BASE_URL + endpoint
            response = requests.request(method, url, headers=headers, data=body_json)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed: {e}, retrying ({attempt+1}/{max_retries})")
            time.sleep(2 ** attempt)
    logging.error(f"Request failed after {max_retries} attempts: {e}")
    return {"error": str(e)}

# --- TRADING CONTROL ---
def set_leverage(leverage=LEVERAGE):
    endpoint = "/api/v5/account/set-leverage"
    body = {"instId": SYMBOL, "lever": str(leverage), "mgnMode": "cross"}
    return send_request("POST", endpoint, body)

def get_account_balance():
    endpoint = "/api/v5/account/balance?ccy=USDT"
    res = send_request("GET", endpoint)
    if "data" in res and res["data"]:
        return float(res["data"][0]["totalEq"])
    return 0

def get_realtime_price():
    endpoint = f"/api/v5/market/ticker?instId={SYMBOL}"
    res = send_request("GET", endpoint)
    if "data" in res:
        return float(res["data"][0]["last"]) if "data" in res and res["data"] else None

def get_prices(limit=100):
    endpoint = f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit={limit}"
    res = send_request("GET", endpoint)
    return [float(candle[4]) for candle in res["data"][::-1]] if "data" in res else []

def place_order(side, pos_side, n, position_size):
    if n <= 0:
        logging.warning("Position size is zero, skipping order.")
        return {"msg": "Position size is zero"}
    body = {
        "instId": SYMBOL,
        "tdMode": "cross",
        "side": side,
        "posSide": pos_side,
        "ordType": "market",
        "sz": str(n)
    }
    res = send_request("POST", "/api/v5/trade/order", body)
    if "data" in res and res["data"]:
        logging.info(f"Order placed: {res['data']}")
    else:
        logging.error(f"Order placement failed: {res}")
    return res

def close_all_positions():
    global current_position
    if current_position is not None:
        side = "sell" if current_position["pos_side"] == "long" else "buy"
        pos_side = current_position["pos_side"]
        n = current_position["n"]
        place_order(side, pos_side, n)
        current_position = None

# --- MODEL TRAINING ---
def train_lstm_model(prices):
    prices_array = np.array(prices).reshape(-1, 1)
    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices_array)
    window = 10
    X, y = [], []
    for i in range(len(prices_scaled) - window):
        X.append(prices_scaled[i:i+window])
        y.append(prices_scaled[i+window])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(Input(shape=(window, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)
    return model, window, scaler

# --- TECHNICAL INDICATORS ---
def compute_macd(prices):
    short_ema = pd.Series(prices).ewm(span=12).mean().iloc[-1]
    long_ema = pd.Series(prices).ewm(span=26).mean().iloc[-1]
    macd_line = short_ema - long_ema
    signal_line = pd.Series([macd_line] * 9).ewm(span=9).mean().iloc[-1]
    return macd_line, signal_line

def ma_cross_signal(prices):
    ma_short = np.mean(prices[-15:])
    ma_long = np.mean(prices[-21:])
    if ma_short > ma_long:
        return "long"
    elif ma_short < ma_long:
        return "short"
    return "neutral"

def hurst_exponent(time_series):
    lags = range(2, len(time_series)//2)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    reg = linregress(np.log(lags), np.log(tau))
    return reg.slope * 2

def detect_regime(prices):
    volatility = np.std(prices[-10:])
    threshold = np.mean(prices[-10:]) * 0.005
    hurst = hurst_exponent(prices)
    trend_regime = "trend" if hurst > 0.55 else "mean_reversion"
    vol_regime = "trend" if volatility < threshold else "mean_reversion"
    return trend_regime if trend_regime == vol_regime else "mixed"

# --- SIGNAL AGGREGATION ---
def aggregate_signals(model_pred, current_price, ma_signal, macd_signal, regime, prices):
    lstm_confidence = abs(model_pred - current_price) / current_price
    ma_distance = abs(np.mean(prices[-15:]) - np.mean(prices[-21:])) / current_price
    macd_strength = abs(compute_macd(prices)[0]) / current_price
    signals = {"long": 0, "short": 0}
    if model_pred > current_price:
        signals["long"] += lstm_confidence * 0.5
    else:
        signals["short"] += lstm_confidence * 0.5
    if ma_signal == "long":
        signals["long"] += ma_distance * 0.3
    elif ma_signal == "short":
        signals["short"] += ma_distance * 0.3
    if macd_signal == "long":
        signals["long"] += macd_strength * 0.2
    elif macd_signal == "short":
        signals["short"] += macd_strength * 0.2
    return "long" if signals["long"] > signals["short"] else "short"

# --- MAIN STRATEGY LOOP ---
def ml_regime_strategy():
    global current_position
    set_leverage(LEVERAGE)
    account_value = get_account_balance() or INITIAL_CAPITAL
    peak_value = account_value
    current_position = None
    trades = []

    # Train LSTM model once at start
    prices = get_prices()
    if not prices:
        logging.error("No initial price data available.")
        return
    lstm_model, window, scaler = train_lstm_model(prices)

    while True:
        try:
            latest_prices = get_prices()
            current_price = get_realtime_price()
            if not latest_prices or not current_price:
                logging.warning("Missing price data, retrying...")
                time.sleep(10)
                continue
            prices = latest_prices[-100:]

            # Update account value and peak value
            account_value = get_account_balance()
            peak_value = max(peak_value, account_value)

            # Check for maximum drawdown
            if (peak_value - account_value) / peak_value > MAX_DRAWDOWN:
                logging.error("Maximum drawdown exceeded. Stopping bot.")
                close_all_positions()
                break

            # Predict next price using LSTM
            scaled_input = scaler.transform(np.array(prices[-window:]).reshape(-1, 1))
            lstm_input = scaled_input.reshape(1, window, 1)
            lstm_pred_scaled = lstm_model.predict(lstm_input, verbose=0)
            lstm_pred = scaler.inverse_transform(lstm_pred_scaled)[0][0]

            # Calculate indicators
            regime = detect_regime(prices)
            ma_sig = ma_cross_signal(prices)
            macd_val, signal_val = compute_macd(prices)
            macd_sig = "long" if macd_val > signal_val else "short"
            final_signal = aggregate_signals(lstm_pred, current_price, ma_sig, macd_sig, regime, prices)

            # Calculate position size
            risk_amount = account_value * RISK_PER_TRADE
            entry_price = current_price
            if final_signal == "long":
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            else:
                stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)
            if abs(entry_price - stop_loss_price) > 0:
                n = int(risk_amount / (abs(entry_price - stop_loss_price) * CONTRACT_SIZE))
            else:
                n = 0

            # Check stop-loss and take-profit for current position
            if current_position is not None:
                entry_price = current_position["entry_price"]
                if current_position["pos_side"] == "long":
                    if current_price <= entry_price * (1 - STOP_LOSS_PCT):
                        logging.info("Stop-loss triggered for long position.")
                        close_all_positions()
                        if trades and trades[-1]["exit"] is None:
                            trades[-1]["exit"] = current_price
                    elif current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                        logging.info("Take-profit triggered for long position.")
                        close_all_positions()
                        if trades and trades[-1]["exit"] is None:
                            trades[-1]["exit"] = current_price
                elif current_position["pos_side"] == "short":
                    if current_price >= entry_price * (1 + STOP_LOSS_PCT):
                        logging.info("Stop-loss triggered for short position.")
                        close_all_positions()
                        if trades and trades[-1]["exit"] is None:
                            trades[-1]["exit"] = current_price
                    elif current_price <= entry_price * (1 - TAKE_PROFIT_PCT):
                        logging.info("Take-profit triggered for short position.")
                        close_all_positions()
                        if trades and trades[-1]["exit"] is None:
                            trades[-1]["exit"] = current_price

            # Place new order if signal changes
            if final_signal == "long" and (current_position is None or current_position["pos_side"] != "long"):
                if current_position is not None:
                    logging.info("Closing existing position before opening new one.")
                    close_all_positions()
                    if trades and trades[-1]["exit"] is None:
                        trades[-1]["exit"] = current_price
                place_order("buy", "long", n, position_size)
                current_position = {"pos_side": "long", "entry_price": current_price, "n": n}
                trades.append({"entry": current_price, "exit": None, "signal": "long"})
            elif final_signal == "short" and (current_position is None or current_position["pos_side"] != "short"):
                if current_position is not None:
                    logging.info("Closing existing position before opening new one.")
                    close_all_positions()
                    if trades and trades[-1]["exit"] is None:
                        trades[-1]["exit"] = current_price
                place_order("sell", "short", n, position_size)
                current_position = {"pos_side": "short", "entry_price": current_price, "n": n}
                trades.append({"entry": current_price, "exit": None, "signal": "short"})

            # Log metrics
            logging.info({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "price": current_price,
                "predicted": lstm_pred,
                "signal": final_signal,
                "position": current_position["pos_side"] if current_position else None,
                "regime": regime
            })

        except Exception as e:
            logging.error(f"Error in loop: {e}")

        time.sleep(15)

if __name__ == "__main__":
    logging.info("Starting Jim Simons-style trading bot...")
    try:
        ml_regime_strategy()
    except KeyboardInterrupt:
        logging.warning("Bot stopped by user.")
        close_all_positions()
