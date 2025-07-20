import time
import hmac
import hashlib
import base64
import requests
import json
import os
from dotenv import load_dotenv
import logging
from sklearn.linear_model import LinearRegression
from predictive_models import predict_arima, predict_xgboost, predict_lstm
import numpy as np

# Load API credentials from .env file
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# Variabel Global
BASE_URL = "https://www.okx.com"
SYMBOL = "OP-USDT-SWAP"
LEVERAGE = 15
ORDER_SIZE = 1000
SHORT_MA = 15
LONG_MA = 21

logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_server_time():
    endpoint = "/api/v5/public/time"
    response = requests.get(BASE_URL + endpoint)
    response.raise_for_status()
    return str(float(response.json()["data"][0]["ts"]) / 1000.0)

def generate_signature(timestamp, method, request_path, body=""):
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def send_request(method, endpoint, body=None):
    try:
        timestamp = get_server_time()
        body_json = json.dumps(body) if body else ""
        headers = {
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": generate_signature(timestamp, method, endpoint, body_json),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/json",
            "x-simulated-trading": "1"
        }
        url = BASE_URL + endpoint
        response = requests.request(method, url, headers=headers, data=body_json)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {"error": str(e)}

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

def get_realtime_price():
    endpoint = f"/api/v5/market/ticker?instId={SYMBOL}"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        return float(response["data"][0]["last"])
    logging.warning("Gagal mendapatkan harga real-time.")
    return None

def get_prices():
    endpoint = f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit=100"
    response = send_request("GET", endpoint)
    if "data" in response:
        return [float(candle[4]) for candle in reversed(response["data"])]
    logging.warning("Gagal mendapatkan harga.")
    return []

def moving_average(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def macd(prices, short=12, long=26, signal=9):
    exp1 = np.array(prices[-short:]).mean()
    exp2 = np.array(prices[-long:]).mean()
    macd_line = exp1 - exp2
    signal_line = np.array(prices[-signal:]).mean()
    return macd_line, signal_line

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

total_profit = 0
successful_trades = 0
failed_trades = 0

def ma_cross_ml_strategy():
    global total_profit, successful_trades, failed_trades
    set_leverage(LEVERAGE)
    position = None
    model = LinearRegression()

    while True:
        prices = get_prices()
        current_price = get_realtime_price()
        if not prices or not current_price:
            time.sleep(15)
            continue

        short_ma = moving_average(prices, SHORT_MA)
        long_ma = moving_average(prices, LONG_MA)
        if short_ma is None or long_ma is None:
            time.sleep(15)
            continue

        X = np.array(range(len(prices))).reshape(-1, 1)
        y = np.array(prices)
        model.fit(X, y)
        linear_pred = model.predict([[len(prices)]])[0]
        arima_pred = predict_arima(prices)
        xgb_pred = predict_xgboost(prices)
        lstm_pred = predict_lstm(prices)
        
        final_prediction = np.mean([linear_pred, arima_pred, xgb_pred, lstm_pred])
        macd_line, signal_line = macd(prices)

        logging.info(f"Short MA: {short_ma}, Long MA: {long_ma}, MACD: {macd_line}, Signal: {signal_line}, Final Pred: {final_prediction}, Current: {current_price}")

        if short_ma > long_ma and macd_line > signal_line and final_prediction > current_price and position != "long":
            close_all_positions()
            response = place_order("buy", "long")
            position = "long"
            if response.get("code") == "0":
                successful_trades += 1
                total_profit += (current_price - short_ma) * ORDER_SIZE * LEVERAGE
            else:
                failed_trades += 1

        elif short_ma < long_ma and macd_line < signal_line and final_prediction < current_price and position != "short":
            close_all_positions()
            response = place_order("sell", "short")
            position = "short"
            if response.get("code") == "0":
                successful_trades += 1
                total_profit += (short_ma - current_price) * ORDER_SIZE * LEVERAGE
            else:
                failed_trades += 1

        win_rate = successful_trades / (successful_trades + failed_trades) if (successful_trades + failed_trades) > 0 else 0
        logging.info(f"Profit: {total_profit}, Win Rate: {win_rate:.2f}, Success: {successful_trades}, Failed: {failed_trades}")
        time.sleep(15)

if __name__ == "__main__":
    logging.info("Starting upgraded bot with MACD + ML Predictors...")
    ma_cross_ml_strategy()
