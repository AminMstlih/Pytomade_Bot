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
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hanya warning dan error penting yang tampil

load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

BASE_URL = "https://www.okx.com"
SYMBOL = "DOGE-USDT-SWAP"
LEVERAGE = 15
ORDER_SIZE = 0.02

logging.basicConfig(
    filename="bot.log",
    level=logging.DEBUG,  # Naikin level debug biar keliatan detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_server_time():
    endpoint = "/api/v5/public/time"
    response = requests.get(BASE_URL + endpoint)
    response.raise_for_status()
    return str(float(response.json()["data"][0]["ts"]) / 1000.0)

def generate_signature(timestamp, method, request_path, body=""):
    # PENTING: method harus uppercase
    message = f"{timestamp}{method.upper()}{request_path}{body}"
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
            "x-simulated-trading": "0"
        }
        url = BASE_URL + endpoint
        response = requests.request(method, url, headers=headers, data=body_json)
        logging.debug(f"Request {method} {url} - status {response.status_code} - response: {response.text}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if e.response is not None:
            logging.error(f"Request failed: {e.response.text}")
            try:
                return e.response.json()
            except:
                return {"error": e.response.text}
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
        return [float(candle[4]) for candle in response["data"][::-1]]
    logging.warning("Gagal mendapatkan harga.")
    return []

def place_order(side, pos_side, order_size=ORDER_SIZE):
    order_data = {
        "instId": SYMBOL,
        "tdMode": "cross",
        "side": side,
        "posSide": pos_side,
        "ordType": "market",
        "sz": str(order_size)
    }
    response = send_request("POST", "/api/v5/trade/order", body=order_data)
    logging.info(f"Order Response: {response}")
    return response

def check_open_positions():
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        for pos in response["data"]:
            if float(pos.get("availPos", 0)) > 0:
                return True
    return False

def close_all_positions():
    if check_open_positions():
        logging.info("Closing all positions...")
        place_order("buy", "short")
        place_order("sell", "long")
    else:
        logging.info("No open positions to close.")

# **Latih LSTM SEKALI sebelum loop utama biar bot ga macet**
def train_lstm_model(prices):
    window = 10
    X, y = [], []
    for i in range(len(prices) - window):
        X.append(prices[i:i+window])
        y.append(prices[i+window])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model = Sequential()
    model.add(Input(shape=(window, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)
    return model, window

def compute_macd(prices):
    short_ema = np.mean(prices[-12:])
    long_ema = np.mean(prices[-26:])
    macd = short_ema - long_ema
    signal = np.mean([macd] + [short_ema - np.mean(prices[-26-i:-i]) for i in range(1, 9)])
    return macd, signal

def ma_cross_signal(prices):
    ma_short = np.mean(prices[-15:])
    ma_long = np.mean(prices[-21:])
    if ma_short > ma_long:
        return "long"
    elif ma_short < ma_long:
        return "short"
    else:
        return "neutral"

def ml_regime_strategy():
    set_leverage(LEVERAGE)
    position = None

    # Latih LSTM SEKALI di awal
    prices_for_lstm = get_prices()
    if not prices_for_lstm:
        logging.error("Gagal mendapatkan harga untuk melatih LSTM, bot berhenti.")
        return
    lstm_model, lstm_window = train_lstm_model(prices_for_lstm)

    while True:
        prices = get_prices()
        current_price = get_realtime_price()
        if not prices or not current_price:
            time.sleep(10)
            continue

        volatility = np.std(prices[-10:])
        threshold = np.mean(prices[-10:]) * 0.005
        regime = "trend" if volatility < threshold else "mean_reversion"

        try:
            # **Gunakan model LSTM yang sudah dilatih**
            lstm_input = np.array(prices[-lstm_window:]).reshape(1, lstm_window, 1)
            lstm_pred = lstm_model.predict(lstm_input, verbose=0)[0][0]

            ma_signal = ma_cross_signal(prices)

            macd, signal = compute_macd(prices)
            macd_signal = "long" if macd > signal else "short"

            signals = [
                "long" if lstm_pred > current_price else "short",
                ma_signal,
                macd_signal
            ]
            final_signal = max(set(signals), key=signals.count)

            if final_signal == "long" and position != "long":
                close_all_positions()
                place_order("buy", "long")
                position = "long"
            elif final_signal == "short" and position != "short":
                close_all_positions()
                place_order("sell", "short")
                position = "short"

        except Exception as e:
            logging.error(f"Prediction error: {e}")

        """logging.info(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_price": lstm_pred,
            "current_price": current_price,
            "position": position,
            "regime": regime,
            "signal": final_signal
        }))"""

        time.sleep(15)

if __name__ == "__main__":
    logging.info("Starting Jim Simons-style trading bot...")
    try:
        ml_regime_strategy()
    except KeyboardInterrupt:
        logging.warning("Bot dihentikan oleh pengguna (KeyboardInterrupt). Menutup semua posisi...")
        close_all_positions()
    except Exception as e:
        logging.error(f"Bot error: {e}. Menutup semua posisi...")
        close_all_positions()
    finally:
        logging.info("Bot telah berhenti.")
