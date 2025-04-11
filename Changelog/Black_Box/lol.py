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
stop_loss = 0
take_profit = 0

# Config
BASE_URL = "https://www.okx.com"
symbol = 'BTC-USDT-SWAP'
SYMBOL = 'BTC-USDT-SWAP'
timeframe = '5m'
LEVERAGE = 15
ORDER_SIZE = 10
limit = 200
loop_interval = 30  # in seconds

# API credentials
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

STOP_LOSS_PERCENTAGE = 0.02
TAKE_PROFIT_PERCENTAGE = 0.05

# Fungsi untuk mendapatkan waktu server OKX
def get_server_time():
    """
    Mendapatkan waktu server OKX untuk digunakan dalam tanda tangan API.
    """
    endpoint = "/api/v5/public/time"
    response = requests.get(BASE_URL + endpoint)
    response.raise_for_status()
    return str(float(response.json()["data"][0]["ts"]) / 1000.0)

# Buat tanda tangan untuk request
def generate_signature(timestamp, method, request_path, body=""):
    """
    Menghasilkan tanda tangan HMAC SHA256 untuk autentikasi API.
    """
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

# Fungsi untuk mengirim request ke OKX
def send_request(method, endpoint, body=None):
    """
    Mengirim request ke API OKX dengan autentikasi dan penanganan error.
    """
    try:
        timestamp = get_server_time()
        body_json = json.dumps(body) if body else ""
        headers = {
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": generate_signature(timestamp, method, endpoint,body_json),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/json",
            "x-simulated-trading": "1"  # Simulated trading mode
        }
        url = BASE_URL + endpoint
        response = requests.request(method, url, headers=headers, data=json.dumps(body) if body else "")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {"error": str(e)}

# Fungsi untuk mengatur leverage
def set_leverage(leverage=LEVERAGE):
    """
    Mengatur leverage untuk akun trading.
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

# Fungsi untuk mendapatkan harga real-time
def get_realtime_price():
    """
    Mendapatkan harga terakhir (real-time) dari ticker.
    """
    endpoint = f"/api/v5/market/ticker?instId={SYMBOL}"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        return float(response["data"][0]["last"])
    logging.warning("Gagal mendapatkan harga real-time.")
    return None

# Fungsi untuk mendapatkan harga historis (DURASI 1 MENIT)
def get_prices():
    """
    Mendapatkan harga candlestick (1 menit) untuk strategi trading.
    """
    endpoint = f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit={LONG_MA}"  # Candlestick 1 menit
    response = send_request("GET", endpoint)
    if "data" in response:
        return [float(candle[4]) for candle in response["data"]]
    logging.warning("Gagal mendapatkan harga.")
    return []

# Fungsi untuk eksekusi market order
def place_order(side, pos_side, order_size=ORDER_SIZE):
    """
    Menempatkan order tanpa stop-loss dan take-profit.
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
    Memeriksa apakah ada posisi terbuka.
    """
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        return True
    return False

# Fungsi untuk menutup semua posisi
def close_all_positions():
    """
    Menutup semua posisi terbuka.
    """
    if check_open_positions():
        logging.info("Closing all positions...")
        place_order("buy", "short")  # Menutup posisi short jika ada
        place_order("sell", "long")  # Menutup posisi long jika ada
    else:
        logging.info("No open positions to close.")


# Fetch historical data
def fetch_data(timeframe='5m'):
    url = f'https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}'
    response = requests.get(url)
    data = response.json()['data']
    df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'vol_usd', 'confirm'])
    df = df[['ts', 'o', 'h', 'l', 'c', 'vol']].astype(float)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.sort_values('timestamp').reset_index(drop=True)

# Technical indicators
def add_indicators(df):
    df['ma15'] = df['close'].rolling(window=15).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['return'] = df['close'].pct_change()
    return df.dropna()

# Feature generation for ML
def generate_features(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna()
    if df.empty:
        return None, None
    X = df[['ma15', 'ma21', 'macd', 'macd_signal', 'return']]
    y = df['target']
    return X, y

# ML model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# ARIMA Forecast
def arima_prediction(df):
    try:
        model = ARIMA(df['close'], order=(2, 1, 2))
        model_fit = model.fit()
        return model_fit.forecast()[0]
    except:
        return df['close'].iloc[-1]

# Real sentiment from Coingecko (simple logic)
def real_sentiment():
    try:
        url = 'https://api.coingecko.com/api/v3/search/trending'
        response = requests.get(url).json()
        btc_trending = any('bitcoin' in coin['item']['id'] for coin in response['coins'])
        return 0.8 if btc_trending else -0.8
    except:
        return 0

# Decision logic
def generate_signal(df, model):
    latest = df.iloc[-1:]
    X = latest[['ma15', 'ma21', 'macd', 'macd_signal', 'return']]
    prediction = model.predict(X)[0]
    ma_cross = latest['ma15'].values[0] > latest['ma21'].values[0]
    macd_cross = latest['macd'].values[0] > latest['macd_signal'].values[0]
    arima_pred = arima_prediction(df)
    current_price = latest['close'].values[0]
    sentiment_score = real_sentiment()

    print(f"[ARIMA] {arima_pred:.2f}, Price: {current_price:.2f}, Sentiment: {sentiment_score:.2f}")

    if ma_cross and macd_cross and prediction == 1 and arima_pred > current_price and sentiment_score > 0:
        return 'BUY'
    elif not ma_cross and not macd_cross and prediction == 0 and arima_pred < current_price and sentiment_score < 0:
        return 'SELL'
    return 'HOLD'

# Simulated trade execution
def simulate_trade(signal, price):
    global position, entry_price, stop_loss, take_profit
    now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    set_leverage(LEVERAGE)  # Atur leverage sesuai variabel global
    position = None

    while True:
        prices = get_prices()
        current_price = get_realtime_price()
        if not prices or not current_price:
            logging.warning("Gagal mendapatkan harga, mencoba lagi...")
            time.sleep(15)  # Tunggu 15 detik sebelum mencoba lagi
            continue


    if signal == 'BUY' and (position != "long"):  # Golden Cross + harga naik
            close_all_positions()
            logging.info("Opening LONG position...")
            response = place_order("buy", "long", ORDER_SIZE)
            position = "long"
    elif signal == 'SELL' and (position != "short"):  # Death Cross + harga turun
            close_all_positions()
            logging.info("Opening SHORT position...")
            response = place_order("sell", "short", ORDER_SIZE)
            position = "short"

    elif position == 'LONG':
        if price <= stop_loss:
            print(f"{now} | STOP LOSS hit at {price:.2f}")
            position = None
        elif price >= take_profit:
            print(f"{now} | TAKE PROFIT hit at {price:.2f}")
            position = None

# Main loop
def main():
    df = fetch_data()
    df = add_indicators(df)
    X, y = generate_features(df)
    if X is None:
        return
    model = train_model(X, y)
    signal = generate_signal(df, model)
    price = df['close'].iloc[-1]
    simulate_trade(signal, price)

if __name__ == "__main__":
    while True:
        main()
        time.sleep(loop_interval)
