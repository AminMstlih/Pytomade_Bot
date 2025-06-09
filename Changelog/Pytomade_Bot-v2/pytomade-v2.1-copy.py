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
import numpy as np

# Load API credentials from .env file
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# Variabel Global
BASE_URL = "https://www.okx.com"
SYMBOL = "DOGE-USDT-SWAP"  # Ganti ticker token di sini
LEVERAGE = 15  # Ganti leverage di sini
ORDER_SIZE = 0.02  # Ganti ukuran order di sini
SHORT_MA = 13  # Moving Average pendek
LONG_MA = 21  # Moving Average panjang

# Set up logging to track bot activity
logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
            "OK-ACCESS-SIGN": generate_signature(timestamp, method, endpoint, body_json),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/json",
            "x-simulated-trading": "0"  # 0 = Simulated trading mode / 1 = real
        }
        url = BASE_URL + endpoint
        response = requests.request(method, url, headers=headers, data=body_json)
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

# Fungsi untuk menghitung Moving Average
def moving_average(prices, period):
    """
    Menghitung Simple Moving Average (SMA) dari harga.
    """
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

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

# Fungsi untuk mengecek posisi terbuka
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

# Variabel untuk melacak performa
total_profit = 0
successful_trades = 0
failed_trades = 0

# Strategi MA Cross (15 dan 21) + Machine Learning
def ma_cross_ml_strategy():
    """
    Strategi trading agresif berdasarkan MA Cross (15 dan 21) dan prediksi ML.
    """
    global total_profit, successful_trades, failed_trades
    # Pastikan leverage sudah diatur
    set_leverage(LEVERAGE)  # Atur leverage sesuai variabel global
    position = None
    model = LinearRegression()  # Model ML untuk prediksi harga

    while True:
        prices = get_prices()
        current_price = get_realtime_price()
        if not prices or not current_price:
            logging.warning("Gagal mendapatkan harga, mencoba lagi...")
            time.sleep(15)  # Tunggu 15 detik sebelum mencoba lagi
            continue

        # Hitung MA Cross (15 dan 21)
        short_ma = moving_average(prices, SHORT_MA)
        long_ma = moving_average(prices, LONG_MA)
        if short_ma is None or long_ma is None:
            logging.warning("Tidak cukup data untuk menghitung MA.")
            time.sleep(15)
            continue

        # Prediksi harga dengan machine learning
        X = np.array(range(len(prices))).reshape(-1, 1)  # Input: indeks harga
        y = np.array(prices)  # Output: harga
        model.fit(X, y)
        predicted_price = model.predict([[len(prices)]])[0]
        logging.info(f"Short MA: {short_ma}, Long MA: {long_ma}, Current Price: {current_price}, Predicted Price: {predicted_price}")

        # Logika trading berdasarkan MA Cross dan prediksi ML
        if short_ma > long_ma and predicted_price > current_price and (position != "long"):  # Golden Cross + harga naik
            close_all_positions()
            logging.info("Opening LONG position...")
            response = place_order("buy", "long", ORDER_SIZE)
            position = "long"
            if response.get("code") == "0":  # Jika order berhasil
                successful_trades += 1
                total_profit += (current_price - short_ma) * ORDER_SIZE * LEVERAGE
            else:
                failed_trades += 1
        elif short_ma < long_ma and predicted_price < current_price and (position != "short"):  # Death Cross + harga turun
            close_all_positions()
            logging.info("Opening SHORT position...")
            response = place_order("sell", "short", ORDER_SIZE)
            position = "short"
            if response.get("code") == "0":  # Jika order berhasil
                successful_trades += 1
                total_profit += (short_ma - current_price) * ORDER_SIZE * LEVERAGE
            else:
                failed_trades += 1

        # Log performa bot
        win_rate = successful_trades / (successful_trades + failed_trades) if (successful_trades + failed_trades) > 0 else 0
        logging.info(f"Total Profit: {total_profit}, Win Rate: {win_rate:.2f}, Successful Trades: {successful_trades}, Failed Trades: {failed_trades}")

        time.sleep(15)  # Cek setiap 15 detik

# Jalankan strategi
if __name__ == "__main__":
    logging.info("Starting MA Cross (15 and 21) + Machine Learning trading bot...")
    ma_cross_ml_strategy()
