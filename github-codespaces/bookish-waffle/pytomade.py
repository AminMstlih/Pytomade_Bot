import time
import hmac
import hashlib
import base64
import requests
import datetime
import json
import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Load API credentials from .env file
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Load trading configuration
CONFIG = {
    "symbol": os.getenv("TRADING_SYMBOL", "BTC-USDT-SWAP"),
    "leverage": int(os.getenv("LEVERAGE", 10)),
    "order_size": float(os.getenv("ORDER_SIZE", 0.01)),
    "short_ma": int(os.getenv("SHORT_MA", 13)),
    "long_ma": int(os.getenv("LONG_MA", 21)),
    "timeframe": os.getenv("TIMEFRAME", "1m"),
    "threshold_percent": float(os.getenv("THRESHOLD_PERCENT", 0.05))
}

BASE_URL = "https://www.okx.com"

# Fungsi untuk mendapatkan waktu server OKX
def get_server_time():
    endpoint = "/api/v5/public/time"
    try:
        response = requests.get(BASE_URL + endpoint)
        response.raise_for_status()
        return str(float(response.json()["data"][0]["ts"]) / 1000.0)
    except Exception as e:
        logging.error(f"Gagal mendapatkan server time: {e}")
        return str(time.time())

# Buat tanda tangan untuk request
def generate_signature(timestamp, method, request_path, body=""):
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

# Fungsi untuk mengirim request ke OKX
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
            "x-simulated-trading": "1"  # 1 untuk testnet, 0 untuk real trading
        }
        url = BASE_URL + endpoint
        response = requests.request(method, url, headers=headers, data=body_json)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": str(e)}

# Fungsi untuk mengatur leverage
def set_leverage():
    endpoint = "/api/v5/account/set-leverage"
    body = {
        "instId": CONFIG["symbol"],
        "lever": str(CONFIG["leverage"]),
        "mgnMode": "cross"
    }
    response = send_request("POST", endpoint, body)
    logging.info(f"Leverage Response: {response}")

# Fungsi untuk mendapatkan harga historis
def get_prices():
    endpoint = f"/api/v5/market/candles?instId={CONFIG['symbol']}&bar={CONFIG['timeframe']}&limit={CONFIG['long_ma']}"
    try:
        response = send_request("GET", endpoint)
        if "data" in response and len(response["data"]) >= CONFIG["long_ma"]:
            return [float(candle[4]) for candle in response["data"]]
        else:
            logging.warning(f"Data tidak cukup atau format tidak valid: {response}")
            return []
    except Exception as e:
        logging.error(f"Error mendapatkan harga: {e}")
        return []

# Fungsi untuk menghitung Moving Average
def moving_average(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

# Fungsi untuk eksekusi market order
def place_order(side, pos_side):
    endpoint = "/api/v5/trade/order"
    order_data = {
        "instId": CONFIG["symbol"],
        "tdMode": "cross",
        "side": side,
        "posSide": pos_side,
        "ordType": "market",
        "sz": str(CONFIG["order_size"] * CONFIG["leverage"])
    }
    response = send_request("POST", endpoint, body=order_data)
    logging.info(f"Order Response ({side}/{pos_side}): {response}")
    send_notification(f"Order executed: {side}/{pos_side} - {CONFIG['symbol']}")
    return response

# Fungsi untuk mengecek posisi terbuka
def check_open_positions():
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        for position in response["data"]:
            if position["instId"] == CONFIG["symbol"] and float(position["pos"]) > 0:
                return True
    return False

# Fungsi untuk menutup semua posisi
def close_all_positions():
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    if "data" in response:
        for position in response["data"]:
            if position["instId"] == CONFIG["symbol"] and float(position["pos"]) > 0:
                side = "buy" if position["posSide"] == "short" else "sell"
                logging.info(f"Closing {position['posSide']} position...")
                place_order(side, position["posSide"])
    else:
        logging.info("No open positions to close.")

# Fungsi untuk mengecek balance akun
def check_balance():
    endpoint = "/api/v5/account/balance"
    response = send_request("GET", endpoint)
    if "data" in response:
        for item in response["data"]:
            if item["ccy"] == "USDT":
                balance = float(item["availBal"])
                logging.info(f"Balance tersedia: {balance} USDT")
                return balance
    logging.warning("Gagal mendapatkan balance")
    return 0.0

# Fungsi untuk mengirim notifikasi Telegram
def send_notification(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": f"Trading Bot Notification:\n{message}"
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            logging.error(f"Gagal mengirim notifikasi Telegram: {e}")
    else:
        logging.info(f"Notification: {message}")

# Strategi MA Cross dengan threshold
def ma_cross_strategy():
    set_leverage()
    last_signal = None
    
    while True:
        # Cek balance terlebih dahulu
        if check_balance() < CONFIG["order_size"] * CONFIG["leverage"]:
            logging.warning("Balance tidak cukup, menunggu...")
            time.sleep(300)
            continue
            
        prices = get_prices()
        if len(prices) < CONFIG["long_ma"]:
            logging.warning(f"Data tidak cukup (hanya {len(prices)}), membutuhkan {CONFIG['long_ma']} data")
            time.sleep(60)
            continue

        short_ma = moving_average(prices, CONFIG["short_ma"])
        long_ma = moving_average(prices, CONFIG["long_ma"])
        
        if short_ma is None or long_ma is None:
            time.sleep(60)
            continue
            
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"{current_time} - Short MA: {short_ma:.2f}, Long MA: {long_ma:.2f}")

        # Hitung threshold absolut
        threshold = CONFIG["threshold_percent"] / 100 * long_ma
        
        if (short_ma - long_ma) > threshold and last_signal != "short":
            logging.info("Signal SHORT terdeteksi")
            close_all_positions()
            place_order("sell", "short")
            last_signal = "short"
            send_notification(f"SHORT signal - {CONFIG['symbol']}")
            
        elif (long_ma - short_ma) > threshold and last_signal != "long":
            logging.info("Signal LONG terdeteksi")
            close_all_positions()
            place_order("buy", "long")
            last_signal = "long"
            send_notification(f"LONG signal - {CONFIG['symbol']}")
        
        time.sleep(60)

# Jalankan strategi
if __name__ == "__main__":
    try:
        logging.info("Memulai trading bot...")
        send_notification(f"🚀 Trading Bot started for {CONFIG['symbol']}")
        ma_cross_strategy()
    except KeyboardInterrupt:
        logging.info("\nMenghentikan bot...")
        close_all_positions()
        send_notification("🛑 Trading Bot stopped manually")
    except Exception as e:
        logging.error(f"Error utama: {e}")
        send_notification(f"⚠️ Bot crashed: {e}")
        raise
