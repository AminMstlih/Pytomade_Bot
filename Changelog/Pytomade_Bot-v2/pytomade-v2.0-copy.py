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
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Load API credentials from .env file
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# Global Configuration
BASE_URL = "https://www.okx.com" 
SYMBOL = "DOGE-USDT-SWAP"
LEVERAGE = 15
ORDER_SIZE = 0.02  # Base order size (scaled dynamically)
SHORT_MA = 50
LONG_MA = 200
LOOKBACK = 60  # LSTM lookback window
COOLDOWN_SECONDS = 60  # Minimum time between trades
MAX_ERRORS = 5  # Circuit breaker threshold

# Logging Configuration
logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Risk Management Parameters
MAX_ACCOUNT_RISK = 0.01  # 1% per trade
MIN_VOLATILITY = 0.001  # Minimum volatility to trade

# Classes and Functions
class OKXTrader:
    def __init__(self):
        self.account_balance = 0.0
        self.position = None
        self.last_trade_time = 0
        self.consecutive_errors = 0

    def get_server_time(self):
        endpoint = "/api/v5/public/time"
        response = requests.get(BASE_URL + endpoint)
        return str(float(response.json()["data"][0]["ts"]) / 1000.0)

    def generate_signature(self, timestamp, method, path, body=""):
        msg = f"{timestamp}{method.upper()}{path}{body}"
        mac = hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def send_request(self, method, endpoint, body=None):
        timestamp = self.get_server_time()
        body_json = json.dumps(body) if body else ""
        headers = {
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": self.generate_signature(timestamp, method, endpoint, body_json),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/json",
        }
        try:
            response = requests.request(method, BASE_URL + endpoint, headers=headers, data=body_json)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.consecutive_errors += 1
            logging.error(f"API Error: {e}")
            return None

    def get_account_balance(self):
        response = self.send_request("GET", "/api/v5/account/account")
        if response and "data" in response:
            self.account_balance = float(response["data"][0]["eq"])
            return self.account_balance
        return None

    def set_leverage(self, leverage):
        endpoint = "/api/v5/account/set-leverage"
        body = {
            "instId": SYMBOL,
            "lever": str(leverage),
            "mgnMode": "cross"
        }
        response = self.send_request("POST", endpoint, body)
        return response

    def place_order(self, side, pos_side, size):
        endpoint = "/api/v5/trade/order"
        body = {
            "instId": SYMBOL,
            "tdMode": "cross",
            "side": side,
            "posSide": pos_side,
            "ordType": "market",
            "sz": str(size)
        }
        response = self.send_request("POST", endpoint, body)
        return response

    def close_position(self):
        if self.position == "long":
            self.place_order("sell", "long", ORDER_SIZE)
        elif self.position == "short":
            self.place_order("buy", "short", ORDER_SIZE)
        self.position = None

    def check_position(self):
        endpoint = "/api/v5/account/positions"
        response = self.send_request("GET", endpoint)
        if response and "data" in response:
            for pos in response["data"]:
                if pos["pos"] != "0":
                    self.position = pos["posSide"].lower()
                    return True
        return False

class DataHandler:
    def __init__(self):
        self.prices = []
        self.volatility = 0.0

    def fetch_historical_prices(self):
        endpoint = f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit=150"
        response = requests.get(BASE_URL + endpoint)
        if response.status_code == 200:
            candles = response.json()["data"]
            self.prices = [float(candle[4]) for candle in candles]
            return self.prices
        return []

    def calculate_atr(self, window=14):
        high = np.array([float(c[2]) for c in candles])
        low = np.array([float(c[3]) for c in candles])
        close = np.array([float(c[4]) for c in candles])
        tr = np.maximum(high - low, np.abs(high - close[:-1]), np.abs(low - close[:-1]))
        atr = np.mean(tr[-window:])
        return atr

    def prepare_lstm_data(self, prices):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(np.array(prices).reshape(-1, 1))
        x = []
        y = []
        for i in range(LOOKBACK, len(prices)):
            x.append(scaled_data[i - LOOKBACK:i, 0])
            y.append(scaled_data[i, 0])
        x = np.array(x)
        y = np.array(y)
        return x, y, scaler

class LSTMStrategy:
    def __init__(self):
        self.model = self.load_trained_model()
        self.ma_short = []
        self.ma_long = []

    def load_trained_model(self):
        try:
            return load_model("lstm_model.h5")
        except:
            logging.error("Model not found. Train and save the model first.")
            return None

    def generate_signal(self, prices, atr):
        if len(prices) < max(SHORT_MA, LONG_MA):
            return "neutral"

        # Calculate MAs
        self.ma_short = np.mean(prices[-SHORT_MA:])
        self.ma_long = np.mean(prices[-LONG_MA:])
        
        # LSTM Prediction
        latest_data = np.array(prices[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        prediction = self.model.predict(latest_data)
        
        # Volatility Filter
        if atr < MIN_VOLATILITY:
            return "neutral"

        # Signal Logic
        if (self.ma_short > self.ma_long) and (prediction > prices[-1]):
            return "long"
        elif (self.ma_short < self.ma_long) and (prediction < prices[-1]):
            return "short"
        else:
            return "neutral"

def main():
    trader = OKXTrader()
    data_handler = DataHandler()
    strategy = LSTMStrategy()
    trade_history = []

    # Initialize settings
    trader.set_leverage(LEVERAGE)
    trader.account_balance = trader.get_account_balance()
    
    while True:
        try:
            # Fetch Data
            prices = data_handler.fetch_historical_prices()
            current_price = prices[-1]
            atr = data_handler.calculate_atr()
            
            # Check for new signal
            signal = strategy.generate_signal(prices, atr)
            
            # Position Checks
            has_position = trader.check_position()
            
            # Execute Trade Logic
            if (time.time() - trader.last_trade_time) > COOLDOWN_SECONDS and trader.consecutive_errors < MAX_ERRORS:
                if signal == "long" and not has_position:
                    # Calculate position size
                    risk_per_trade = trader.account_balance * MAX_ACCOUNT_RISK
                    order_size = (risk_per_trade / (current_price * LEVERAGE)) * current_price
                    order_size = min(order_size, ORDER_SIZE * 10)  # Cap order size
                    
                    # Place order
                    response = trader.place_order("buy", "long", order_size)
                    if response.get("code") == "0":
                        trader.position = "long"
                        trader.last_trade_time = time.time()
                        trade_history.append({
                            "entry": current_price,
                            "time": datetime.now(),
                            "signal": signal
                        })
                        logging.info(f"Opened LONG position at {current_price} with size {order_size}")
                
                elif signal == "short" and not has_position:
                    risk_per_trade = trader.account_balance * MAX_ACCOUNT_RISK
                    order_size = (risk_per_trade / (current_price * LEVERAGE)) * current_price
                    order_size = min(order_size, ORDER_SIZE * 10)
                    
                    response = trader.place_order("sell", "short", order_size)
                    if response.get("code") == "0":
                        trader.position = "short"
                        trader.last_trade_time = time.time()
                        trade_history.append({
                            "entry": current_price,
                            "time": datetime.now(),
                            "signal": signal
                        })
                        logging.info(f"Opened SHORT position at {current_price} with size {order_size}")
                
                # Close position if signal reverses
                elif (signal == "neutral" and has_position):
                    trader.close_position()
                    exit_price = current_price
                    entry_price = trade_history[-1]["entry"]
                    profit = (exit_price - entry_price) * order_size * LEVERAGE if trader.position == "long" else (entry_price - exit_price) * order_size * LEVERAGE
                    trade_history[-1]["exit"] = exit_price
                    trade_history[-1]["profit"] = profit
                    logging.info(f"Closed position with P/L: {profit}")
                    trader.position = None
                
            # Log performance
            total_profit = sum([t["profit"] for t in trade_history if "profit" in t])
            win_rate = len([t for t in trade_history if t["profit"] > 0]) / len(trade_history) if trade_history else 0
            logging.info(f"Total Profit: {total_profit:.2f} | Win Rate: {win_rate:.2%}")
            
            # Sleep
            time.sleep(15)
        
        except Exception as e:
            logging.critical(f"FATAL ERROR: {e}")
            trader.close_all_positions()
            exit()

if __name__ == "__main__":
    main()
