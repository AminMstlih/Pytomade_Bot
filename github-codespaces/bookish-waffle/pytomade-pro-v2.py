import os
import time
import hmac
import hashlib
import base64
import requests
import numpy as np
import pandas as pd
import logging
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_bot.log'),
        logging.StreamHandler()
    ]
)

# Load environment
load_dotenv()

class QuantOKXTrader:
    def __init__(self):
        self.config = {
            "api_key": os.getenv("OKX_API_KEY"),
            "secret_key": os.getenv("OKX_SECRET_KEY"),
            "passphrase": os.getenv("OKX_PASSPHRASE"),
            "symbol": os.getenv("SYMBOL", "BTC-USDT-SWAP"),
            "leverage": int(os.getenv("LEVERAGE", 10)),
            "position_size": float(os.getenv("POSITION_SIZE", 0.05)),
            "testnet": os.getenv("TESTNET", "1") == "1",
            "base_url": "https://www.okx.com",
            "max_retries": 3,
            "timeframes": {
                "trend": "1H",
                "signal": "15m",
                "execution": "1m"
            }
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-simulated-trading": "1" if self.config["testnet"] else "0"
        })
        
        # Strategy parameters
        self.params = {
            "ema_fast": 50,
            "ema_slow": 200,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "vwap_period": 20,
            "atr_period": 14,
            "atr_multiplier": 1.5
        }
        
        # State management
        self.current_position = None
        self.last_signal = None
        self.order_retries = 0

    # --- Core Trading Functions ---
    def _generate_signature(self, timestamp, method, request_path, body=""):
        message = f"{timestamp}{method}{request_path}{body}"
        mac = hmac.new(
            self.config["secret_key"].encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _send_request(self, method, endpoint, body=None, retry=0):
        try:
            timestamp = str(time.time())
            body_json = json.dumps(body) if body else ""
            
            headers = {
                "OK-ACCESS-KEY": self.config["api_key"],
                "OK-ACCESS-SIGN": self._generate_signature(timestamp, method, endpoint, body_json),
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.config["passphrase"]
            }
            
            response = self.session.request(
                method,
                self.config["base_url"] + endpoint,
                headers=headers,
                json=body if method == "POST" else None,
                params=body if method == "GET" else None,
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if retry < self.config["max_retries"]:
                wait_time = 2 ** retry
                logging.warning(f"Retry {retry+1} for {endpoint} after {wait_time}s")
                time.sleep(wait_time)
                return self._send_request(method, endpoint, body, retry+1)
            logging.error(f"Request failed after retries: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return {"error": str(e)}

    # --- Data Processing ---
    def get_candles(self, timeframe, limit=300):
        endpoint = f"/api/v5/market/candles"
        params = {
            "instId": self.config["symbol"],
            "bar": timeframe,
            "limit": limit
        }
        response = self._send_request("GET", endpoint, params)
        
        if "data" not in response:
            logging.error(f"Failed to get candles: {response}")
            return None
            
        df = pd.DataFrame(response["data"], columns=[
            "timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"
        ])
        
        # Convert types
        numeric_cols = ["open", "high", "low", "close", "volume", "volCcy", "volCcyQuote"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        return df.sort_values("timestamp")

    def calculate_indicators(self, df):
        # EMAs for trend
        df["ema_fast"] = df["close"].ewm(span=self.params["ema_fast"], adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.params["ema_slow"], adjust=False).mean()
        
        # MACD for signals
        exp1 = df["close"].ewm(span=self.params["macd_fast"], adjust=False).mean()
        exp2 = df["close"].ewm(span=self.params["macd_slow"], adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=self.params["macd_signal"], adjust=False).mean()
        
        # RSI for confirmation
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.params["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.params["rsi_period"]).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # VWAP for execution
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["cumulative_vwap"] = (df["typical_price"] * df["volume"]).cumsum()
        df["cumulative_volume"] = df["volume"].cumsum()
        df["vwap"] = df["cumulative_vwap"] / df["cumulative_volume"]
        
        # ATR for risk management
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(self.params["atr_period"]).mean()
        
        return df.dropna()

    # --- Trading Logic ---
    def determine_trend(self):
        """1H timeframe for market bias"""
        df = self.get_candles(self.config["timeframes"]["trend"], 300)
        if df is None:
            return None
            
        df = self.calculate_indicators(df)
        last_row = df.iloc[-1]
        
        # Bullish trend condition
        if last_row["ema_fast"] > last_row["ema_slow"] and last_row["close"] > last_row["ema_fast"]:
            return "bullish"
        # Bearish trend condition
        elif last_row["ema_fast"] < last_row["ema_slow"] and last_row["close"] < last_row["ema_fast"]:
            return "bearish"
        return "neutral"

    def generate_signal(self):
        """15m timeframe for entry signals"""
        df = self.get_candles(self.config["timeframes"]["signal"], 100)
        if df is None:
            return None
            
        df = self.calculate_indicators(df)
        last_row = df.iloc[-1]
        trend = self.determine_trend()
        
        # Long signal conditions
        long_cond = (
            (trend == "bullish") and
            (last_row["macd"] > last_row["macd_signal"]) and
            (last_row["rsi"] > 50 and last_row["rsi"] < self.params["rsi_overbought"])
        )
        
        # Short signal conditions
        short_cond = (
            (trend == "bearish") and
            (last_row["macd"] < last_row["macd_signal"]) and
            (last_row["rsi"] < 50 and last_row["rsi"] > self.params["rsi_oversold"])
        )
        
        if long_cond:
            return "long"
        elif short_cond:
            return "short"
        return None

    def execution_decision(self, signal):
        """1m timeframe for precise entry"""
        df = self.get_candles(self.config["timeframes"]["execution"], 50)
        if df is None:
            return False
            
        df = self.calculate_indicators(df)
        last_row = df.iloc[-1]
        
        # Confirm with volume and price action
        if signal == "long":
            return (
                (last_row["close"] > last_row["vwap"]) and
                (last_row["volume"] > df["volume"].rolling(5).mean().iloc[-1])
            )
        elif signal == "short":
            return (
                (last_row["close"] < last_row["vwap"]) and
                (last_row["volume"] > df["volume"].rolling(5).mean().iloc[-1])
            )
        return False

    # --- Order Management ---
    def get_position(self):
        endpoint = "/api/v5/account/positions"
        response = self._send_request("GET", endpoint)
        
        if "data" not in response:
            return None
            
        for pos in response["data"]:
            if pos["instId"] == self.config["symbol"] and float(pos["pos"]) > 0:
                return {
                    "side": "long" if pos["posSide"] == "long" else "short",
                    "size": float(pos["pos"]),
                    "entry_price": float(pos["avgPx"])
                }
        return None

    def place_order(self, side, size=None):
        if size is None:
            size = self.config["position_size"]
            
        endpoint = "/api/v5/trade/order"
        order_data = {
            "instId": self.config["symbol"],
            "tdMode": "cross",
            "side": "buy" if side == "long" else "sell",
            "posSide": side,
            "ordType": "market",
            "sz": str(size),
            "reduceOnly": False
        }
        
        response = self._send_request("POST", endpoint, order_data)
        if "data" in response:
            logging.info(f"Order executed: {side} {size}")
            return True
        logging.error(f"Order failed: {response}")
        return False

    def close_position(self):
        pos = self.get_position()
        if not pos:
            return True
            
        endpoint = "/api/v5/trade/close-position"
        order_data = {
            "instId": self.config["symbol"],
            "mgnMode": "cross",
            "posSide": pos["side"]
        }
        
        response = self._send_request("POST", endpoint, order_data)
        if "data" in response:
            logging.info(f"Position closed: {pos['side']}")
            return True
        logging.error(f"Close position failed: {response}")
        return False

    def calculate_position_size(self):
        """Risk-managed position sizing based on ATR"""
        df = self.get_candles(self.config["timeframes"]["signal"], 50)
        if df is None:
            return self.config["position_size"]
            
        df = self.calculate_indicators(df)
        atr = df["atr"].iloc[-1]
        price = df["close"].iloc[-1]
        
        # Risk 1% of capital per trade (simplified)
        risk_percent = 0.01
        account_balance = self.get_account_balance()
        if account_balance is None:
            return self.config["position_size"]
            
        risk_amount = account_balance * risk_percent
        position_size = risk_amount / (atr * self.params["atr_multiplier"])
        
        # Convert to contract size
        return round(position_size / price, 4)

    def get_account_balance(self):
        endpoint = "/api/v5/account/balance"
        response = self._send_request("GET", endpoint)
        
        if "data" not in response:
            return None
            
        for currency in response["data"][0]["details"]:
            if currency["ccy"] == "USDT":
                return float(currency["availBal"])
        return None

    # --- Main Strategy Loop ---
    def run(self):
        logging.info("Starting Quant Trading Bot")
        
        # Initial setup
        self.current_position = self.get_position()
        
        while True:
            try:
                # Step 1: Determine market trend
                trend = self.determine_trend()
                if not trend:
                    time.sleep(60)
                    continue
                
                # Step 2: Generate trading signal
                signal = self.generate_signal()
                if not signal:
                    time.sleep(30)
                    continue
                
                # Step 3: Check execution conditions
                should_execute = self.execution_decision(signal)
                
                # Step 4: Manage positions
                if self.current_position:
                    if self.current_position["side"] != signal:
                        logging.info(f"Signal reversal detected: {self.current_position['side']} -> {signal}")
                        if self.close_position():
                            self.current_position = None
                
                # Step 5: Enter new position
                if not self.current_position and should_execute:
                    position_size = self.calculate_position_size()
                    if self.place_order(signal, position_size):
                        self.current_position = {
                            "side": signal,
                            "size": position_size,
                            "entry_time": time.time()
                        }
                
                time.sleep(15)
                
            except KeyboardInterrupt:
                logging.info("Stopping bot...")
                self.close_position()
                break
            except Exception as e:
                logging.error(f"Strategy error: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    trader = QuantOKXTrader()
    trader.run()
