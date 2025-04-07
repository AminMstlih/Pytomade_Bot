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

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "api_key": os.getenv("OKX_API_KEY"),
    "secret_key": os.getenv("OKX_SECRET_KEY"),
    "passphrase": os.getenv("OKX_PASSPHRASE"),
    "symbol": os.getenv("TRADING_SYMBOL", "BTC-USDT-SWAP"),
    "leverage": int(os.getenv("LEVERAGE", 15)),
    "order_size": float(os.getenv("ORDER_SIZE", 0.1)),
    "short_ma": int(os.getenv("SHORT_MA", 13)),
    "long_ma": int(os.getenv("LONG_MA", 21)),
    "timeframe": os.getenv("TIMEFRAME", "1m"),
    "threshold_percent": float(os.getenv("THRESHOLD_PERCENT", 0.05)),
    "testnet": os.getenv("TESTNET", "1") == "1"
}

BASE_URL = "https://www.okx.com"

class OKXTrader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-simulated-trading": "1" if CONFIG["testnet"] else "0"
        })
        
    def _generate_signature(self, timestamp, method, request_path, body=""):
        message = f"{timestamp}{method}{request_path}{body}"
        mac = hmac.new(
            CONFIG["secret_key"].encode(),
            message.encode(),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()

    def _send_request(self, method, endpoint, body=None):
        try:
            timestamp = self._get_server_time()
            body_json = json.dumps(body) if body else ""
            
            headers = {
                "OK-ACCESS-KEY": CONFIG["api_key"],
                "OK-ACCESS-SIGN": self._generate_signature(timestamp, method, endpoint, body_json),
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": CONFIG["passphrase"]
            }
            
            url = BASE_URL + endpoint
            response = self.session.request(
                method,
                url,
                headers=headers,
                json=body if body else None,
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return {"error": str(e)}

    def _get_server_time(self):
        endpoint = "/api/v5/public/time"
        try:
            response = requests.get(BASE_URL + endpoint, timeout=5)
            response.raise_for_status()
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                return str(float(data["data"][0]["ts"]) / 1000.0)
            raise ValueError("Invalid time response structure")
        except Exception as e:
            logging.warning(f"Failed to get server time, using local time: {str(e)}")
            return str(time.time())

    def set_leverage(self):
        endpoint = "/api/v5/account/set-leverage"
        body = {
            "instId": CONFIG["symbol"],
            "lever": str(CONFIG["leverage"]),
            "mgnMode": "cross"
        }
        response = self._send_request("POST", endpoint, body)
        if "error" in response:
            logging.error(f"Failed to set leverage: {response['error']}")
        else:
            logging.info(f"Leverage set successfully: {response}")
        return response

    def get_balance(self, ccy="USDT"):
        endpoint = "/api/v5/account/balance"
        response = self._send_request("GET", endpoint)
        
        if "error" in response:
            logging.error(f"Balance check failed: {response['error']}")
            return 0.0
            
        if "data" in response and response["data"]:
            for item in response["data"]:
                if "details" in item:
                    for detail in item["details"]:
                        if detail.get("ccy") == ccy:
                            return float(detail.get("availBal", 0))
                elif item.get("ccy") == ccy:
                    return float(item.get("availBal", 0))
                    
        logging.warning("USDT balance not found in response")
        return 0.0

    def get_prices(self):
        endpoint = f"/api/v5/market/candles?instId={CONFIG['symbol']}&bar={CONFIG['timeframe']}&limit={CONFIG['long_ma']}"
        response = self._send_request("GET", endpoint)
        
        if "error" in response:
            logging.error(f"Failed to get prices: {response['error']}")
            return []
            
        if "data" in response and len(response["data"]) >= CONFIG["long_ma"]:
            try:
                return [float(candle[4]) for candle in response["data"]]
            except (IndexError, ValueError) as e:
                logging.error(f"Error parsing prices: {str(e)}")
                return []
        return []

    def moving_average(self, prices, period):
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def place_order(self, side, pos_side):
        endpoint = "/api/v5/trade/order"
        order_data = {
            "instId": CONFIG["symbol"],
            "tdMode": "cross",
            "side": side,
            "posSide": pos_side,
            "ordType": "market",
            "sz": str(CONFIG["order_size"])
        }
        response = self._send_request("POST", endpoint, order_data)
        if "error" in response:
            logging.error(f"Order failed: {response['error']}")
            return False
        else:
            logging.info(f"Order executed: {side}/{pos_side}")
            return True

    def get_positions(self):
        endpoint = "/api/v5/account/positions"
        response = self._send_request("GET", endpoint)
        if "error" in response:
            logging.error(f"Position check failed: {response['error']}")
            return []
        return [p for p in response.get("data", []) if p["instId"] == CONFIG["symbol"]]

    def close_all_positions(self):
        positions = self.get_positions()
        if not positions:
            logging.info("No positions to close")
            return True
            
        success = True
        for pos in positions:
            if float(pos.get("pos", 0)) > 0:
                side = "buy" if pos["posSide"] == "short" else "sell"
                if not self.place_order(side, pos["posSide"]):
                    success = False
        return success

    def get_current_signal(self):
        prices = self.get_prices()
        if len(prices) < CONFIG["long_ma"]:
            logging.error("Not enough price data")
            return None
            
        short_ma = self.moving_average(prices, CONFIG["short_ma"])
        long_ma = self.moving_average(prices, CONFIG["long_ma"])
        
        if short_ma is None or long_ma is None:
            return None
            
        threshold = CONFIG["threshold_percent"] / 100 * long_ma
        
        if (short_ma - long_ma) > threshold:
            return "short"
        elif (long_ma - short_ma) > threshold:
            return "long"
        return None

    def run_strategy(self):
        # Set leverage first
        self.set_leverage()
        
        # Check existing positions
        positions = self.get_positions()
        has_position = any(float(p["pos"]) > 0 for p in positions)
        
        # If no position, open immediately based on current MA
        if not has_position:
            logging.info("No existing positions found, opening initial position...")
            signal = self.get_current_signal()
            
            if signal == "short":
                if self.place_order("sell", "short"):
                    logging.info("Initial SHORT position opened")
            elif signal == "long":
                if self.place_order("buy", "long"):
                    logging.info("Initial LONG position opened")
            else:
                logging.info("No clear signal for initial position")
        
        # Main trading loop
        last_signal = None
        while True:
            try:
                # Check balance
                balance = self.get_balance()
                required = CONFIG["order_size"] * CONFIG["leverage"]
                
                if balance < required:
                    logging.warning(f"Insufficient balance: {balance} USDT (needed: {required})")
                    time.sleep(300)
                    continue
                
                # Get current signal
                signal = self.get_current_signal()
                if signal is None:
                    time.sleep(60)
                    continue
                
                # Only act if signal changed
                if signal != last_signal:
                    logging.info(f"Signal changed to: {signal.upper()}")
                    if self.close_all_positions():
                        if signal == "short":
                            self.place_order("sell", "short")
                        elif signal == "long":
                            self.place_order("buy", "long")
                    last_signal = signal
                
                time.sleep(60)
                
            except KeyboardInterrupt:
                logging.info("Bot stopped by user")
                self.close_all_positions()
                break
            except Exception as e:
                logging.error(f"Error in strategy loop: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    logging.info("Starting OKX Trading Bot with Immediate Entry")
    try:
        trader = OKXTrader()
        trader.run_strategy()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    finally:
        logging.info("Bot stopped")
