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
    "leverage": int(os.getenv("LEVERAGE", 10)),
    "order_size": float(os.getenv("ORDER_SIZE", 0.01)),
    "short_ma": int(os.getenv("SHORT_MA", 13)),
    "long_ma": int(os.getenv("LONG_MA", 21)),
    "timeframe": os.getenv("TIMEFRAME", "1m"),
    "threshold_percent": float(os.getenv("THRESHOLD_PERCENT", 0.05)),
    "testnet": os.getenv("TESTNET", "1") == "1"
}

BASE_URL = "https://www.okx.com"
if CONFIG["testnet"]:
    BASE_URL = "https://www.okx.com"  # OKX menggunakan domain yang sama untuk testnet dengan header berbeda

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
                "OK-ACCESS-PASSPHRASE": CONFIG["passphrase"],
                "x-simulated-trading": "1" if CONFIG["testnet"] else "0"
            }
            
            url = BASE_URL + endpoint
            response = self.session.request(
                method,
                url,
                headers=headers,
                json=body if body else None,
                timeout=10
            )
            
            # Debugging response
            logging.debug(f"Response status: {response.status_code}")
            logging.debug(f"Response text: {response.text}")
            
            response.raise_for_status()
            
            try:
                return response.json()
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response: {response.text}")
                return {"error": "Invalid JSON response"}
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {str(e)}")
            logging.error(f"Response content: {e.response.text if hasattr(e, 'response') else 'No response'}")
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
        else:
            logging.info(f"Order executed: {side}/{pos_side}")
        return response

    def get_positions(self):
        endpoint = "/api/v5/account/positions"
        response = self._send_request("GET", endpoint)
        if "error" in response:
            logging.error(f"Position check failed: {response['error']}")
            return []
        return response.get("data", [])

    def close_all_positions(self):
        positions = self.get_positions()
        if not positions:
            logging.info("No positions to close")
            return
            
        for pos in positions:
            if pos["instId"] == CONFIG["symbol"] and float(pos.get("pos", 0)) > 0:
                side = "buy" if pos["posSide"] == "short" else "sell"
                self.place_order(side, pos["posSide"])

    def run_strategy(self):
        # Initial setup
        self.set_leverage()
        last_signal = None
        
        # Main trading loop
        while True:
            try:
                # Check balance
                balance = self.get_balance()
                required = CONFIG["order_size"] * CONFIG["leverage"]
                
                if balance < required:
                    logging.warning(f"Insufficient balance: {balance} USDT (needed: {required})")
                    time.sleep(300)
                    continue
                
                # Get market data
                prices = self.get_prices()
                if len(prices) < CONFIG["long_ma"]:
                    logging.warning(f"Insufficient data: {len(prices)}/{CONFIG['long_ma']}")
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                short_ma = self.moving_average(prices, CONFIG["short_ma"])
                long_ma = self.moving_average(prices, CONFIG["long_ma"])
                
                if None in [short_ma, long_ma]:
                    time.sleep(60)
                    continue
                
                logging.info(f"MA Values - Short: {short_ma:.2f}, Long: {long_ma:.2f}")
                
                # Trading logic
                threshold = CONFIG["threshold_percent"] / 100 * long_ma
                
                if (short_ma - long_ma) > threshold and last_signal != "short":
                    logging.info("SHORT signal detected")
                    self.close_all_positions()
                    self.place_order("sell", "short")
                    last_signal = "short"
                    
                elif (long_ma - short_ma) > threshold and last_signal != "long":
                    logging.info("LONG signal detected")
                    self.close_all_positions()
                    self.place_order("buy", "long")
                    last_signal = "long"
                    
                time.sleep(60)
                
            except KeyboardInterrupt:
                logging.info("Bot stopped by user")
                self.close_all_positions()
                break
            except Exception as e:
                logging.error(f"Error in strategy loop: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    logging.info("Starting OKX Trading Bot")
    try:
        trader = OKXTrader()
        trader.run_strategy()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    finally:
        logging.info("Bot stopped")
