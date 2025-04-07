import time
import hmac
import hashlib
import base64
import requests
import numpy as np
import pandas as pd
import logging
import json
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('professional_bot.log'),
        logging.StreamHandler()
    ]
)

# Load environment
load_dotenv()

class ProfessionalOKXTrader:
    def __init__(self):
        self._validate_credentials()
        
        self.config = {
            "api_key": os.getenv("OKX_API_KEY"),
            "secret_key": os.getenv("OKX_SECRET_KEY"),
            "passphrase": os.getenv("OKX_PASSPHRASE"),
            "symbol": os.getenv("SYMBOL", "BTC-USDT-SWAP"),
            "leverage": int(os.getenv("LEVERAGE", 10)),
            "risk_per_trade": float(os.getenv("RISK_PER_TRADE", 0.01)),
            "testnet": os.getenv("TESTNET", "1") == "1",
            "base_url": "https://www.okx.com",
            "max_retries": 3,
            "timeout": 10
        }

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-simulated-trading": "1" if self.config["testnet"] else "0"
        })

        self._setup()

    def _validate_credentials(self):
        required = ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"]
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {missing}")

    def _setup(self):
        if not self._test_connection():
            raise ConnectionError("API connection failed")
        self._set_leverage()
        self._set_position_mode()

    def _test_connection(self) -> bool:
        try:
            response = self._send_request("GET", "/api/v5/account/balance")
            return "data" in response
        except Exception as e:
            logging.error(f"Connection test failed: {e}")
            return False

    def _get_server_time(self):
        """Get server time in milliseconds as string"""
        endpoint = "/api/v5/public/time"
        try:
            response = requests.get(
                self.config["base_url"] + endpoint,
                timeout=self.config["timeout"]
            )
            response.raise_for_status()
            return str(float(response.json()["data"][0]["ts"]) / 1000.0)  # Return timestamp as string
        except Exception as e:
            logging.error(f"Failed to get server time: {e}")
            return str(int(time.time() * 1000))  # Fallback to local time in ms

    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC SHA256 signature"""
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            self.config["secret_key"].encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _send_request(self, method: str, endpoint: str, body=None, retry=0):
        """Improved request handler with proper timestamp"""
        try:
            timestamp = self._get_server_time()
            body_json = json.dumps(body) if body else ""
            
            headers = {
                "OK-ACCESS-KEY": self.config["api_key"],
                "OK-ACCESS-SIGN": self._generate_signature(timestamp, method, endpoint, body_json),
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.config["passphrase"],
                "x-simulated-trading": "1" if self.config["testnet"] else "0"
            }

            response = self.session.request(
                method,
                self.config["base_url"] + endpoint,
                headers=headers,
                json=body if method == "POST" else None,
                params=body if method == "GET" else None,
                timeout=self.config["timeout"]
            )

            if response.status_code == 401:
                error_data = response.json()
                if error_data.get("code") == "50102":
                    if retry < self.config["max_retries"]:
                        logging.warning(f"Timestamp expired - retry {retry+1}")
                        time.sleep(1)
                        return self._send_request(method, endpoint, body, retry+1)
                    raise PermissionError("Timestamp expired after retries")
                raise PermissionError(f"API authentication failed: {error_data}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if retry < self.config["max_retries"]:
                wait_time = 2 ** retry
                logging.warning(f"Retry {retry+1} for {endpoint} after {wait_time}s")
                time.sleep(wait_time)
                return self._send_request(method, endpoint, body, retry+1)
            raise
        except Exception as e:
            logging.error(f"Request failed: {e}")
            raise

    def _set_leverage(self):
        endpoint = "/api/v5/account/set-leverage"
        body = {
            "instId": self.config["symbol"],
            "lever": str(self.config["leverage"]),
            "mgnMode": "cross"
        }
        response = self._send_request("POST", endpoint, body)
        logging.info(f"Leverage set: {response}")

    def _set_position_mode(self):
        endpoint = "/api/v5/account/set-position-mode"
        body = {
            "posMode": "long_short_mode"
        }
        response = self._send_request("POST", endpoint, body)
        logging.info(f"Position mode set: {response}")

    # ... [rest of your class methods remain unchanged] ...

if __name__ == "__main__":
    try:
        bot = ProfessionalOKXTrader()
        bot.run()
    except Exception as e:
        logging.error(f"Failed to start bot: {e}")
