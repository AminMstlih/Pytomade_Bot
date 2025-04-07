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
        # Validasi environment variables
        required_vars = ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {missing_vars}")

        self.config = {
            "api_key": os.getenv("OKX_API_KEY"),
            "secret_key": os.getenv("OKX_SECRET_KEY"),
            "passphrase": os.getenv("OKX_PASSPHRASE"),
            "symbol": os.getenv("SYMBOL", "BTC-USDT-SWAP"),
            "leverage": int(os.getenv("LEVERAGE", 10)),
            "position_size": float(os.getenv("POSITION_SIZE", 0.01)),
            "testnet": os.getenv("TESTNET", "1") == "1",
            "base_url": "https://www.okx.com",
            "max_retries": 3
        }

        # Validasi kredensial
        if not all([self.config["api_key"], self.config["secret_key"], self.config["passphrase"]]):
            raise ValueError("API credentials are incomplete")

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-simulated-trading": "1" if self.config["testnet"] else "0"
        })

        # Test koneksi API
        if not self._test_api_connection():
            raise ConnectionError("Failed to connect to OKX API")

    def _test_api_connection(self):
        """Test koneksi awal ke API"""
        try:
            response = self._send_request("GET", "/api/v5/account/balance")
            return "data" in response
        except Exception as e:
            logging.error(f"API connection test failed: {str(e)}")
            return False

    def _generate_signature(self, timestamp, method, request_path, body=""):
        """Generate signature dengan format yang benar"""
        try:
            if not isinstance(self.config["secret_key"], str):
                raise ValueError("Secret key must be a string")
                
            message = timestamp + method.upper() + request_path + body
            mac = hmac.new(
                bytes(self.config["secret_key"], 'utf-8'),
                bytes(message, 'utf-8'),
                hashlib.sha256
            )
            return base64.b64encode(mac.digest()).decode('utf-8')
        except Exception as e:
            logging.error(f"Signature generation failed: {str(e)}")
            raise

    def _send_request(self, method, endpoint, body=None, retry=0):
        """Improved request handler dengan debugging"""
        try:
            timestamp = str(time.time())
            body_json = json.dumps(body) if body else ""

            # Debugging info
            logging.debug(f"Preparing {method} request to {endpoint}")
            logging.debug(f"Timestamp: {timestamp}")
            logging.debug(f"Body: {body_json}")

            headers = {
                "OK-ACCESS-KEY": self.config["api_key"],
                "OK-ACCESS-SIGN": self._generate_signature(timestamp, method, endpoint, body_json),
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.config["passphrase"],
                "x-simulated-trading": "1" if self.config["testnet"] else "0"
            }

            # Debugging headers (without sensitive info)
            logging.debug(f"Headers: { {k: v for k, v in headers.items() if k != 'OK-ACCESS-SIGN'} }")

            url = self.config["base_url"] + endpoint
            response = self.session.request(
                method,
                url,
                headers=headers,
                json=body if method == "POST" else None,
                params=body if method == "GET" else None,
                timeout=10
            )

            # Debugging response
            logging.debug(f"Response status: {response.status_code}")
            logging.debug(f"Response headers: {response.headers}")
            logging.debug(f"Response text: {response.text[:200]}...")  # Log partial response

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                logging.error("Authentication failed. Please check:")
                logging.error("- API Key permissions")
                logging.error("- IP whitelisting")
                logging.error("- Timestamp synchronization")
                logging.error(f"Server time: {response.headers.get('Date')}")
                logging.error(f"Local time: {time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())}")
            if retry < self.config["max_retries"]:
                wait_time = 2 ** retry
                logging.warning(f"Retry {retry+1} for {endpoint} after {wait_time}s")
                time.sleep(wait_time)
                return self._send_request(method, endpoint, body, retry+1)
            logging.error(f"Request failed after retries: {str(e)}")
            logging.error(f"Response content: {e.response.text if hasattr(e, 'response') else 'No response'}")
            return {"error": str(e)}
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return {"error": str(e)}

    # ... (fungsi-fungsi lainnya tetap sama seperti sebelumnya) ...

if __name__ == "__main__":
    try:
        logging.info("Starting OKX Trading Bot with enhanced authentication")
        trader = QuantOKXTrader()
        
        # Test balance check
        balance = trader.get_account_balance()
        if balance is not None:
            logging.info(f"Initial balance check successful: {balance} USDT")
            trader.run()
        else:
            logging.error("Failed to verify account balance. Check API permissions.")
    except Exception as e:
        logging.error(f"Failed to initialize bot: {str(e)}")
