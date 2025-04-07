import time
import datetime
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
from typing import Optional, Dict, List

# Configure professional logging
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
        # Validate credentials first
        self._validate_credentials()
        
        self.config = {
            "api_key": os.getenv("OKX_API_KEY"),
            "secret_key": os.getenv("OKX_SECRET_KEY"),
            "passphrase": os.getenv("OKX_PASSPHRASE"),
            "symbol": os.getenv("SYMBOL", "BTC-USDT-SWAP"),
            "leverage": int(os.getenv("LEVERAGE", 10)),
            "risk_per_trade": float(os.getenv("RISK_PER_TRADE", 0.01)),  # 1% risk per trade
            "testnet": os.getenv("TESTNET", "1") == "1",
            "base_url": "https://www.okex.com" if os.getenv("TESTNET") == "1" else "https://www.okx.com",
            "max_retries": 3,
            "timeout": 10
        }

        # Professional strategy parameters
        self.params = {
            "trend_ema": (50, 200),        # Fast, Slow EMAs for trend
            "entry_sma": (13, 21),         # Short, Long SMAs for entries
            "rsi_period": 14,
            "atr_period": 14,              # For stop loss placement
            "vwap_period": 20,             # For execution quality
            "max_daily_trades": 5,         # Prevent over-trading
            "daily_stop_loss": -0.05,      # -5% daily stop
            "daily_take_profit": 0.03      # +3% daily target
        }

        # Trading session state
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-simulated-trading": "1" if self.config["testnet"] else "0"
        })

        # Risk management state
        self.today = datetime.date.today()
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.current_position = None

        # Initialize
        self._setup()

    def _validate_credentials(self):
        """Validate all required credentials exist"""
        required = ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"]
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {missing}")

    def _setup(self):
        """Initialize trading session"""
        # Verify API connectivity
        if not self._test_connection():
            raise ConnectionError("API connection failed - check credentials/IP whitelist")

        # Set account configuration
        self._set_leverage()
        self._set_position_mode()

        # Load initial market data
        self._warmup_strategy()

    def _test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            response = self._send_request("GET", "/api/v5/account/balance")
            return "data" in response
        except Exception as e:
            logging.error(f"Connection test failed: {e}")
            return False

    def _send_request(self, method: str, endpoint: str, body=None, retry=0):
        """Professional-grade request handler"""
        try:
            # Get timestamp in ISO 8601 format
            timestamp = self._get_timestamp()
            
            # Prepare signature
            body_json = json.dumps(body) if body else ""
            signature = self._generate_signature(timestamp, method, endpoint, body_json)
            
            headers = {
                "OK-ACCESS-KEY": self.config["api_key"],
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.config["passphrase"],
                "x-simulated-trading": "1" if self.config["testnet"] else "0"
            }

            # Send request
            response = self.session.request(
                method,
                self.config["base_url"] + endpoint,
                headers=headers,
                json=body if method == "POST" else None,
                params=body if method == "GET" else None,
                timeout=self.config["timeout"]
            )

            # Handle 401 errors specifically
            if response.status_code == 401:
                error_data = response.json()
                if error_data.get("code") == "50102":  # Timestamp expired
                    logging.warning("Timestamp expired - retrying with fresh timestamp")
                    return self._send_request(method, endpoint, body, retry+1)
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

    def _get_timestamp(self) -> str:
        """Get timestamp in ISO 8601 format"""
        return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC SHA256 signature"""
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            self.config["secret_key"].encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _set_leverage(self):
        """Set leverage for the instrument"""
        endpoint = "/api/v5/account/set-leverage"
        body = {
            "instId": self.config["symbol"],
            "lever": str(self.config["leverage"]),
            "mgnMode": "cross"
        }
        response = self._send_request("POST", endpoint, body)
        logging.info(f"Leverage set: {response}")

    def _set_position_mode(self):
        """Set position mode to net or long/short"""
        endpoint = "/api/v5/account/set-position-mode"
        body = {
            "posMode": "long_short_mode"
        }
        response = self._send_request("POST", endpoint, body)
        logging.info(f"Position mode set: {response}")

    def _warmup_strategy(self):
        """Load initial market data for strategy"""
        logging.info("Loading initial market data...")
        # Load data for all timeframes
        self.trend_data = self._get_candles("1H", 300)
        self.signal_data = self._get_candles("15m", 200)
        self.execution_data = self._get_candles("1m", 100)
        
        # Calculate initial indicators
        self._calculate_indicators()
        logging.info("Strategy warmup complete")

    def _get_candles(self, timeframe: str, limit: int) -> pd.DataFrame:
        """Get candle data with professional error handling"""
        endpoint = "/api/v5/market/candles"
        params = {
            "instId": self.config["symbol"],
            "bar": timeframe,
            "limit": limit
        }
        
        try:
            response = self._send_request("GET", endpoint, params)
            if "data" not in response:
                raise ValueError("No data in response")
                
            df = pd.DataFrame(response["data"], columns=[
                "timestamp", "open", "high", "low", "close", "volume", 
                "volCcy", "volCcyQuote", "confirm"
            ])
            
            # Convert types
            numeric_cols = ["open", "high", "low", "close", "volume", "volCcy", "volCcyQuote"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            return df.sort_values("timestamp").dropna()
            
        except Exception as e:
            logging.error(f"Failed to get candles: {e}")
            raise

    def _calculate_indicators(self):
        """Calculate all technical indicators"""
        # Trend indicators (1H)
        self.trend_data["ema_fast"] = self.trend_data["close"].ewm(
            span=self.params["trend_ema"][0], adjust=False).mean()
        self.trend_data["ema_slow"] = self.trend_data["close"].ewm(
            span=self.params["trend_ema"][1], adjust=False).mean()
        
        # Signal indicators (15m)
        self.signal_data["sma_fast"] = self.signal_data["close"].rolling(
            self.params["entry_sma"][0]).mean()
        self.signal_data["sma_slow"] = self.signal_data["close"].rolling(
            self.params["entry_sma"][1]).mean()
        
        # Execution indicators (1m)
        typical_price = (self.execution_data["high"] + self.execution_data["low"] + self.execution_data["close"]) / 3
        self.execution_data["vwap"] = (typical_price * self.execution_data["volume"]).cumsum() / self.execution_data["volume"].cumsum()
        
        # Calculate ATR for risk management
        high_low = self.execution_data["high"] - self.execution_data["low"]
        high_close = (self.execution_data["high"] - self.execution_data["close"].shift()).abs()
        low_close = (self.execution_data["low"] - self.execution_data["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.execution_data["atr"] = true_range.rolling(self.params["atr_period"]).mean()

    def _get_position_size(self) -> float:
        """Calculate professional position sizing based on volatility"""
        atr = self.execution_data["atr"].iloc[-1]
        price = self.execution_data["close"].iloc[-1]
        account_balance = self._get_account_balance()
        
        if not account_balance or not atr or not price:
            return self.config.get("default_size", 0.01)
            
        risk_amount = account_balance * self.config["risk_per_trade"]
        position_size = risk_amount / (atr * 1.5)  # Use 1.5x ATR as risk buffer
        
        # Convert to contract size
        return round(position_size / price, 4)

    def _get_account_balance(self) -> Optional[float]:
        """Get available USDT balance"""
        endpoint = "/api/v5/account/balance"
        try:
            response = self._send_request("GET", endpoint)
            if "data" not in response:
                return None
                
            for currency in response["data"][0]["details"]:
                if currency["ccy"] == "USDT":
                    return float(currency["availBal"])
            return None
        except Exception as e:
            logging.error(f"Failed to get balance: {e}")
            return None

    def _check_daily_limits(self) -> bool:
        """Check if we've hit daily trading limits"""
        if datetime.date.today() != self.today:
            # Reset for new trading day
            self.today = datetime.date.today()
            self.daily_pnl = 0.0
            self.trades_today = 0
            return True
            
        if self.trades_today >= self.params["max_daily_trades"]:
            logging.warning("Max daily trades reached")
            return False
            
        if self.daily_pnl <= self.params["daily_stop_loss"]:
            logging.warning("Daily stop loss hit")
            return False
            
        if self.daily_pnl >= self.params["daily_take_profit"]:
            logging.warning("Daily take profit hit")
            return False
            
        return True

    def _get_trade_signal(self) -> Optional[str]:
        """Generate professional trading signal"""
        # Check trend first
        last_trend = self.trend_data.iloc[-1]
        if last_trend["ema_fast"] > last_trend["ema_slow"]:
            trend = "bullish"
        elif last_trend["ema_fast"] < last_trend["ema_slow"]:
            trend = "bearish"
        else:
            return None  # No clear trend
            
        # Check entry signals
        last_signal = self.signal_data.iloc[-1]
        if trend == "bullish" and last_signal["sma_fast"] > last_signal["sma_slow"]:
            return "long"
        elif trend == "bearish" and last_signal["sma_fast"] < last_signal["sma_slow"]:
            return "short"
            
        return None

    def _execute_trade(self, signal: str):
        """Execute trade with professional order handling"""
        # Get current price and position size
        price = self.execution_data["close"].iloc[-1]
        size = self._get_position_size()
        
        if not size or size <= 0:
            logging.error("Invalid position size")
            return False
            
        # Place order
        endpoint = "/api/v5/trade/order"
        order_data = {
            "instId": self.config["symbol"],
            "tdMode": "cross",
            "side": "buy" if signal == "long" else "sell",
            "posSide": signal,
            "ordType": "market",
            "sz": str(size),
            "reduceOnly": False
        }
        
        try:
            response = self._send_request("POST", endpoint, order_data)
            if "data" in response and response["data"][0]["sCode"] == "0":
                self.current_position = signal
                self.trades_today += 1
                logging.info(f"Executed {signal} position of {size} contracts")
                return True
                
            logging.error(f"Order failed: {response}")
            return False
        except Exception as e:
            logging.error(f"Trade execution failed: {e}")
            return False

    def _manage_position(self):
        """Professional position management"""
        if not self.current_position:
            return
            
        # Check exit conditions
        current_price = self.execution_data["close"].iloc[-1]
        atr = self.execution_data["atr"].iloc[-1]
        
        # Update stop loss/take profit logic here
        # This is where you'd implement trailing stops, etc.
        
        # For now, we'll just check if we should reverse position
        signal = self._get_trade_signal()
        if signal and signal != self.current_position:
            logging.info(f"Reversing position from {self.current_position} to {signal}")
            self._execute_trade(signal)

    def run(self):
        """Main trading loop"""
        logging.info("Starting professional trading bot")
        
        while True:
            try:
                # Check if we can trade today
                if not self._check_daily_limits():
                    time.sleep(60)
                    continue
                    
                # Refresh market data
                self._warmup_strategy()
                
                # Manage existing position
                self._manage_position()
                
                # Get new signal if no position
                if not self.current_position:
                    signal = self._get_trade_signal()
                    if signal:
                        self._execute_trade(signal)
                
                # Sleep until next cycle
                time.sleep(30)
                
            except KeyboardInterrupt:
                logging.info("Shutting down gracefully...")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    try:
        bot = ProfessionalOKXTrader()
        bot.run()
    except Exception as e:
        logging.error(f"Failed to start bot: {e}")
