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
from statsmodels.tsa.arima.model import ARIMA

# Load API credentials from .env file
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# Global Variables
BASE_URL = "https://www.okx.com"
SYMBOL = "BTC-USDT-SWAP"  # Change token ticker here
LEVERAGE = 2  # Change leverage here
ORDER_SIZE = 100  # Change order size here
SHORT_MA = 21  # Short Moving Average
LONG_MA = 11  # Long Moving Average

# Set up logging to track bot activity
logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function to get server time from OKX
def get_server_time():
    """
    Get OKX server time for API signature.
    """
    endpoint = "/api/v5/public/time"
    try:
        response = requests.get(BASE_URL + endpoint)
        response.raise_for_status()
        return str(float(response.json()["data"][0]["ts"]) / 1000.0)
    except Exception as e:
        logging.error(f"Failed to get server time: {e}")
        return None

# Create HMAC SHA256 signature for API requests
def generate_signature(timestamp, method, request_path, body=""):
    """
    Generate HMAC SHA256 signature for authentication.
    """
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

# Function to send requests to OKX
def send_request(method, endpoint, body=None):
    """
    Send authenticated requests to OKX API.
    """
    try:
        timestamp = get_server_time()
        if not timestamp:
            raise ValueError("Server time is unavailable.")
        body_json = json.dumps(body) if body else ""
        headers = {
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": generate_signature(timestamp, method, endpoint, body_json),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/json",
            "x-simulated-trading": "1"  # Simulated trading mode
        }
        url = BASE_URL + endpoint
        response = requests.request(method, url, headers=headers, data=body_json)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {"error": str(e)}

# Set leverage for trading account
def set_leverage(leverage=LEVERAGE):
    """
    Set leverage for the trading account.
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

# Get real-time price
def get_realtime_price():
    """
    Get the latest (real-time) price from the ticker.
    """
    endpoint = f"/api/v5/market/ticker?instId={SYMBOL}"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        return float(response["data"][0]["last"])
    logging.warning("Failed to get real-time price.")
    return None

# Get historical prices (1-minute candles)
def get_prices():
    """
    Get historical candlestick data for strategy.
    """
    endpoint = f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit={max(SHORT_MA, LONG_MA)}"  # 1-minute candles
    response = send_request("GET", endpoint)
    if "data" in response:
        return [float(candle[4]) for candle in response["data"]]  # Close prices
    logging.warning("Failed to get historical prices.")
    return []

# Calculate Simple Moving Average (SMA)
def moving_average(prices, period):
    """
    Calculate SMA from prices.
    """
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

# ARIMA Prediction
def arima_prediction(prices):
    """
    Predict future price using ARIMA model.
    """
    try:
        model = ARIMA(prices, order=(5, 1, 0))  # ARIMA(p, d, q)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]
        return forecast
    except Exception as e:
        logging.error(f"ARIMA prediction failed: {e}")
        return None

# MACD Signal
def macd_signal(prices):
    """
    Calculate MACD signal for trading decision.
    """
    try:
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return "long" if macd.iloc[-1] > signal.iloc[-1] else "short"
    except Exception as e:
        logging.error(f"MACD calculation failed: {e}")
        return None

# Bollinger Bands
def bollinger_bands(prices, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    """
    try:
        if len(prices) < window:
            return None, None, None
        rolling_mean = pd.Series(prices).rolling(window=window).mean()
        rolling_std = pd.Series(prices).rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return rolling_mean.iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1]
    except Exception as e:
        logging.error(f"Bollinger Bands calculation failed: {e}")
        return None, None, None

# Chaikin Money Flow (CMF)
def chaikin_money_flow(candles, window=20):
    """
    Calculate Chaikin Money Flow.
    """
    try:
        if len(candles) < window:
            return None
        high = np.array([float(candle[2]) for candle in candles])  # High prices
        low = np.array([float(candle[3]) for candle in candles])   # Low prices
        close = np.array([float(candle[4]) for candle in candles]) # Close prices
        volume = np.array([float(candle[5]) for candle in candles]) # Volume
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        cmf = money_flow_volume[-window:].sum() / volume[-window:].sum()
        return cmf
    except Exception as e:
        logging.error(f"Chaikin Money Flow calculation failed: {e}")
        return None

# On-Balance Volume (OBV)
def on_balance_volume(candles):
    """
    Calculate On-Balance Volume.
    """
    try:
        if len(candles) < 2:
            return None
        close = np.array([float(candle[4]) for candle in candles]) # Close prices
        volume = np.array([float(candle[5]) for candle in candles]) # Volume
        obv = [volume[0]]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i - 1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        return obv[-1]
    except Exception as e:
        logging.error(f"OBV calculation failed: {e}")
        return None

# Place market order
def place_order(side, pos_side, order_size=ORDER_SIZE):
    """
    Place a market order without stop-loss or take-profit.
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

# Check open positions
def check_open_positions():
    """
    Check if there are any open positions.
    """
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    if "data" in response and response["data"]:
        return True
    return False

# Close all positions
def close_all_positions():
    """
    Close all open positions.
    """
    if check_open_positions():
        logging.info("Closing all positions...")
        place_order("buy", "short")  # Close short position if exists
        place_order("sell", "long")  # Close long position if exists
    else:
        logging.info("No open positions to close.")

# Track performance metrics
total_profit = 0
successful_trades = 0
failed_trades = 0

# Trading strategy: ARIMA, MA, MACD, Bollinger Bands, CMF, and OBV Voting System
def combined_strategy():
    """
    Trading strategy based on ARIMA, Moving Average, MACD, Bollinger Bands, CMF, and OBV voting system.
    """
    global total_profit, successful_trades, failed_trades
    set_leverage(LEVERAGE)  # Ensure leverage is set
    position = None

    while True:
        # Fetch data
        candles = send_request("GET", f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit={max(SHORT_MA, LONG_MA)}")["data"]
        if not candles:
            logging.warning("Failed to fetch candlestick data, retrying...")
            time.sleep(15)
            continue

        prices = [float(candle[4]) for candle in candles]  # Close prices
        current_price = float(candles[0][4])  # Latest close price

        # ARIMA Prediction
        predicted_price = arima_prediction(prices)
        arima_vote = "long" if predicted_price > current_price else "short"

        # Moving Average Signal
        short_ma = moving_average(prices, SHORT_MA)
        long_ma = moving_average(prices, LONG_MA)
        ma_vote = "short" if short_ma > long_ma else "long"

        # MACD Signal
        macd_vote = macd_signal(prices)

        # Bollinger Bands Signal
        rolling_mean, upper_band, lower_band = bollinger_bands(prices)
        bb_vote = None
        if rolling_mean is not None and upper_band is not None and lower_band is not None:
            if current_price > upper_band:
                bb_vote = "short"
            elif current_price < lower_band:
                bb_vote = "long"
            else:
                bb_vote = "neutral"

        # Chaikin Money Flow Signal
        cmf_vote = None
        cmf = chaikin_money_flow(candles)
        if cmf is not None:
            cmf_vote = "long" if cmf > 0 else "short"

        # On-Balance Volume Signal
        obv_vote = None
        obv = on_balance_volume(candles)
        if obv is not None:
            obv_vote = "long" if obv > 0 else "short"

        # Combine Votes
        votes = [arima_vote, ma_vote, macd_vote, bb_vote, cmf_vote, obv_vote]
        long_votes = votes.count("long")
        short_votes = votes.count("short")
        decision = "long" if long_votes > short_votes else "short"

        logging.info(f"Votes: ARIMA={arima_vote}, MA={ma_vote}, MACD={macd_vote}, BB={bb_vote}, CMF={cmf_vote}, OBV={obv_vote}, Decision={decision}")

        # Execute Trade Based on Majority Vote
        if decision == "long" and position != "long":
            close_all_positions()
            logging.info("Opening LONG position...")
            response = place_order("buy", "long", ORDER_SIZE)
            position = "long"
            if response.get("code") == "0":  # If order succeeds
                successful_trades += 1
                total_profit += (current_price - short_ma) * ORDER_SIZE * LEVERAGE
            else:
                failed_trades += 1
        elif decision == "short" and position != "short":
            close_all_positions()
            logging.info("Opening SHORT position...")
            response = place_order("sell", "short", ORDER_SIZE)
            position = "short"
            if response.get("code") == "0":  # If order succeeds
                successful_trades += 1
                total_profit += (short_ma - current_price) * ORDER_SIZE * LEVERAGE
            else:
                failed_trades += 1

        # Log Performance Metrics
        win_rate = successful_trades / (successful_trades + failed_trades) if (successful_trades + failed_trades) > 0 else 0
        logging.info(f"Total Profit: {total_profit}, Win Rate: {win_rate:.2f}, Successful Trades: {successful_trades}, Failed Trades: {failed_trades}"),

        time.sleep(15)  # Wait 15 seconds before next iteration

# Run the strategy
if __name__ == "__main__":
    logging.info("Starting ARIMA + MA + MACD + Bollinger Bands + CMF + OBV Voting System trading bot...")
    combined_strategy()
