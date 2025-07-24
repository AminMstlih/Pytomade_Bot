import time
import hmac
import base64
import requests
import pandas as pd
import numpy as np
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# OKX API credentials
API_KEY = "api_key"
SECRET_KEY = "secret_key"
PASSPHRASE = "passphrase"
BASE_URL = "https://www.okx.com"

# Trading parameters
INSTRUMENT = "DOGE-USDT-SWAP"
LEVERAGE = 15
MARGIN_USD = 5
POSITION_SIZE_USD = MARGIN_USD * LEVERAGE  # $75
TP_PNL = 0.07  # +7% PNL
SL_PNL = -0.05  # -5% PNL
CONTRACT_SIZE = 10  # 10 DOGE per contract
POLLING_INTERVAL = 10  # seconds

# Global state
current_position = None
last_signal = None

# Fetch OKX server time
def get_server_time():
    endpoint = "/api/v5/public/time"
    try:
        response = requests.get(BASE_URL + endpoint)
        response.raise_for_status()
        server_time = str(float(response.json()["data"][0]["ts"]) / 1000.0)
        logger.debug(f"Fetched server time: {server_time}", extra={'force': True})
        return server_time
    except Exception as e:
        logger.error(f"Error fetching server time: {e}", extra={'force': True})
        return str(int(time.time()))

# Authentication headers
def generate_headers(method, request_path, body=''):
    timestamp = get_server_time()
    message = timestamp + method + request_path + body
    signature = base64.b64encode(hmac.new(SECRET_KEY.encode(), message.encode(), digestmod='sha256').digest()).decode()
    headers = {
        'OK-ACCESS-KEY': API_KEY,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE,
        'Content-Type': 'application/json',
        "x-simulated-trading": "1"
    }
    logger.debug(f"Generated headers for {method} {request_path}", extra={'force': True})
    return headers

# Private API request
def private_request(method, request_path, body=''):
    url = BASE_URL + request_path
    headers = generate_headers(method, request_path, body)
    try:
        response = requests.get(url, headers=headers) if method == 'GET' else requests.post(url, headers=headers, data=body)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Private request {method} {request_path} response: {data}", extra={'force': True})
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Private request {method} {request_path} failed: {e}", extra={'force': True})
        return {"code": "1", "msg": f"Network error: {str(e)}"}

# Public API request
def public_request(request_path):
    url = BASE_URL + request_path
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Public request {request_path} response: {data}", extra={'force': True})
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Public request {request_path} failed: {e}", extra={'force': True})
        return {"code": "1", "msg": f"Network error: {str(e)}"}

# Check account balance
def check_balance():
    response = private_request("GET", "/api/v5/account/balance?ccy=USDT")
    if response["code"] == "0":
        balance = float(response["data"][0]["details"][0]["cashBal"]) if response["data"][0]["details"] else 0
        logger.info(f"USDT balance: ${balance:.2f}", extra={'force': True})
        return balance >= MARGIN_USD
    logger.error(f"Failed to check balance: {response['msg']} (code: {response['code']})", extra={'force': True})
    return False

# Check Unified Account mode
def check_unified_account():
    response = private_request("GET", "/api/v5/account/config")
    if response["code"] == "0":
        acct_mode = response["data"][0].get("acctLv", "0")
        is_unified = acct_mode in ["2", "3", "4"]
        logger.info(f"Account mode: {'Unified' if is_unified else 'Non-Unified'} (acctLv: {acct_mode})", extra={'force': True})
        return is_unified
    logger.error(f"Failed to check account mode: {response['msg']} (code: {response['code']})", extra={'force': True})
    return False

# Set leverage with retries
def set_leverage():
    payload = {'instId': INSTRUMENT, 'lever': str(LEVERAGE), 'mgnMode': 'cross'}
    for attempt in range(3):
        response = private_request('POST', '/api/v5/account/set-leverage', json.dumps(payload))
        if response['code'] == '0':
            logger.info(f"Leverage set to {LEVERAGE}x for {INSTRUMENT}", extra={'force': True})
            return True
        logger.error(f"Leverage set error (attempt {attempt+1}/3): {response['msg']} (code: {response['code']})", extra={'force': True})
        time.sleep(5)
    logger.warning("Failed to set leverage after retries. Please set leverage manually to 15x via OKX web interface.", extra={'force': True})
    return False

# Fetch candle data
def get_historical_data():
    request_path = f"/api/v5/market/candles?instId={INSTRUMENT}&bar=1m&limit=100"
    response = public_request(request_path)
    if response['code'] == '0':
        candles = response['data']
        candles.reverse()
        df = pd.DataFrame({
            'high': [float(c[2]) for c in candles],
            'low': [float(c[3]) for c in candles],
            'close': [float(c[4]) for c in candles]
        })
        logger.info(f"Fetched {len(candles)} candles. Latest 3 closes: {df['close'].tail(3).to_list()}", extra={'force': True})
        sys.stdout.flush()
        return df
    logger.error(f"Failed to fetch candles: {response['msg']} (code: {response['code']})", extra={'force': True})
    sys.stdout.flush()
    return None

# Manual indicator calculations
def calculate_indicators(df):
    def sma(series, period):
        return series.rolling(window=period, min_periods=period).mean()
    
    def stochastic(high, low, close, k_period=5, d_period=3, smooth_k=3):
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k_smooth = k.rolling(window=smooth_k, min_periods=smooth_k).mean()
        d = k_smooth.rolling(window=d_period, min_periods=d_period).mean()
        return k_smooth, d

    df["ma13"] = sma(df["close"], 13)
    df["ma21"] = sma(df["close"], 21)
    df["stoch_k"], df["stoch_d"] = stochastic(df["high"], df["low"], df["close"])
    if len(df) >= 2:
        logger.info(f"Indicators: MA13={df['ma13'].iloc[-1]:.4f}, MA21={df['ma21'].iloc[-1]:.4f}, "
                    f"Stoch_K={df['stoch_k'].iloc[-1]:.2f}, Stoch_D={df['stoch_d'].iloc[-1]:.2f}", extra={'force': True})
        ma_diff = abs(df['ma13'].iloc[-1] - df['ma21'].iloc[-1]) / df['ma21'].iloc[-1] * 100
        logger.info(f"MA13-MA21 difference: {ma_diff:.2f}%", extra={'force': True})
    sys.stdout.flush()
    return df

# Generate signal
def get_signal(df):
    if len(df) < 21:
        logger.info(f"Insufficient data: {len(df)} candles (need 21)", extra={'force': True})
        return None
    if np.isnan(df["ma13"].iloc[-1]) or np.isnan(df["ma21"].iloc[-1]) or np.isnan(df["stoch_k"].iloc[-1]):
        logger.info("Indicators contain NaN values", extra={'force': True})
        return None
    latest = df.iloc[-1]
    ma_long = latest["ma13"] > latest["ma21"]
    ma_short = latest["ma13"] < latest["ma21"]
    stoch_long = latest["stoch_k"] > latest["stoch_d"] and latest["stoch_k"] < 95
    stoch_short = latest["stoch_k"] < latest["stoch_d"] and latest["stoch_k"] > 5
    signal = "long" if ma_long and stoch_long else "short" if ma_short and stoch_short else None
    logger.info(f"Signal check: MA_Long={ma_long}, MA_Short={ma_short}, Stoch_Long={stoch_long}, "
                f"Stoch_Short={stoch_short}, Signal={signal}", extra={'force': True})
    sys.stdout.flush()
    return signal

# Calculate contracts
def calculate_contracts(price):
    contracts = int(POSITION_SIZE_USD / (price * CONTRACT_SIZE))
    contracts = max(contracts, 1)
    logger.info(f"Calculated contracts: {contracts} for price ${price:.4f}", extra={'force': True})
    return contracts

# Place order with TP/SL
def place_order(side, contracts, entry_price):
    tp_price = entry_price * (1 + TP_PNL) if side == "long" else entry_price * (1 - TP_PNL)
    sl_price = entry_price * (1 + SL_PNL) if side == "long" else entry_price * (1 - TP_PNL)
    payload = {
        "instId": INSTRUMENT,
        "tdMode": "cross",
        "side": "buy" if side == "long" else "sell",
        "posSide": side,  # Explicitly set posSide
        "ordType": "market",
        "sz": str(contracts),
        "tpTriggerPx": str(round(tp_price, 4)),
        "tpOrdPx": "-1",
        "slTriggerPx": str(round(sl_price, 4)),
        "slOrdPx": "-1",
        "tpTriggerPxType": "last",
        "slTriggerPxType": "last"
    }
    logger.info(f"Placing {side} order: {contracts} contracts, Entry=${entry_price:.4f}, TP=${tp_price:.4f}, SL=${sl_price:.4f}", extra={'force': True})
    for attempt in range(3):
        response = private_request("POST", "/api/v5/trade/order", json.dumps(payload))
        if response["code"] == "0":
            ord_id = response["data"][0]["ordId"]
            algo_response = private_request("GET", f"/api/v5/trade/orders-algo-pending?instId={INSTRUMENT}")
            if algo_response["code"] == "0":
                algo_ids = [algo["algoId"] for algo in algo_response["data"] if algo.get("ordId") == ord_id]
                logger.info(f"Order placed successfully: ordId={ord_id}, algo_ids={algo_ids}", extra={'force': True})
                sys.stdout.flush()
                return {"ordId": ord_id, "size": contracts, "entry_price": entry_price, "algo_ids": algo_ids}
            logger.error(f"Failed to fetch algo orders: {algo_response['msg']} (code: {algo_response['code']})", extra={'force': True})
        else:
            logger.error(f"Order placement failed (attempt {attempt+1}/3): {response['msg']} (code: {response['code']})", extra={'force': True})
        time.sleep(5)
    sys.stdout.flush()
    return None

# Close position
def close_position(pos):
    if pos and "algo_ids" in pos:
        for algo_id in pos["algo_ids"]:
            response = private_request("POST", "/api/v5/trade/cancel-algo-order", json.dumps([{"algoId": algo_id}]))
            if response["code"] != "0":
                logger.error(f"Failed to cancel algo order {algo_id}: {response['msg']} (code: {response['code']})", extra={'force': True})
            else:
                logger.info(f"Cancelled algo order {algo_id}", extra={'force': True})
    if pos:
        side = "sell" if pos["side"] == "long" else "buy"
        pos_side = "short" if pos["side"] == "long" else "long"
        payload = {"instId": INSTRUMENT, "tdMode": "cross", "side": side, "posSide": pos_side, "ordType": "market", "sz": str(pos["size"])}
        logger.info(f"Closing {pos['side']} position: {pos['size']} contracts at entry ${pos['entry_price']:.4f}", extra={'force': True})
        response = private_request("POST", "/api/v5/trade/order", json.dumps(payload))
        if response["code"] == "0":
            logger.info(f"Closed {pos['side']} position successfully", extra={'force': True})
        else:
            logger.error(f"Close position failed: {response['msg']} (code: {response['code']})", extra={'force': True})
        sys.stdout.flush()

# Check position
def check_position():
    response = private_request("GET", f"/api/v5/account/positions?instId={INSTRUMENT}")
    if response["code"] == "0":
        if response["data"]:
            pos = response["data"][0]
            position = {"side": "long" if pos["posSide"] == "long" else "short", 
                       "size": int(pos["pos"]), 
                       "entry_price": float(pos["avgPx"])}
            logger.info(f"Current position: {position}", extra={'force': True})
            return position
        logger.info("No open position", extra={'force': True})
        return None
    logger.error(f"Failed to check position: {response['msg']} (code: {response['code']})", extra={'force': True})
    sys.stdout.flush()
    return None

# Main loop
def main():
    global current_position, last_signal
    logger.info("Starting trading bot for DOGE-USDT-SWAP", extra={'force': True})
    sys.stdout.flush()
    # Check Unified Account mode
    if not check_unified_account():
        logger.error("Account is not in Unified Account mode. Please enable it in OKX settings.", extra={'force': True})
        sys.stdout.flush()
        return
    # Check balance
    if not check_balance():
        logger.error("Insufficient USDT balance. Please transfer at least $5 to Trading Account.", extra={'force': True})
        sys.stdout.flush()
        return
    # Try setting leverage
    if not set_leverage():
        logger.warning("Proceeding without auto-setting leverage. Ensure leverage is set to 15x via OKX web interface.", extra={'force': True})
        sys.stdout.flush()
    while True:
        try:
            df = get_historical_data()
            if df is None:
                time.sleep(POLLING_INTERVAL)
                continue
            df = calculate_indicators(df)
            signal = get_signal(df)
            if not signal or signal == last_signal:
                time.sleep(POLLING_INTERVAL)
                continue
            last_signal = signal
            current_position = check_position()
            if not current_position:
                price = df["close"].iloc[-1]
                contracts = calculate_contracts(price)
                order = place_order(signal, contracts, price)
                if order:
                    current_position = {**order, "side": signal}
            elif (current_position["side"] == "long" and signal == "short") or \
                 (current_position["side"] == "short" and signal == "long"):
                close_position(current_position)
                price = df["close"].iloc[-1]
                contracts = calculate_contracts(price)
                order = place_order(signal, contracts, price)
                if order:
                    current_position = {**order, "side": signal}
            time.sleep(POLLING_INTERVAL)
        except Exception as e:
            logger.error(f"Error in main loop: {e}", extra={'force': True})
            sys.stdout.flush()
            time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        current_position = check_position()
        if current_position:
            close_position(current_position)
        logger.info("Bot stopped, no open position" if not current_position else "Bot stopped, position closed", extra={'force': True})
        sys.stdout.flush()
