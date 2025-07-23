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
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# OKX API credentials (demo, as provided)
API_KEY = "API_KEY"
SECRET_KEY = "SECRET_KEY"
PASSPHRASE = "PASSPHRASE"
BASE_URL = "https://www.okx.com"

# Trading parameters
INSTRUMENT = "DOGE-USDT-SWAP"
LEVERAGE = 15
MARGIN_COST_USD = 5  # Fixed margin cost in USDT before leverage
CONTRACT_SIZE = 1000  # 1 contract = 10 DOGE
MAX_CONTRACTS = 30  # Hard cap to prevent oversized trades
TP_PNL = 0.07  # +7% PNL
SL_PNL = -0.05  # -5% PNL
USE_PNL_BASED = True  # Use % PNL for TP/SL
POLLING_INTERVAL = 10  # seconds
SIGNAL_COOLDOWN = 30  # seconds
RESET_SIGNAL_AFTER = 30  # seconds to reset last_signal if no position
MIN_MA_DIFF = 0.001  # 0.1% minimum MA difference
STOCH_THRESHOLD = 0.05  # 0.5% threshold for Stoch_K vs Stoch_D

# Global state
current_position = None
last_signal = None
last_trade_time = 0
last_position_close_time = 0

# Fetch OKX server time
def get_server_time():
    endpoint = "/api/v5/public/time"
    try:
        response = requests.get(BASE_URL + endpoint)
        response.raise_for_status()
        server_time = str(float(response.json()["data"][0]["ts"]) / 1000.0)
        logger.debug(f"Fetched server time: {server_time}")
        return server_time
    except Exception as e:
        logger.error(f"Error fetching server time: {e}")
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
        "x-simulated-trading": "0"  # 0 real / 1 Demo trading
    }
    logger.debug(f"Generated headers for {method} {request_path}")
    return headers

# Private API request
def private_request(method, request_path, body=''):
    url = BASE_URL + request_path
    headers = generate_headers(method, request_path, body)
    for attempt in range(3):
        try:
            response = requests.request(method, url, headers=headers, data=body)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Private request {method} {request_path} response: code={data['code']}, data={data.get('data', [])}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Private request {method} {request_path} failed (attempt {attempt+1}/3): {e}")
            if hasattr(e.response, 'json'):
                try:
                    error_data = e.response.json()
                    logger.error(f"OKX error details: code={error_data.get('code', 'N/A')}, msg={error_data.get('msg', 'N/A')}")
                except ValueError:
                    logger.error("Failed to parse OKX error response")
            if attempt < 2:
                time.sleep(10)  # Increased delay for rate limits
            else:
                return {"code": "1", "msg": f"Network error: {str(e)}"}
    sys.stdout.flush()

# Public API request
def public_request(request_path):
    url = BASE_URL + request_path
    for attempt in range(3):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Public request {request_path} response: {data}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Public request {request_path} failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(10)
            else:
                return {"code": "1", "msg": f"Network error: {str(e)}"}
    sys.stdout.flush()

# Check account balance
def check_balance():
    response = private_request("GET", "/api/v5/account/balance?ccy=USDT")
    if response["code"] == "0" and response["data"]:
        balance = float(response["data"][0]["details"][0]["cashBal"]) if response["data"][0]["details"] else 0
        logger.info(f"USDT balance: ${balance:.2f}")
        return balance >= MARGIN_COST_USD
    logger.error(f"Failed to check balance: {response['msg']} (code: {response['code']})")
    return False

# Check Unified Account mode
def check_unified_account():
    response = private_request("GET", "/api/v5/account/config")
    if response["code"] == "0" and response["data"]:
        acct_mode = response["data"][0].get("acctLv", "0")
        is_unified = acct_mode in ["2", "3", "4"]
        logger.info(f"Account mode: {'Unified' if is_unified else 'Non-Unified'} (acctLv: {acct_mode})")
        return is_unified
    logger.error(f"Failed to check account mode: {response['msg']} (code: {response['code']})")
    return False

# Set leverage
def set_leverage():
    payload = {'instId': INSTRUMENT, 'lever': str(LEVERAGE), 'mgnMode': 'cross'}
    response = private_request('POST', '/api/v5/account/set-leverage', json.dumps(payload))
    if response['code'] == '0':
        logger.info(f"Leverage set to {LEVERAGE}x for {INSTRUMENT}")
        return True
    logger.error(f"Failed to set leverage: {response['msg']} (code: {response['code']})")
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
        logger.info(f"Fetched {len(candles)} candles. Latest close: ${df['close'].iloc[-1]:.4f}")
        return df
    logger.error(f"Failed to fetch candles: {response['msg']} (code: {response['code']})")
    return None

# Manual indicator calculations
def calculate_indicators(df):
    def sma(series, period):
        return series.rolling(window=period, min_periods=period).mean()
    
    def stochastic(high, low, close, k_period=5, d_period=5, smooth_k=3):
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
                    f"Stoch_K={df['stoch_k'].iloc[-1]:.2f}, Stoch_D={df['stoch_d'].iloc[-1]:.2f}")
        ma_diff = abs(df['ma13'].iloc[-1] - df['ma21'].iloc[-1]) / df['ma21'].iloc[-1]
        logger.info(f"MA13-MA21 difference: {ma_diff*100:.2f}%")
        if ma_diff < MIN_MA_DIFF:
            logger.warning(f"MA13 and MA21 too close ({ma_diff*100:.2f}% < {MIN_MA_DIFF*100}%); may reduce signal reliability")
        stoch_diff = abs(df['stoch_k'].iloc[-1] - df['stoch_d'].iloc[-1]) / max(df['stoch_d'].iloc[-1], 0.01)
        logger.info(f"Stoch_K-Stoch_D difference: {stoch_diff*100:.2f}%")
        if stoch_diff < STOCH_THRESHOLD:
            logger.warning(f"Stoch_K and Stoch_D too close ({stoch_diff*100:.2f}% < {STOCH_THRESHOLD*100}%); may prevent signal")
    return df

# Generate signal
def get_signal(df):
    if len(df) < 21:
        logger.info(f"Insufficient data: {len(df)} candles (need 21)")
        return None
    if np.isnan(df["ma13"].iloc[-1]) or np.isnan(df["ma21"].iloc[-1]) or np.isnan(df["stoch_k"].iloc[-1]):
        logger.info("Indicators contain NaN values")
        return None
    latest = df.iloc[-1]
    ma_long = latest["ma13"] > latest["ma21"]
    ma_short = latest["ma13"] < latest["ma21"]
    stoch_long = latest["stoch_k"] > latest["stoch_d"] and latest["stoch_k"] < 98
    stoch_short = latest["stoch_k"] < latest["stoch_d"] and latest["stoch_k"] > 2
    signal = "long" if ma_long and stoch_long else "short" if ma_short and stoch_short else None
    logger.info(f"Signal check: MA_Long={ma_long}, MA_Short={ma_short}, Stoch_Long={stoch_long}, "
                f"Stoch_Short={stoch_short}, Signal={signal}")
    return signal

# Calculate contracts based on USDT margin cost
def calculate_contracts(price):
    notional_value = MARGIN_COST_USD * LEVERAGE
    contracts = int(notional_value / (price * CONTRACT_SIZE))
    contracts = min(max(contracts, 0.1), MAX_CONTRACTS)
    actual_notional = contracts * price * CONTRACT_SIZE
    margin_used = actual_notional / LEVERAGE
    logger.info(f"Calculated contracts: {contracts} ({contracts * CONTRACT_SIZE} DOGE), "
                f"price ${price:.4f}, notional ${actual_notional:.2f}, margin used ${margin_used:.2f}")
    return contracts

# Check for TP/SL closure
def check_tp_sl_closure():
    response = private_request("GET", f"/api/v5/trade/orders-algo-pending?instId={INSTRUMENT}&ordType=oco")
    if response["code"] == "0":
        if not response["data"]:
            logger.info("No pending OCO orders, likely TP/SL closed")
            return True
        logger.debug(f"Pending OCO orders exist: {[order['algoId'] for order in response['data']]}")
        return False
    logger.error(f"Failed to check algo orders: {response['msg']} (code: {response['code']})")
    return False

# Place order with TP/SL
def place_order(side, contracts, entry_price):
    if USE_PNL_BASED:
        tp_factor = TP_PNL / LEVERAGE
        sl_factor = abs(SL_PNL) / LEVERAGE
    else:
        tp_factor = TP_PNL
        sl_factor = abs(SL_PNL)
    if side == "long":
        tp_price = entry_price * (1 + tp_factor)
        sl_price = entry_price * (1 - sl_factor)
    else:  # short
        tp_price = entry_price * (1 - tp_factor)
        sl_price = entry_price * (1 + sl_factor)
    payload = {
        "instId": INSTRUMENT,
        "tdMode": "cross",
        "side": "buy" if side == "long" else "sell",
        "posSide": side,
        "ordType": "market",
        "sz": str(contracts),
        "attachAlgoOrds": [
            {
                "algoOrdType": "oco",
                "tpTriggerPx": str(round(tp_price, 4)),
                "tpOrdPx": "-1",
                "slTriggerPx": str(round(sl_price, 4)),
                "slOrdPx": "-1",
                "tpTriggerPxType": "last",
                "slTriggerPxType": "last"
            }
        ]
    }
    logger.info(f"Placing {side} order: {contracts} contracts ({contracts * CONTRACT_SIZE} DOGE), "
                f"Entry=${entry_price:.4f}, TP=${tp_price:.4f}, SL=${sl_price:.4f}")
    response = private_request("POST", "/api/v5/trade/order", json.dumps(payload))
    if response["code"] == "0" and response["data"]:
        ord_id = response["data"][0]["ordId"]
        logger.info(f"Order placed successfully: ordId={ord_id}, size={contracts} contracts")
        return {"ordId": ord_id, "size": contracts, "entry_price": entry_price}
    logger.error(f"Order placement failed: {response['msg']} (code: {response['code']})")
    return None

# Close all positions
def close_all_positions():
    # Cancel all algo orders
    response = private_request("GET", f"/api/v5/trade/orders-algo-pending?instId={INSTRUMENT}&ordType=oco")
    if response["code"] == "0" and response["data"]:
        algo_ids = [algo["algoId"] for algo in response["data"]]
        if algo_ids:
            cancel_payload = [{"instId": INSTRUMENT, "algoId": algo_id} for algo_id in algo_ids]
            cancel_response = private_request("POST", "/api/v5/trade/cancel-algo-order", json.dumps(cancel_payload))
            if cancel_response["code"] == "0":
                logger.info(f"Cancelled algo orders: {algo_ids}")
            else:
                logger.error(f"Failed to cancel algo orders: {cancel_response['msg']} (code: {cancel_response['code']})")
    
    # Close all positions
    payload = {"instId": INSTRUMENT, "mgnMode": "cross"}
    logger.info(f"Closing all positions for {INSTRUMENT}")
    for attempt in range(3):
        response = private_request("POST", "/api/v5/trade/close-position", json.dumps(payload))
        if response["code"] == "0":
            logger.info(f"All positions closed successfully for {INSTRUMENT}")
            pos_check = check_position()
            if not pos_check:
                return True
            logger.error(f"Position still open after close attempt {attempt+1}/3: {pos_check}")
        else:
            logger.error(f"Close all positions failed (attempt {attempt+1}/3): {response['msg']} (code: {response['code']})")
        time.sleep(10)
    return False

# Check position
def check_position():
    response = private_request("GET", f"/api/v5/account/positions?instId={INSTRUMENT}")
    if response["code"] == "0":
        for pos in response["data"]:
            if pos["pos"] != "0" and pos["avgPx"]:
                try:
                    position = {
                        "side": pos["posSide"],
                        "size": int(pos["pos"]),
                        "entry_price": float(pos["avgPx"])
                    }
                    logger.info(f"Current position: {position['side']}, {position['size']} contracts "
                                f"({position['size'] * CONTRACT_SIZE} DOGE), entry ${position['entry_price']:.4f}")
                    return position
                except ValueError as e:
                    logger.warning(f"Skipping position with invalid data: {pos} (error: {e})")
        logger.info("No open position")
        return None
    logger.error(f"Failed to check position: {response['msg']} (code: {response['code']})")
    return None

# Main loop
def main():
    global current_position, last_signal, last_trade_time, last_position_close_time
    logger.info("Starting trading bot for DOGE-USDT-SWAP")
    # Check Unified Account mode
    if not check_unified_account():
        logger.error("Account is not in Unified Account mode. Please enable it in OKX settings.")
        return
    # Check balance
    if not check_balance():
        logger.error("Insufficient USDT balance. Please transfer at least $5 to Trading Account.")
        return
    # Set leverage
    if not set_leverage():
        logger.warning("Proceeding without auto-setting leverage. Ensure leverage is set to 15x via OKX web interface.")
    while True:
        try:
            # Always check position
            current_position = check_position()
            # Check for TP/SL closure or no position
            current_time = time.time()
            if current_position is None:
                if check_tp_sl_closure() or (last_position_close_time > 0 and current_time - last_position_close_time > RESET_SIGNAL_AFTER):
                    logger.info("No position and TP/SL closed or timeout; resetting last_signal")
                    last_signal = None
                    last_position_close_time = current_time
            # Check signal cooldown
            if current_time - last_trade_time < SIGNAL_COOLDOWN:
                logger.info(f"Waiting for {SIGNAL_COOLDOWN - (current_time - last_trade_time):.1f}s cooldown")
                time.sleep(POLLING_INTERVAL)
                continue
            df = get_historical_data()
            if df is None:
                time.sleep(POLLING_INTERVAL)
                continue
            df = calculate_indicators(df)
            signal = get_signal(df)
            if not signal:
                time.sleep(POLLING_INTERVAL)
                continue
            if current_position is None:
                if signal != last_signal or last_signal is None:
                    price = df["close"].iloc[-1]
                    contracts = calculate_contracts(price)
                    order = place_order(signal, contracts, price)
                    if order:
                        current_position = {**order, "side": signal}
                        last_signal = signal
                        last_trade_time = current_time
                        last_position_close_time = 0
            elif (current_position["side"] == "long" and signal == "short") or \
                 (current_position["side"] == "short" and signal == "long"):
                if close_all_positions():
                    current_position = check_position()
                    if current_position is None:
                        price = df["close"].iloc[-1]
                        contracts = calculate_contracts(price)
                        order = place_order(signal, contracts, price)
                        if order:
                            current_position = {**order, "side": signal}
                            last_signal = signal
                            last_trade_time = current_time
                            last_position_close_time = 0
                    else:
                        logger.error(f"Position still open after closure: {current_position}")
                else:
                    logger.error("Failed to close all positions; skipping new order")
            time.sleep(POLLING_INTERVAL)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if check_position():
            close_all_positions()
        logger.info("Bot stopped, all positions closed")
    sys.stdout.flush()
