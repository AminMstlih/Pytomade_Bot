import time
import hmac
import base64
import requests
import pandas as pd
import numpy as np
import json

# OKX API credentials (replace with your simulated mode keys)
API_KEY = "api_key"
SECRET_KEY = "secret_key"
PASSPHRASE = "passphrase"
BASE_URL = "https://www.okx.com"

# Trading parameters
INSTRUMENT = "DOGE-USD-SWAP"
LEVERAGE = 15
MARGIN_USD = 5
POSITION_SIZE_USD = MARGIN_USD * LEVERAGE  # $75
TP_PNL = 0.07  # +7% PNL
SL_PNL = -0.05  # -5% PNL
CONTRACT_SIZE = 10  # 10 DOGE per contract
POLLING_INTERVAL = 10  # seconds

# Global state
current_position = None  # {"side": "long"/"short", "size": int, "entry_price": float, "algo_ids": list}
last_signal = None

# Fetch OKX server time
def get_server_time():
    """
    Mendapatkan waktu server OKX untuk digunakan dalam tanda tangan API.
    """
    endpoint = "/api/v5/public/time"
    response = requests.get(BASE_URL + endpoint)
    response.raise_for_status()
    return str(float(response.json()["data"][0]["ts"]) / 1000.0)

# Authentication headers
def generate_headers(method, request_path, body=''):
    timestamp = str(get_server_time())
    message = timestamp + method + request_path + body
    signature = base64.b64encode(hmac.new(SECRET_KEY.encode(), message.encode(), digestmod='sha256').digest()).decode()
    return {
        'OK-ACCESS-KEY': API_KEY,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE,
        'Content-Type': 'application/json',
        "x-simulated-trading": "1"
    }

# Private API request
def private_request(method, request_path, body=''):
    url = BASE_URL + request_path
    headers = generate_headers(method, request_path, body)
    response = requests.get(url, headers=headers) if method == 'GET' else requests.post(url, headers=headers, data=body)
    return response.json()

# Public API request
def public_request(request_path):
    url = BASE_URL + request_path
    response = requests.get(url)
    return response.json()

# Set leverage
def set_leverage():
    payload = {'instId': INSTRUMENT, 'lever': str(LEVERAGE), 'mgnMode': 'cross'}
    response = private_request('POST', '/api/v5/account/set-leverage', json.dumps(payload))
    if response['code'] != '0':
        print(f"Leverage set error: {response['msg']} (code: {response['code']})")
    return response['code'] == '0'

# Fetch candle data
def get_historical_data():
    request_path = f"/api/v5/market/candles?instId={INSTRUMENT}&bar=1m&limit=100"
    response = public_request(request_path)
    if response['code'] == '0':
        candles = response['data']
        candles.reverse()  # Oldest first
        df = pd.DataFrame({
            'high': [float(c[2]) for c in candles],
            'low': [float(c[3]) for c in candles],
            'close': [float(c[4]) for c in candles]
        })
        return df
    return None

# Manual indicator calculations
def calculate_indicators(df):
    # SMA: Mean of closing prices over window
    def sma(series, period):
        return series.rolling(window=period, min_periods=period).mean()
    
    # Stochastic Oscillator (5,5,3,3)
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
    return df

# Generate signal
def get_signal(df):
    if len(df) < 21 or np.isnan(df["ma13"].iloc[-1]) or np.isnan(df["ma21"].iloc[-1]) or np.isnan(df["stoch_k"].iloc[-1]):
        return None
    latest, prev = df.iloc[-1], df.iloc[-2]
    ma_long = (prev["ma13"] <= prev["ma21"]) and (latest["ma13"] > latest["ma21"])
    ma_short = (prev["ma13"] >= prev["ma21"]) and (latest["ma13"] < latest["ma21"])
    stoch_long = latest["stoch_k"] > latest["stoch_d"] and latest["stoch_k"] < 80
    stoch_short = latest["stoch_k"] < latest["stoch_d"] and latest["stoch_k"] > 20
    return "long" if ma_long and stoch_long else "short" if ma_short and stoch_short else None

# Calculate contracts
def calculate_contracts(price):
    contracts = int(POSITION_SIZE_USD / (price * CONTRACT_SIZE))
    return max(contracts, 1)

# Place order with TP/SL
def place_order(side, contracts, entry_price):
    tp_price = entry_price * (1 + TP_PNL) if side == "long" else entry_price * (1 - TP_PNL)
    sl_price = entry_price * (1 + SL_PNL) if side == "long" else entry_price * (1 - SL_PNL)
    payload = {
        "instId": INSTRUMENT,
        "tdMode": "cross",
        "side": "buy" if side == "long" else "sell",
        "ordType": "market",
        "sz": str(contracts),
        "tpTriggerPx": str(round(tp_price, 4)),
        "tpOrdPx": "-1",
        "slTriggerPx": str(round(sl_price, 4)),
        "slOrdPx": "-1",
        "tpTriggerPxType": "last",
        "slTriggerPxType": "last"
    }
    response = private_request("POST", "/api/v5/trade/order", json.dumps(payload))
    if response["code"] == "0":
        ord_id = response["data"][0]["ordId"]
        algo_response = private_request("GET", f"/api/v5/trade/orders-algo-pending?instId={INSTRUMENT}")
        if algo_response["code"] == "0":
            algo_ids = [algo["algoId"] for algo in algo_response["data"] if algo.get("ordId") == ord_id]
            return {"ordId": ord_id, "size": contracts, "entry_price": entry_price, "algo_ids": algo_ids}
    print(f"Order placement failed: {response['msg']} (code: {response['code']})")
    return None

# Close position
def close_position(pos):
    if pos and "algo_ids" in pos:
        for algo_id in pos["algo_ids"]:
            private_request("POST", "/api/v5/trade/cancel-algo-order", json.dumps([{"algoId": algo_id}]))
    if pos:
        side = "sell" if pos["side"] == "long" else "buy"
        payload = {"instId": INSTRUMENT, "tdMode": "cross", "side": side, "ordType": "market", "sz": str(pos["size"])}
        response = private_request("POST", "/api/v5/trade/order", json.dumps(payload))
        if response["code"] != "0":
            print(f"Close position failed: {response['msg']} (code: {response['code']})")

# Check position
def check_position():
    response = private_request("GET", f"/api/v5/account/positions?instId={INSTRUMENT}")
    if response["code"] == "0" and response["data"]:
        pos = response["data"][0]
        return {"side": "long" if pos["posSide"] == "long" else "short", "size": int(pos["pos"]), "entry_price": float(pos["avgPx"])}
    return None

# Main loop
def main():
    global current_position, last_signal
    if not set_leverage():
        return
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
                    print(f"Opened {signal} position: {contracts} contracts at ${price}")
            elif (current_position["side"] == "long" and signal == "short") or \
                 (current_position["side"] == "short" and signal == "long"):
                close_position(current_position)
                price = df["close"].iloc[-1]
                contracts = calculate_contracts(price)
                order = place_order(signal, contracts, price)
                if order:
                    current_position = {**order, "side": signal}
                    print(f"Reversed to {signal} position: {contracts} contracts at ${price}")
            time.sleep(POLLING_INTERVAL)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        current_position = check_position()
        if current_position:
            close_position(current_position)
            print("Bot stopped, position closed")
        else:
            print("Bot stopped, no open position")
