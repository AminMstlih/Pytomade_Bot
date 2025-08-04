import os
import time
import hmac
import base64
import requests
import pandas as pd
import numpy as np
import json
import logging
import sys
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('trading_bot.log')
                    ])
logger = logging.getLogger(__name__)

# OKX API credentials
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")
BASE_URL = "https://www.okx.com"

# Trading parameters
INSTRUMENT = "DOGE-USDT-SWAP"
LEVERAGE = 15
MARGIN_COST_USD = 2  # Fixed margin cost in USDT per position for hedging
CONTRACT_SIZE = 1000  # 1 contract = 10 DOGE for DOGE-USDT-SWAP
MAX_CONTRACTS = 10  # Hard cap on contracts per position
MAX_POSITION_SIZE = 10000  # Max position size in USDT

# Risk Management Parameters
TP_PNL = 0.07  # +7% PNL
SL_PNL = -0.05  # -5% PNL
USE_PNL_BASED = True
MAX_DAILY_LOSS = -1  # Max daily loss in USD
USE_TRAILING_STOP = True
TRAILING_STOP_ACTIVATION = 0.02  # Activate at 5% profit
TRAILING_STOP_DISTANCE = 0.05  # 5% trailing distance

# Volume & Volatility Filters
MIN_24H_VOLUME = 1000000  # Min 24h volume in USDT
MAX_SPREAD = 0.001  # Max spread (0.1%)
VOLATILITY_THRESHOLD = 0.002  # Min 24h price range
POLLING_INTERVAL = 10  # seconds
SIGNAL_COOLDOWN = 60  # seconds
RESET_SIGNAL_AFTER = 300  # Reset last_signal
MIN_MA_DIFF = 0.0001  # 0.01% min MA difference
STOCH_THRESHOLD = 0.005  # 0.5% min StochRSI difference
ADX_THRESHOLD = 15  # Min ADX for trend confirmation
VOLUME_THRESHOLD = 0.7  # Minimum volume relative to SMA
API_RATE_LIMIT_DELAY = 0.1  # Delay per API request

# Global state
current_positions = {}  # Dictionary for long/short positions
last_signal = None
last_trade_time = 0
last_position_close_time = 0

# HTTP session for optimized API calls
session = requests.Session()

# Fetch OKX server time
def get_server_time():
    endpoint = "/api/v5/public/time"
    try:
        response = session.get(BASE_URL + endpoint)
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
    signature = base64.b64encode(
        hmac.new(SECRET_KEY.encode(), message.encode(),
                 digestmod='sha256').digest()).decode()
    headers = {
        'OK-ACCESS-KEY': API_KEY,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE,
        'Content-Type': 'application/json',
        'x-simulated-trading': '0'  # 0 for real market
    }
    logger.debug(f"Generated headers for {method} {request_path}")
    return headers

# Private API request
def private_request(method, request_path, body=''):
    url = BASE_URL + request_path
    headers = generate_headers(method, request_path, body)
    for attempt in range(3):
        try:
            response = session.request(method, url, headers=headers, data=body)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Private request {method} {request_path} response: code={data['code']}")
            time.sleep(API_RATE_LIMIT_DELAY)
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Private request {method} {request_path} failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(10)
            else:
                return {"code": "1", "msg": f"Network error: {str(e)}"}
    sys.stdout.flush()

# Public API request
def public_request(request_path):
    url = BASE_URL + request_path
    for attempt in range(3):
        try:
            response = session.get(url)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Public request {request_path} response: {data}")
            time.sleep(API_RATE_LIMIT_DELAY)
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Public request {request_path} failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(10)
            else:
                return {"code": "1", "msg": f"Network error: {str(e)}"}
    sys.stdout.flush()

# Check account balance for hedging (ensure enough for two positions)
def check_balance():
    response = private_request("GET", "/api/v5/account/balance?ccy=USDT")
    if response["code"] == "0" and response["data"]:
        balance = float(response["data"][0]["details"][0]["cashBal"]) if response["data"][0]["details"] else 0
        logger.info(f"USDT balance: ${balance:.2f}")
        # Check if balance is enough for two positions (long + short)
        return balance >= MARGIN_COST_USD * 2
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

# Set leverage for isolated margin mode
def set_leverage():
    payload = {'instId': INSTRUMENT, 'lever': str(LEVERAGE), 'mgnMode': 'cross'}
    response = private_request('POST', '/api/v5/account/set-leverage', json.dumps(payload))
    if response['code'] == '0':
        logger.info(f"Leverage set to {LEVERAGE}x for {INSTRUMENT} with isolated margin")
        return True
    logger.error(f"Failed to set leverage: {response['msg']} (code: {response['code']})")
    return False

# Check market conditions
def check_market_conditions():
    request_path = f"/api/v5/market/ticker?instId={INSTRUMENT}"
    response = public_request(request_path)
    if response['code'] == '0' and response['data']:
        ticker = response['data'][0]
        volume_24h = float(ticker['volCcy24h'])
        last_price = float(ticker['last'])
        best_bid = float(ticker['bidPx'])
        best_ask = float(ticker['askPx'])

        spread = (best_ask - best_bid) / best_bid
        if spread > MAX_SPREAD:
            logger.warning(f"Spread too high: {spread*100:.3f}% > {MAX_SPREAD*100:.3f}%")
            return False

        if volume_24h < MIN_24H_VOLUME:
            logger.warning(f"24h volume too low: ${volume_24h:,.0f} < ${MIN_24H_VOLUME:,.0f}")
            return False

        high_24h = float(ticker['high24h'])
        low_24h = float(ticker['low24h'])
        volatility = (high_24h - low_24h) / low_24h
        if volatility < VOLATILITY_THRESHOLD:
            logger.warning(f"24h volatility too low: {volatility*100:.1f}% < {VOLATILITY_THRESHOLD*100:.1f}%")
            return False

        logger.info(f"Market conditions good: Spread={spread*100:.3f}%, Volume=${volume_24h:,.0f}, Volatility={volatility*100:.1f}%")
        return True
    logger.error(f"Failed to get ticker data: {response.get('msg', 'Unknown error')}")
    return False

# Fetch candle data
def get_historical_data():
    request_path = f"/api/v5/market/candles?instId={INSTRUMENT}&bar=1m&limit=100"
    response = public_request(request_path)
    if response['code'] != '0':
        logger.error(f"Failed to fetch candles: {response['msg']} (code: {response['code']})")
        return None
    candles = response['data']
    candles.reverse()
    if len(candles) < 21:
        logger.warning(f"Insufficient candles: {len(candles)} (need at least 21)")
        return None
    df = pd.DataFrame({
        'timestamp': [float(c[0]) for c in candles],
        'open': [float(c[1]) for c in candles],
        'high': [float(c[2]) for c in candles],
        'low': [float(c[3]) for c in candles],
        'close': [float(c[4]) for c in candles],
        'volume': [float(c[5]) for c in candles]
    })
    if df[['open', 'high', 'low', 'close', 'volume']].isna().any().any() or (df[['open', 'high', 'low', 'close', 'volume']] <= 0).any().any():
        logger.warning("Invalid candle data: contains NaN or non-positive values")
        return None
    logger.info(f"Fetched {len(candles)} candles. Latest close: ${df['close'].iloc[-1]:.4f}")
    return df

# Manual indicator calculations
def calculate_indicators(df):
    def sma(series, period):
        return series.rolling(window=period, min_periods=period).mean()

    def atr(high, low, close, period):
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return tr['tr'].rolling(window=period).mean()

    def rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def adx(high, low, close, period):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = atr(high, low, close, period)
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / tr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1 / period).mean() / tr))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1 / period).mean()
        return adx

    def stochrsi(series, rsi_length=5, stoch_length=5, k=3, d=3):
        rsi_series = rsi(series, rsi_length)
        min_rsi = rsi_series.rolling(window=stoch_length, min_periods=stoch_length).min()
        max_rsi = rsi_series.rolling(window=stoch_length, min_periods=stoch_length).max()
        stochrsi_raw = (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-10)
        k_values = stochrsi_raw.rolling(window=k, min_periods=k).mean() * 100
        d_values = k_values.rolling(window=d, min_periods=d).mean()
        return k_values, d_values

    if len(df) < 21:
        logger.warning("Insufficient data for indicators: len(df) < 21")
        return None
    if df[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
        logger.warning("Input data contains NaN values")
        return None

    df["ma13"] = sma(df["close"], 13)
    df["ma21"] = sma(df["close"], 21)
    df["volume_sma"] = sma(df["volume"], 5)
    df["stochrsi_k"], df["stochrsi_d"] = stochrsi(df["close"], rsi_length=5, stoch_length=5, k=3, d=3)
    df["atr"] = atr(df["high"], df["low"], df["close"], period=14)
    df["adx"] = adx(df["high"], df["low"], df["close"], period=14)

    logger.info(f"Indicators: MA13={df['ma13'].iloc[-1]:.4f}, MA21={df['ma21'].iloc[-1]:.4f}, "
                f"StochRSI_K={df['stochrsi_k'].iloc[-1]:.2f}, StochRSI_D={df['stochrsi_d'].iloc[-1]:.2f}, "
                f"ATR={df['atr'].iloc[-1]:.4f}, ADX={df['adx'].iloc[-1]:.2f}")
    return df

# Generate signal
def get_signal(df):
    if len(df) < 21:
        logger.warning(f"Insufficient data: {len(df)} candles (need 21)")
        return None

    if df.iloc[-1][["ma13", "ma21", "stochrsi_k", "stochrsi_d", "atr", "adx"]].isna().any():
        logger.warning("NaN detected in indicators; skipping signal generation")
        return None

    latest = df.iloc[-1]
    ma_long = latest["ma13"] > latest["ma21"]
    ma_short = latest["ma13"] < latest["ma21"]
    stochrsi_long = latest["stochrsi_k"] > latest["stochrsi_d"]
    stochrsi_short = latest["stochrsi_k"] < latest["stochrsi_d"]
    volume_trend = latest['volume'] > VOLUME_THRESHOLD * df['volume'].rolling(window=5).mean().iloc[-1]
    adx_strong = latest["adx"] > ADX_THRESHOLD

    signal = None
    if ma_long and stochrsi_long and volume_trend and latest["stochrsi_k"] < 85 and adx_strong:
        signal = "long"
    elif ma_short and stochrsi_short and volume_trend and latest["stochrsi_k"] > 15 and adx_strong:
        signal = "short"

    logger.info(f"Signal check: MA_Long={ma_long}, MA_Short={ma_short}, "
                f"StochRSI_Long={stochrsi_long}, StochRSI_Short={stochrsi_short}, "
                f"Volume_Confirmed={volume_trend}, ADX={latest['adx']:.2f}, Signal={signal}")
    return signal

# Calculate contracts
def calculate_contracts(price):
    notional_value = MARGIN_COST_USD * LEVERAGE
    contracts = notional_value / (price * CONTRACT_SIZE)
    contracts = min(max(contracts, 0.001), MAX_CONTRACTS)
    contracts_rounded = max(round(contracts, 3), 0.001)
    actual_notional = contracts_rounded * price * CONTRACT_SIZE
    margin_used = actual_notional / LEVERAGE
    logger.info(f"Calculated contracts: {contracts_rounded:.3f} contracts, notional ${actual_notional:.2f}, margin used ${margin_used:.2f}")
    return contracts_rounded

# Calculate TP/SL
def calculate_tp_sl(price, volatility, side):
    atr_multiple = 2.0
    if volatility == 0:
        volatility = 0.01  # Fallback volatility
        logger.warning(f"Using default volatility: {volatility}")
    if USE_PNL_BASED:
        tp_distance = TP_PNL / LEVERAGE
        sl_distance = abs(SL_PNL) / LEVERAGE
    else:
        tp_distance = volatility * atr_multiple
        sl_distance = volatility * atr_multiple

    if side == "long":
        tp_price = price * (1 + tp_distance)
        sl_price = price * (1 - sl_distance)
    else:
        tp_price = price * (1 - tp_distance)
        sl_price = price * (1 + sl_distance)

    tp_price = round(tp_price, 8)
    sl_price = round(sl_price, 8)
    return tp_price, sl_price

# Place order with isolated margin for hedging
def place_order(side, contracts, entry_price):
    if not check_balance():
        logger.error("Insufficient balance for hedging")
        return None

    request_path = f"/api/v5/market/candles?instId={INSTRUMENT}&bar=1m&limit=20"
    response = public_request(request_path)
    volatility = 0.01  # Default fallback
    if response['code'] == '0' and response['data']:
        candles = response['data']
        closes = [float(c[4]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        df_volatility = pd.DataFrame({'close': closes, 'high': highs, 'low': lows})
        atr_value = calculate_indicators(df_volatility)['atr'].iloc[-1] if calculate_indicators(df_volatility) is not None else 0.01
        volatility = atr_value
        logger.info(f"Calculated volatility: {volatility:.4f} (ATR based)")

    tp_price, sl_price = calculate_tp_sl(entry_price, volatility, side)

    payload = {
        "instId": INSTRUMENT,
        "tdMode": "isolated",  # Changed to isolated for hedging
        "side": "buy" if side == "long" else "sell",
        "posSide": side,
        "ordType": "market",
        "sz": str(round(contracts, 2)),
        "lever": str(LEVERAGE),
        "attachAlgoOrds": [{
            "algoOrdType": "oco",
            "tpTriggerPx": str(tp_price),
            "tpOrdPx": "-1",
            "slTriggerPx": str(sl_price),
            "slOrdPx": "-1",
            "tpTriggerPxType": "last",
            "slTriggerPxType": "last"
        }]
    }
    logger.info(f"Attempting to place {side} order with isolated margin: {contracts:.3f} contracts, Entry=${entry_price:.4f}, TP=${tp_price:.4f}, SL=${sl_price:.4f}")
    response = private_request("POST", "/api/v5/trade/order", json.dumps(payload))
    if response["code"] == "0" and response["data"]:
        ord_id = response["data"][0]["ordId"]
        logger.info(f"Order placed successfully: ordId={ord_id}")
        return {"ordId": ord_id, "size": contracts, "entry_price": entry_price}
    else:
        logger.error(f"Order placement failed: {response.get('msg', 'Unknown error')}")
        return None

# Close position with isolated margin
def close_position(pos_side):
    payload = {"instId": INSTRUMENT, "mgnMode": "isolated", "posSide": pos_side}
    response = private_request("POST", "/api/v5/trade/close-position", json.dumps(payload))
    if response["code"] == "0":
        logger.info(f"Position {pos_side} closed successfully")
        return True
    logger.error(f"Failed to close position {pos_side}: {response['msg']}")
    return False

# Check positions
def check_positions():
    response = private_request("GET", f"/api/v5/account/positions?instId={INSTRUMENT}")
    positions = {}
    if response["code"] == "0":
        for pos in response["data"]:
            if pos["pos"] != "0":
                pos_side = pos["posSide"]
                positions[pos_side] = {
                    "side": pos_side,
                    "size": float(pos["pos"]),
                    "entry_price": float(pos["avgPx"])
                }
                logger.info(f"Current position: {pos_side}, {positions[pos_side]['size']:.3f} contracts")
    return positions

# Main loop with hedging logic
def main():
    global current_positions, last_signal, last_trade_time
    logger.info("Starting trading bot for DOGE-USDT-SWAP with hedging")
    if not check_unified_account():
        logger.error("Account is not in Unified Account mode")
        return
    if not check_balance():
        logger.error("Insufficient USDT balance for hedging")
        return
    set_leverage()

    while True:
        try:
            current_positions = check_positions()
            current_time = time.time()

            if current_time - last_trade_time < SIGNAL_COOLDOWN:
                time.sleep(POLLING_INTERVAL)
                continue

            if not check_market_conditions():
                time.sleep(POLLING_INTERVAL)
                continue

            df = get_historical_data()
            if df is None:
                time.sleep(POLLING_INTERVAL)
                continue
            df = calculate_indicators(df)
            if df is None:
                time.sleep(POLLING_INTERVAL)
                continue
            signal = get_signal(df)
            if signal:
                # Untuk hedging, izinkan pembukaan posisi meskipun posisi berlawanan sudah ada
                if signal not in current_positions:
                    price = df["close"].iloc[-1]
                    contracts = calculate_contracts(price)
                    order = place_order(signal, contracts, price)
                    if order:
                        current_positions[signal] = order
                        last_signal = signal
                        last_trade_time = current_time
                        logger.info(f"Hedging position {signal} opened")
                    else:
                        logger.warning("Failed to open hedging position")
                else:
                    logger.info(f"Position {signal} already exists, skipping")

            time.sleep(POLLING_INTERVAL)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(15)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        for pos_side in list(current_positions.keys()):
            close_position(pos_side)
        logger.info("Bot stopped, all positions closed")
    sys.stdout.flush()
