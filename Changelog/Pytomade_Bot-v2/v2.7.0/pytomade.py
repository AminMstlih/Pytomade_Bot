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
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from scipy.stats import linregress
import math
from datetime import datetime, timedelta
import pandas_ta as ta

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

if not all([API_KEY, SECRET_KEY, PASSPHRASE]):
    raise ValueError("Missing OKX API credentials in .env file")

# Configuration
BASE_URL = "https://www.okx.com"
SYMBOL = "DOGE-USDT-SWAP"
LEVERAGE = 10
INITIAL_CAPITAL = 10000
BASE_RISK_PER_TRADE = 0.01
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10
CONTRACT_SIZE = 10
MAX_DRAWDOWN = 0.15
MAX_POSITION_SIZE = 500
MIN_POSITION_SIZE = 1
API_RATE_LIMIT_PAUSE = 0.5
BACKTEST_LOOKBACK = 1440  # 1 day of 1-minute candles
train_interval = 3600  # 1 hour

# Logging setup
logging.basicConfig(level=logging.INFO, filename='trading_bot.log', filemode='a',
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Utility functions
def get_server_time():
    endpoint = "/api/v5/public/time"
    try:
        response = requests.get(BASE_URL + endpoint)
        response.raise_for_status()
        return str(float(response.json()["data"][0]["ts"]) / 1000.0)
    except Exception as e:
        logging.error(f"Failed to get server time: {e}")
        raise

def generate_signature(timestamp, method, request_path, body=""):
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def send_request(method, endpoint, body=None):
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            timestamp = get_server_time()
            body_json = json.dumps(body) if body else ""
            headers = {
                "OK-ACCESS-KEY": API_KEY,
                "OK-ACCESS-SIGN": generate_signature(timestamp, method, endpoint, body_json),
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": PASSPHRASE,
                "Content-Type": "application/json",
                "x-simulated-trading": "1"  # Remove for live trading
            }
            url = BASE_URL + endpoint
            response = requests.request(method, url, headers=headers, data=body_json)
            response.raise_for_status()
            time.sleep(API_RATE_LIMIT_PAUSE)
            return response.json()
        except requests.exceptions.RequestException as e:
            last_error = e
            logging.warning(f"Request failed: {e}, retrying ({attempt+1}/{max_retries})")
            time.sleep(2 * (2 ** attempt))
    logging.error(f"Request failed after {max_retries} attempts: {last_error}")
    return {"error": str(last_error)}

# Data fetching
def get_account_balance():
    endpoint = "/api/v5/account/balance?ccy=USDT"
    res = send_request("GET", endpoint)
    if "data" in res and res["data"]:
        balance = float(res["data"][0]["totalEq"])
        logging.info(f"Account balance: {balance:.2f} USDT")
        return balance
    logging.error("Failed to fetch account balance")
    return 0

def get_realtime_price():
    endpoint = f"/api/v5/market/ticker?instIdgrinding={SYMBOL}"
    res = send_request("GET", endpoint)
    if "data" in res and res["data"]:
        price = float(res["data"][0]["last"])
        logging.info(f"Current price: {price:.6f}")
        return price
    logging.error("Failed to fetch real-time price")
    return None

def get_prices(limit=100):
    endpoint = f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit={limit}"
    res = send_request("GET", endpoint)
    if "data" in res:
        candles = res["data"][::-1]
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        logging.info(f"Retrieved {len(df)} candles, last close: {df['close'].iloc[-1]:.6f}")
        return df
    logging.error("Failed to fetch historical prices")
    return pd.DataFrame()

# Trading control
def set_leverage(leverage=LEVERAGE):
    endpoint = "/api/v5/account/set-leverage"
    body = {"instId": SYMBOL, "lever": str(leverage), "mgnMode": "cross"}
    res = send_request("POST", endpoint, body)
    if "data" in res and res["data"]:
        logging.info(f"Leverage set to {leverage}x")
        return True
    logging.error(f"Failed to set leverage: {res}")
    return False

def place_order(side, pos_side, pos_size):
    if pos_size < MIN_POSITION_SIZE:
        logging.warning("Position size too small, skipping order")
        return {"msg": "Position size too small"}
    if pos_size > MAX_POSITION_SIZE:
        logging.warning(f"Position size {pos_size} > {MAX_POSITION_SIZE}, capping")
        pos_size = MAX_POSITION_SIZE
    body = {
        "instId": SYMBOL,
        "side": side,
        "posSide": pos_side,
        "ordType": "market",
        "sz": str(int(pos_size))
    }
    res = send_request("POST", "/api/v5/trade/order", body)
    if "data" in res and res["data"] and res["data"][0]["sCode"] == "0":
        logging.info(f"Order placed: {side} {pos_side} {pos_size} contracts")
        return res
    logging.error(f"Order placement failed: {res}")
    if res.get("data", [{}])[0].get("sCode") == "51202":
        logging.warning("Order size too large, retrying with reduced size")
        return place_order(side, pos_side, pos_size // 2)
    return res

def close_all_positions():
    global current_position
    if current_position is not None:
        side = "sell" if current_position["pos_side"] == "long" else "buy"
        pos_side = current_position["pos_side"]
        pos_size = current_position["pos_size"]
        logging.info(f"Closing position: {side} {pos_side} {pos_size} contracts")
        res = place_order(side, pos_side, pos_size)
        if "data" in res and res["data"] and res["data"][0]["sCode"] == "0":
            current_position = None
            logging.info("Position closed successfully")
        else:
            logging.error("Failed to close position")

# Feature engineering
def compute_features(df):
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(50).std() * np.sqrt(252 * 60 * 24)
    df["ma_short"] = df["close"].ewm(span=12).mean()
    df["ma_long"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ma_short"] - df["ma_long"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["rsi"] = 100 - (100 / (1 + df["returns"].clip(lower=0.0).rolling(14).mean() /
                           (df["returns"].clip(upper=0.0).rolling(14).mean().abs() + 1e-10)))
    df["momentum"] = df["close"].diff(10)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx["ADX_14"]
    return df.dropna()

# Model training
def train_lstm_model(df):
    prices = df["close"].values
    features = df[["returns", "volatility", "macd", "rsi", "momentum", "adx"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    window = 20
    X, y = [], []
    for i in range(len(features_scaled) - window):
        X.append(features_scaled[i:i+window])
        y.append(prices[i+window] / prices[i+window-1] - 1)
    X, y = np.array(X), np.array(y)
    model = Sequential([
        Input(shape=(window, features_scaled.shape[1])),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    logging.info("LSTM model trained")
    return model, window, scaler

def train_xgboost_model(df):
    features = df[["returns", "volatility", "macd", "rsi", "momentum", "adx"]].values
    target = df["close"].pct_change().shift(-1).values
    mask = ~np.isnan(target)
    features, target = features[mask], target[mask]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(features_scaled, target)
    logging.info("XGBoost model trained")
    return model, scaler

def train_arima_model(prices):
    try:
        model = ARIMA(prices, order=(5, 1, 0))
        model_fit = model.fit()
        logging.info("ARIMA model trained")
        return model_fit
    except Exception as e:
        logging.warning(f"ARIMA training failed: {e}")
        return None

# Signal generation
def hurst_exponent(time_series):
    lags = range(2, len(time_series)//2)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    reg = linregress(np.log(lags), np.log(tau))
    return reg.slope * 2

def detect_regime(df):
    hurst = hurst_exponent(df["close"].values[-50:])
    volatility = df["volatility"].iloc[-1]
    adx = df["adx"].iloc[-1]
    if hurst > 0.55 and adx > 25:
        return "trend"
    elif hurst < 0.45 or volatility > df["volatility"].mean() * 1.5:
        return "mean_reversion"
    return "neutral"

def generate_signals(df, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model, current_price):
    features = df[["returns", "volatility", "macd", "rsi", "momentum", "adx"]].values[-lstm_window:]
    features_scaled = lstm_scaler.transform(features)
    lstm_input = features_scaled.reshape(1, lstm_window, features.shape[1])
    lstm_pred_return = lstm_model.predict(lstm_input, verbose=0)[0][0]
    lstm_pred = current_price * (1 + lstm_pred_return)
    lstm_confidence = min(1.0, abs(lstm_pred - current_price) / current_price)

    xgb_features = df[["returns", "macd", "rsi", "momentum", "adx"]].iloc[-1:].values
    xgb_features_scaled = xgb_scaler.transform(xgb_features)
    xgb_pred_return = xgb_model.predict(xgb_features_scaled)[0]
    xgb_pred = current_price * (1 + xgb_pred_return)
    xgb_confidence = min(1.0, abs(xgb_pred - current_price) / current_price)

    arima_pred = None
    if arima_model:
        try:
            arima_forecast = arima_model.forecast(steps=1)
            arima_pred = arima_forecast[0]
            arima_confidence = min(1.0, abs(arima_pred - current_price) / current_price)
        except:
            arima_pred = None
            arima_confidence = 0.0
    else:
        arima_confidence = 0.0

    macd = df["macd"].iloc[-1]
    macd_signal = df["macd_signal"].iloc[-1]
    macd_strength = abs(macd) / current_price
    rsi = df["rsi"].iloc[-1]

    regime = detect_regime(df)
    logging.info(f"Regime: {regime}, Hurst: {hurst_exponent(df['close'].values[-50:]):.2f}, Volatility: {volatility:.4f}, ADX: {adx:.2f}")

    signals = {"long": 0.0, "short": 0.0}
    weights = {"lstm": 0.4, "xgb": 0.3, "arima": 0.2, "macd": 0.1} if arima_pred else {"lstm": 0.5, "xgb": 0.4, "macd": 0.1}

    if lstm_pred > current_price:
        signals["long"] += lstm_confidence * weights["lstm"]
    else:
        signals["short"] += lstm_confidence * weights["lstm"]

    if xgb_pred > current_price:
        signals["long"] += xgb_confidence * weights["xgb"]
    else:
        signals["short"] += xgb_confidence * weights["xgb"]

    if arima_pred and arima_pred > current_price:
        signals["long"] += arima_confidence * weights["arima"]
    elif arima_pred:
        signals["short"] += arima_confidence * weights["arima"]

    if macd > macd_signal and rsi < 70:
        signals["long"] += macd_strength * weights["macd"]
    elif macd < macd_signal and rsi > 30:
        signals["short"] += macd_strength * weights["macd"]

    if regime == "mean_reversion":
        signals["long"] *= 0.5
        signals["short"] *= 0.5

    logging.info(f"Signals: LSTM={lstm_pred:.2f} ({lstm_confidence:.2f}), XGB={xgb_pred:.2f} ({xgb_confidence:.2f}), "
                f"ARIMA={arima_pred:.2f} ({arima_confidence:.2f}), MACD={macd_strength:.4f}, Total={signals}")
    return "long" if signals["long"] > signals["short"] and signals["long"] > 0.3 else "short" if signals["short"] > signals["long"] and signals["short"] > 0.3 else "neutral"

# Backtest validation
def backtest_signals(df, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model):
    signals = [generate_signals(df.iloc[:i+1], lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model, df["close"].iloc[i]) for i in range(len(df) - 1)]
    df_signals = pd.DataFrame({"signal": signals, "close": df["close"].iloc[:-1], "high": df["high"].iloc[:-1], "low": df["low"].iloc[:-1]})

    position = None
    strategy_returns = []
    fee_rate = 0.0005  # 0.05%
    slippage_rate = 0.0001  # 0.01%

    for i in range(len(df_signals)):
        signal = df_signals["signal"].iloc[i]
        close_price = df_signals["close"].iloc[i]
        high_price = df_signals["high"].iloc[i]
        low_price = df_signals["low"].iloc[i]

        if position is None:
            if signal != "neutral":
                position = {"side": signal, "entry_price": close_price}
                strategy_returns.append(-fee_rate - slippage_rate)
        else:
            if position["side"] == "long":
                sl_price = position["entry_price"] * (1 - STOP_LOSS_PCT)
                tp_price = position["entry_price"] * (1 + TAKE_PROFIT_PCT)
                if low_price <= sl_price:
                    exit_price = sl_price
                    return_pct = (exit_price / position["entry_price"]) - 1
                    strategy_returns.append(return_pct - fee_rate - slippage_rate)
                    position = None
                elif high_price >= tp_price:
                    exit_price = tp_price
                    return_pct = (exit_price / position["entry_price"]) - 1
                    strategy_returns.append(return_pct - fee_rate - slippage_rate)
                    position = None
                elif signal != position["side"]:
                    exit_price = close_price
                    return_pct = (exit_price / position["entry_price"]) - 1
                    strategy_returns.append(return_pct - fee_rate - slippage_rate)
                    position = None
                else:
                    strategy_returns.append(0)
            else:  # short
                sl_price = position["entry_price"] * (1 + STOP_LOSS_PCT)
                tp_price = position["entry_price"] * (1 - TAKE_PROFIT_PCT)
                if high_price >= sl_price:
                    exit_price = sl_price
                    return_pct = 1 - (exit_price / position["entry_price"])
                    strategy_returns.append(return_pct - fee_rate - slippage_rate)
                    position = None
                elif low_price <= tp_price:
                    exit_price = tp_price
                    return_pct = 1 - (exit_price / position["entry_price"])
                    strategy_returns.append(return_pct - fee_rate - slippage_rate)
                    position = None
                elif signal != position["side"]:
                    exit_price = close_price
                    return_pct = 1 - (exit_price / position["entry_price"])
                    strategy_returns.append(return_pct - fee_rate - slippage_rate)
                    position = None
                else:
                    strategy_returns.append(0)

    strategy_returns = pd.Series(strategy_returns)
    if len(strategy_returns) == 0:
        return False
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 60 * 24) if strategy_returns.std() != 0 else 0
    cumulative_returns = (1 + strategy_returns).cumprod()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    profit_factor = strategy_returns[strategy_returns > 0].sum() / abs(strategy_returns[strategy_returns < 0].sum()) if strategy_returns[strategy_returns < 0].sum() != 0 else np.inf
    win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns[strategy_returns != 0]) if len(strategy_returns[strategy_returns != 0]) > 0 else 0
    logging.info(f"Backtest: Sharpe={sharpe:.2f}, Max Drawdown={max_drawdown:.2f}, Profit Factor={profit_factor:.2f}, Win Rate={win_rate:.2f}")
    return sharpe > 1.0 and max_drawdown < 0.1 and profit_factor > 1.5 and win_rate > 0.5

# Main loop
if __name__ == "__main__":
    logging.info("Starting Jim Simons-inspired trading bot...")
    res = send_request("GET", "/api/v5/account/balance?ccy=USDT")
    if "error" in res:
        logging.error(f"API credential validation failed: {res['error']}")
    else:
        set_leverage()
        last_train_time = time.time()
        current_position = None
        trades = []
        account_value = get_account_balance() or INITIAL_CAPITAL
        peak_value = account_value

        while True:
            try:
                # Retrain models periodically
                if time.time() - last_train_time > train_interval:
                    df = get_prices(limit=BACKTEST_LOOKBACK)
                    if not df.empty:
                        df = compute_features(df)
                        lstm_model, lstm_window, lstm_scaler = train_lstm_model(df)
                        xgb_model, xgb_scaler = train_xgboost_model(df)
                        arima_model = train_arima_model(df["close"].values)
                        last_train_time = time.time()
                        logging.info("Models retrained.")

                # Fetch latest data
                df = get_prices(limit=100)
                if df.empty:
                    time.sleep(60)
                    continue
                df = compute_features(df)
                current_price = get_realtime_price()
                if not current_price:
                    continue

                account_value = get_account_balance()
                peak_value = max(peak_value, account_value)
                logging.info(f"Account: {account_value:,.2f}, Peak: {peak_value:,.2f}")

                volatility = df["volatility"].iloc[-1]
                risk_per_trade = BASE_RISK_PER_TRADE * (1 + volatility * 100)
                risk_per_trade = max(0.005, min(0.02, risk_per_trade))

                signal = generate_signals(df, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model, current_price)

                risk_amount = account_value * risk_per_trade
                entry_price = current_price
                if signal == "long":
                    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                elif signal == "short":
                    stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)
                else:
                    stop_loss_price = None
                if stop_loss_price and abs(entry_price - stop_loss_price) > 0:
                    pos_size = max(MIN_POSITION_SIZE, int(risk_amount / (abs(entry_price - stop_loss_price) * CONTRACT_SIZE)))
                    pos_size = min(pos_size, MAX_POSITION_SIZE)
                else:
                    pos_size = 0
                logging.info(f"Signal: {signal}, Risk: {risk_per_trade:.5f}, Pos Size: {pos_size}")

                if current_position is not None:
                    entry_price = current_position["entry_price"]
                    entry_time = current_position["entry_time"]
                    if time.time() - entry_time < 60:
                        continue
                    if current_position["pos_side"] == "long":
                        if current_price <= entry_price * (1 - STOP_LOSS_PCT):
                            logging.info("Stop-loss triggered: long")
                            close_all_positions()
                            if trades and trades[-1]["exit"] is None:
                                trades[-1]["exit"] = current_price
                        elif current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                            logging.info("Take-profit triggered: long")
                            close_all_positions()
                            if trades and trades[-1]["exit"] is None:
                                trades[-1]["exit"] = current_price
                    else:
                        if current_price >= entry_price * (1 + STOP_LOSS_PCT):
                            logging.info("Stop-loss triggered: short")
                            close_all_positions()
                            if trades and trades[-1]["exit"] is None:
                                trades[-1]["exit"] = current_price
                        elif current_price <= entry_price * (1 - TAKE_PROFIT_PCT):
                            logging.info("Take-profit triggered: short")
                            close_all_positions()
                            if trades and trades[-1]["exit"] is None:
                                trades[-1]["exit"] = current_price

                if signal != "neutral":
                    desired_pos_side = "long" if signal == "long" else "short"
                    if current_position is None or current_position["pos_side"] != desired_pos_side:
                        if current_position is not None:
                            logging.info("Closing existing position")
                            close_all_positions()
                            if trades and trades[-1]["exit"] is None:
                                trades[-1]["exit"] = current_price
                        if pos_size > 0:
                            side = "buy" if signal == "long" else "sell"
                            place_order(side, desired_pos_side, pos_size)
                            current_position = {
                                "pos_side": desired_pos_side,
                                "entry_price": current_price,
                                "pos_size": pos_size,
                                "entry_time": time.time()
                            }
                            trades.append({"entry": current_price, "exit": None, "signal": signal})

                time.sleep(60)
            except Exception as e:
                logging.error(f"Loop error: {e}")
                time.sleep(60)
