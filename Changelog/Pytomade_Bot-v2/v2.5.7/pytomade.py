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
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from scipy.stats import linregress
import math
from datetime import datetime, timedelta

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# Validate credentials
if not all([API_KEY, SECRET_KEY, PASSPHRASE]):
    raise ValueError("Missing OKX API credentials in .env file")

# Configuration
BASE_URL = "https://www.okx.com"
SYMBOL = "DOGE-USDT-SWAP"
LEVERAGE = 10
INITIAL_CAPITAL = 10000
BASE_RISK_PER_TRADE = 0.01
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
CONTRACT_SIZE = 10
MAX_DRAWDOWN = 0.15
MAX_POSITION_SIZE = 500
MIN_POSITION_SIZE = 1
API_RATE_LIMIT_PAUSE = 0.5
BACKTEST_LOOKBACK = 200

# Logging setup (console only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Global state
current_position = None
trades = []

# --- UTILITY FUNCTIONS ---
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
                "x-simulated-trading": "1"
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

# --- DATA FETCHING ---
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
    endpoint = f"/api/v5/market/ticker?instId={SYMBOL}"
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
        # Updated to handle all 9 columns
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df["close"] = df["close"].astype(float)
        logging.info(f"Retrieved {len(df)} candles, last close: {df['close'].iloc[-1]:.6f}")
        return df
    logging.error("Failed to fetch historical prices")
    return pd.DataFrame()

# --- TRADING CONTROL ---
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

# --- FEATURE ENGINEERING ---
def compute_features(df):
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252 * 60 * 24)
    df["ma_short"] = df["close"].ewm(span=12).mean()
    df["ma_long"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ma_short"] - df["ma_long"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["rsi"] = 100 - (100 / (1 + df["returns"].clip(lower=0.0).rolling(14).mean() /
                           (-df["returns"].clip(upper=0.0).rolling(14).mean())))
    df["momentum"] = df["close"].diff(10)
    return df.dropna()

# --- MODEL TRAINING ---
def train_lstm_model(df):
    prices = df["close"].values
    features = df[["returns", "volatility", "macd", "rsi", "momentum"]].values
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
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)  # Reduced epochs for Colab
    logging.info("LSTM model trained")
    return model, window, scaler

def train_xgboost_model(df):
    features = df[["returns", "volatility", "macd", "rsi", "momentum"]].values
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

# --- SIGNAL GENERATION ---
def hurst_exponent(time_series):
    lags = range(2, len(time_series)//2)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    reg = linregress(np.log(lags), np.log(tau))
    return reg.slope * 2

def detect_regime(df):
    hurst = hurst_exponent(df["close"].values[-50:])
    volatility = df["volatility"].iloc[-1]
    vol_threshold = df["volatility"].mean()
    if hurst > 0.55 and volatility < vol_threshold:
        return "trend"
    elif hurst < 0.45 or volatility > vol_threshold * 1.5:
        return "mean_reversion"
    return "neutral"

def generate_signals(df, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model, current_price):
    features = df[["returns", "volatility", "macd", "rsi", "momentum"]].values[-lstm_window:]
    features_scaled = lstm_scaler.transform(features)
    lstm_input = features_scaled.reshape(1, lstm_window, features.shape[1])
    lstm_pred_return = lstm_model.predict(lstm_input, verbose=0)[0][0]
    lstm_pred = current_price * (1 + lstm_pred_return)
    lstm_confidence = min(1.0, abs(lstm_pred - current_price) / current_price)

    xgb_features = df[["returns", "macd", "rsi", "momentum"]].iloc[-1:].values
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
    logging.info(f"Regime: {regime}, Hurst: {hurst_exponent(df['close'].values[-50:]):.2f}, Volatility: {macd:.4f}")

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

# --- BACKTEST VALIDATION ---
def backtest_signals(df, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model):
    signals = []
    for i in range(len(df) - 1):
        current_price = df["close"].iloc[i]
        signal = generate_signals(df.iloc[:i+1], lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model, current_price)
        signals.append(signal)
    df_signals = pd.DataFrame({"signal": signals, "price": df["close"].iloc[:-1]})
    df_signals["return"] = df["close"].pct_change().shift(-1)
    df_signals["strategy_return"] = df_signals.apply(
        lambda x: x["return"] if x["signal"] == "long" else -x["return"] if x["signal"] == "short" else 0, axis=1)
    sharpe = df_signals["strategy_return"].mean() / df_signals["strategy_return"].std() * np.sqrt(252 * 60 * 24)
    win_rate = len(df_signals[df_signals["strategy_return"] > 0]) / len(df_signals[df_signals["strategy_return"] != 0]) if df_signals["strategy_return"].ne(0).sum() > 0 else 0
    logging.info(f"Backtest: Sharpe={sharpe:.2f}, Win Rate={win_rate:.2f}")
    return sharpe > 0.5 and win_rate > 0.5

# --- MAIN STRATEGY LOOP ---
def ml_regime_strategy():
    global current_position, trades
    logging.info("Starting Jim Simons-inspired trading bot...")

    res = send_request("GET", "/api/v5/account/balance?ccy=USDT")
    if "error" in res:
        logging.error(f"API credential validation failed: {res['error']}")
        return

    if not set_leverage():
        return

    df = get_prices(limit=BACKTEST_LOOKBACK)
    if df.empty:
        logging.error("No initial price data")
        return
    df = compute_features(df)

    lstm_model, lstm_window, lstm_scaler = train_lstm_model(df)
    xgb_model, xgb_scaler = train_xgboost_model(df)
    arima_model = train_arima_model(df["close"].values)
    if lstm_model is None or xgb_model is None:
        logging.error("Model training failed")
        return

    if not backtest_signals(df, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model):
        logging.error("Backtest validation failed")
        return

    account_value = get_account_balance() or INITIAL_CAPITAL
    peak_value = account_value
    loop_count = 0

    while loop_count < 5:
        try:
            df_new = get_prices(limit=100)
            if df_new.empty:
                logging.warning("No price data, retrying...")
                time.sleep(10)
                continue
            df_new = compute_features(df_new)
            current_price = get_realtime_price()
            if not current_price:
                continue

            account_value = get_account_balance()
            peak_value = max(peak_value, account_value)
            logging.info(f"Account: {account_value:,.2f}, Peak: {peak_value:,.2f}")
            volatility = df_new["volatility"].iloc[-1]
            risk_per_trade = BASE_RISK_PER_TRADE * (1 + volatility * 100)
            risk_per_trade = max(0.005, min(0.02, risk_per_trade))

            signal = generate_signals(df_new, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model, current_price)

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
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from scipy.stats import linregress
import math
from datetime import datetime, timedelta

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# Validate credentials
if not all([API_KEY, SECRET_KEY, PASSPHRASE]):
    raise ValueError("Missing OKX API credentials in .env file")

# Configuration
BASE_URL = "https://www.okx.com"
SYMBOL = "DOGE-USDT-SWAP"
LEVERAGE = 10
INITIAL_CAPITAL = 10000
BASE_RISK_PER_TRADE = 0.01
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
CONTRACT_SIZE = 10
MAX_DRAWDOWN = 0.15
MAX_POSITION_SIZE = 500
MIN_POSITION_SIZE = 1
API_RATE_LIMIT_PAUSE = 0.5
BACKTEST_LOOKBACK = 200

# Logging setup (console only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Global state
current_position = None
trades = []

# --- UTILITY FUNCTIONS ---
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
                "x-simulated-trading": "1"
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

# --- DATA FETCHING ---
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
    endpoint = f"/api/v5/market/ticker?instId={SYMBOL}"
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
        # Updated to handle all 9 columns
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df["close"] = df["close"].astype(float)
        logging.info(f"Retrieved {len(df)} candles, last close: {df['close'].iloc[-1]:.6f}")
        return df
    logging.error("Failed to fetch historical prices")
    return pd.DataFrame()

# --- TRADING CONTROL ---
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

# --- FEATURE ENGINEERING ---
def compute_features(df):
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252 * 60 * 24)
    df["ma_short"] = df["close"].ewm(span=12).mean()
    df["ma_long"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ma_short"] - df["ma_long"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["rsi"] = 100 - (100 / (1 + df["returns"].clip(lower=0.0).rolling(14).mean() /
                           (-df["returns"].clip(upper=0.0).rolling(14).mean())))
    df["momentum"] = df["close"].diff(10)
    return df.dropna()

# --- MODEL TRAINING ---
def train_lstm_model(df):
    prices = df["close"].values
    features = df[["returns", "volatility", "macd", "rsi", "momentum"]].values
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
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)  # Reduced epochs for Colab
    logging.info("LSTM model trained")
    return model, window, scaler

def train_xgboost_model(df):
    features = df[["returns", "volatility", "macd", "rsi", "momentum"]].values
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

# --- SIGNAL GENERATION ---
def hurst_exponent(time_series):
    lags = range(2, len(time_series)//2)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    reg = linregress(np.log(lags), np.log(tau))
    return reg.slope * 2

def detect_regime(df):
    hurst = hurst_exponent(df["close"].values[-50:])
    volatility = df["volatility"].iloc[-1]
    vol_threshold = df["volatility"].mean()
    if hurst > 0.55 and volatility < vol_threshold:
        return "trend"
    elif hurst < 0.45 or volatility > vol_threshold * 1.5:
        return "mean_reversion"
    return "neutral"

def generate_signals(df, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model, current_price):
    features = df[["returns", "volatility", "macd", "rsi", "momentum"]].values[-lstm_window:]
    features_scaled = lstm_scaler.transform(features)
    lstm_input = features_scaled.reshape(1, lstm_window, features.shape[1])
    lstm_pred_return = lstm_model.predict(lstm_input, verbose=0)[0][0]
    lstm_pred = current_price * (1 + lstm_pred_return)
    lstm_confidence = min(1.0, abs(lstm_pred - current_price) / current_price)

    xgb_features = df[["returns", "macd", "rsi", "momentum"]].iloc[-1:].values
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
    logging.info(f"Regime: {regime}, Hurst: {hurst_exponent(df['close'].values[-50:]):.2f}, Volatility: {macd:.4f}")

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

# --- BACKTEST VALIDATION ---
def backtest_signals(df, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model):
    signals = []
    for i in range(len(df) - 1):
        current_price = df["close"].iloc[i]
        signal = generate_signals(df.iloc[:i+1], lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model, current_price)
        signals.append(signal)
    df_signals = pd.DataFrame({"signal": signals, "price": df["close"].iloc[:-1]})
    df_signals["return"] = df["close"].pct_change().shift(-1)
    df_signals["strategy_return"] = df_signals.apply(
        lambda x: x["return"] if x["signal"] == "long" else -x["return"] if x["signal"] == "short" else 0, axis=1)
    sharpe = df_signals["strategy_return"].mean() / df_signals["strategy_return"].std() * np.sqrt(252 * 60 * 24) if df_signals["strategy_return"].std() != 0 else 0
    win_rate = len(df_signals[df_signals["strategy_return"] > 0]) / len(df_signals[df_signals["strategy_return"] != 0]) if df_signals["strategy_return"].ne(0).sum() > 0 else 0
    logging.info(f"Backtest: Sharpe={sharpe:.2f}, Win Rate={win_rate:.2f}")
    return sharpe > 0.5 and win_rate > 0.5

# --- MAIN STRATEGY LOOP ---
def ml_regime_strategy():
    global current_position, trades
    logging.info("Starting Jim Simons-inspired trading bot...")

    res = send_request("GET", "/api/v5/account/balance?ccy=USDT")
    if "error" in res:
        logging.error(f"API credential validation failed: {res['error']}")
        return

    if not set_leverage():
        return

    df = get_prices(limit=BACKTEST_LOOKBACK)
    if df.empty:
        logging.error("No initial price data")
        return
    df = compute_features(df)

    lstm_model, lstm_window, lstm_scaler = train_lstm_model(df)
    xgb_model, xgb_scaler = train_xgboost_model(df)
    arima_model = train_arima_model(df["close"].values)
    if lstm_model is None or xgb_model is None:
        logging.error("Model training failed")
        return

    if not backtest_signals(df, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model):
        logging.error("Backtest validation failed")
        return

    account_value = get_account_balance() or INITIAL_CAPITAL
    peak_value = account_value
    loop_count = 0

    while loop_count < 5:
        try:
            df_new = get_prices(limit=100)
            if df_new.empty:
                logging.warning("No price data, retrying...")
                time.sleep(10)
                continue
            df_new = compute_features(df_new)
            current_price = get_realtime_price()
            if not current_price:
                continue

            account_value = get_account_balance()
            peak_value = max(peak_value, account_value)
            logging.info(f"Account: {account_value:,.2f}, Peak: {peak_value:,.2f}")
            volatility = df_new["volatility"].iloc[-1]
            risk_per_trade = BASE_RISK_PER_TRADE * (1 + volatility * 100)
            risk_per_trade = max(0.005, min(0.02, risk_per_trade))

            signal = generate_signals(df_new, lstm_model, lstm_window, lstm_scaler, xgb_model, xgb_scaler, arima_model, current_price)

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
                        side = "buy" if signal == "long" else "sell" if signal == "short" else None
                        if side:
                            place_order(side, desired_pos_side, pos_size)
                            current_position = {
                                "pos_side": desired_pos_side,
                                "entry_price": current_price,
                                "pos_size": pos_size,
                                "entry_time": time.time()
                            }
                            trades.append({"entry": current_price, "exit": None, "signal": signal})

            loop_count += 1
        except Exception as e:
            logging.error(f"Loop error: {e}")
        time.sleep(15)

if __name__ == "__main__":
    logging.info(f"API_KEY: {API_KEY[:4]}..., PASSPHRASE: {PASSPHRASE[:4]}...")
    try:
        ml_regime_strategy()
    except KeyboardInterrupt:
        logging.warning("Bot stopped by user")
        close_all_positions()
                        current_position = {
                            "pos_side": desired_pos_side,
                            "entry_price": current_price,
                            "pos_size": pos_size,
                            "entry_time": time.time()
                        }
                        trades.append({"entry": current_price, "exit": None, "signal": signal})

            loop_count += 1
        except Exception as e:
            logging.error(f"Loop error: {e}")
        time.sleep(15)

if __name__ == "__main__":
    logging.info(f"API_KEY: {API_KEY[:4]}..., PASSPHRASE: {PASSPHRASE[:4]}...")
    try:
        ml_regime_strategy()
    except KeyboardInterrupt:
        logging.warning("Bot stopped by user")
        close_all_positions()
