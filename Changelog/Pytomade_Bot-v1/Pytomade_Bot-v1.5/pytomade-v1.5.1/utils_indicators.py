import numpy as np
import logging
import os
from datetime import datetime

# Setup logging
LOG_PATH = "bot.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Fungsi untuk logging cepat
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    logging.info(msg)

# Moving Average

def moving_average(prices, period):
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

# MACD (Moving Average Convergence Divergence)
def macd(prices, short_period=12, long_period=26, signal_period=9):
    if len(prices) < long_period + signal_period:
        return None, None, None

    short_ema = np.convolve(prices, np.ones(short_period)/short_period, mode='valid')
    long_ema = np.convolve(prices, np.ones(long_period)/long_period, mode='valid')
    ema_diff = short_ema[-len(long_ema):] - long_ema  # MACD line

    if len(ema_diff) < signal_period:
        return None, None, None

    signal_line = np.convolve(ema_diff, np.ones(signal_period)/signal_period, mode='valid')
    macd_line = ema_diff[-len(signal_line):]
    histogram = macd_line - signal_line

    return macd_line[-1], signal_line[-1], histogram[-1]  # ambil data terakhir

# Fungsi normalisasi harga

def normalize_prices(prices):
    prices = np.array(prices)
    return (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)

# Fungsi split dataset

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
