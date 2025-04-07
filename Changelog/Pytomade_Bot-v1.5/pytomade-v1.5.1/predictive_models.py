import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

from utils_indicators import split_sequence, normalize_prices

# ===============================
# ARIMA Model
# ===============================
def predict_arima(prices, steps=1):
    try:
        model = ARIMA(prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast[-1]
    except Exception as e:
        print(f"ARIMA prediction error: {e}")
        return None

# ===============================
# XGBoost Model
# ===============================
def predict_xgboost(prices, n_steps=5):
    try:
        X, y = split_sequence(prices, n_steps)
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)
        x_input = np.array(prices[-n_steps:]).reshape(1, -1)
        return model.predict(x_input)[0]
    except Exception as e:
        print(f"XGBoost prediction error: {e}")
        return None

# ===============================
# LSTM Model
# ===============================
def predict_lstm(prices, n_steps=5, epochs=20):
    try:
        # Normalisasi harga
        prices, scaler = normalize_prices(prices)

        # Sequence
        X, y = split_sequence(prices, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(X, y, epochs=epochs, verbose=0, callbacks=[EarlyStopping(patience=5)])

        x_input = np.array(prices[-n_steps:]).reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        return scaler.inverse_transform(yhat)[0][0]
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return None
