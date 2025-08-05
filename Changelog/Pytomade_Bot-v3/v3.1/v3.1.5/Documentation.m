# Pytomade Bot v3.1.5 Documentation

## Overview
Pytomade Bot v3.1.5 is a Python-based trading bot designed for trading DOGE perpetual futures on OKX with isolated margin mode, supporting hedging (simultaneous long and short positions). It integrates technical indicators (MA, StochRSI, ADX, ATR) for signal generation, strict risk management, and robust error handling for live trading. The bot operates in a single file for simplicity and is optimized for environments like Google Colab, GitHub Codespaces, or Replit.

## Key Features
- **Trading Strategy**: Combines MA13/MA21 crossover, StochRSI, ADX, and volume trend confirmation for signal generation.
- **Hedging Support**: Allows simultaneous long and short positions using isolated margin mode.
- **Risk Management**:
  - Fixed margin: $2 per position with 15x leverage.
  - Maximum position size: 10,000 USDT.
  - Take Profit (TP): +21% PnL or ATR-based.
  - Stop Loss (SL): -15% PnL or ATR-based.
  - Optional trailing stop: Activates at 5% profit, trails by 5%.
- **Market Filters**: Ensures trading only under favorable conditions (min 24h volume, max spread, min volatility).
- **Logging**: Comprehensive logging to console and file (`trading_bot.log`) for debugging and monitoring.
- **API Integration**: Uses OKX v5 API with HMAC authentication and rate limit handling.

## Key Parameters
| Parameter                | Value                     | Description                                      |
|--------------------------|---------------------------|--------------------------------------------------|
| `INSTRUMENT`             | DOGE-USDT-SWAP            | Trading pair.                                    |
| `LEVERAGE`               | 15                        | Leverage for isolated margin.                    |
| `MARGIN_COST_USD`        | 2                         | Margin per position (USDT).                      |
| `CONTRACT_SIZE`          | 1000                      | 1 contract = 1000 DOGE.                          |
| `MAX_CONTRACTS`          | 10                        | Max contracts per position.                      |
| `TP_PNL` / `SL_PNL`      | +0.21 / -0.15             | TP/SL based on % PnL (optional ATR-based).       |
| `TRAILING_STOP`          | True                      | Trailing stop enabled (5% activation, 5% trail). |
| `MIN_24H_VOLUME`         | 1,000,000 USDT            | Min 24h volume filter.                           |
| `MAX_SPREAD`             | 0.1%                      | Max bid-ask spread filter.                       |
| `VOLATILITY_THRESHOLD`   | 0.2%                      | Min 24h price range filter.                      |
| `ADX_THRESHOLD`          | 25                        | Min ADX for trend confirmation.                  |
| `POLLING_INTERVAL`       | 10 seconds                | Time between loop iterations.                    |
| `SIGNAL_COOLDOWN`        | 60 seconds                | Min time between trades.                         |

## Code Structure
- **Imports & Setup**: Uses `requests`, `pandas`, `numpy`, and `logging` for API calls, data processing, and logging.
- **API Functions**:
  - `get_server_time()`: Fetches OKX server time for API authentication.
  - `generate_headers()`: Creates HMAC-signed headers for private API calls.
  - `private_request()` / `public_request()`: Handles API requests with retries and rate limiting.
- **Account & Market Checks**:
  - `check_balance()`: Ensures sufficient USDT for two positions (hedging).
  - `check_unified_account()`: Verifies Unified Account mode.
  - `set_leverage()`: Sets 15x leverage for isolated margin.
  - `check_market_conditions()`: Validates spread, volume, and volatility.
- **Data & Indicators**:
  - `get_historical_data()`: Fetches 1-minute candles (100 bars).
  - `calculate_indicators()`: Computes MA13, MA21, StochRSI, ATR, ADX.
- **Trading Logic**:
  - `get_signal()`: Generates long/short signals based on indicators.
  - `calculate_contracts()`: Computes position size within constraints.
  - `calculate_tp_sl()`: Sets TP/SL prices (PnL or ATR-based).
  - `place_order()`: Places market orders with OCO (TP/SL) for hedging.
  - `close_position()`: Closes positions by side (long/short).
  - `check_positions()`: Monitors current positions.
- **Main Loop**:
  - Runs continuously, checking market conditions, generating signals, and managing positions.
  - Handles hedging by allowing opposite positions.
  - Includes error handling and signal cooldown.

## Dependencies
- Python libraries: `requests`, `pandas`, `numpy`, `python-dotenv`.
- OKX API credentials: `OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_PASSPHRASE` (stored in `.env`).

## Notes for Future Development
- **Versioning Strategy**: Each version should modify only one specific feature or function to maintain clean, documented changes.
- **Potential Improvements**:
  - Add dynamic ATR multiplier based on market conditions.
  - Implement machine learning (e.g., Linear Regression, XGBoost) for signal enhancement.
  - Enhance logging with performance metrics (e.g., win/loss ratio, ROI).
- **Known Considerations**:
  - Ensure API rate limits are respected to avoid bans.
  - Validate candle data to prevent NaN or invalid values from affecting indicators.
  - Monitor hedging behavior to avoid unintended position overlaps.

## Usage
1. Set up `.env` file with OKX API credentials.
2. Install dependencies: `pip install requests pandas numpy python-dotenv`.
3. Run in a supported environment (Google Colab, GitHub Codespaces, Replit).
4. Monitor `trading_bot.log` for debugging and performance tracking.
