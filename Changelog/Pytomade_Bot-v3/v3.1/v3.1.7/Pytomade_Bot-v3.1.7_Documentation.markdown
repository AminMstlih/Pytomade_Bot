# Pytomade_Bot-v3.1.7 Documentation

## Overview
**Pytomade_Bot-v3.1.7** is a Python-based trading bot designed for OKX perpetual futures, specifically for the `KAITO-USDT-SWAP` pair. The bot implements USDT-based position sizing, combining MA13/MA21 crossover, StochRSI confirmation, and ADX trend filtering, with strict risk management (TP1, TP2, SL based on % PNL). This version focuses on fixing order placement issues related to decimals and the unsupported `tgtCcy="quote_ccy"` parameter, ensuring robust sizing in contracts.

## Development Goals
- **Primary Objective**: Implement accurate USDT-based position sizing for `KAITO-USDT-SWAP` using OKX API, with a $15 position size at 15x leverage (margin=$1).
- **Constraints**: 
  - Single-file implementation for simplicity.
  - Customizable thresholds (e.g., TP1=5%, TP2=7%, SL=-5%).
  - Strict position sizing (max $10,000).
  - Continuous operation with logging for debugging.
  - Adherence to OKX API rules ([docs](https://okx.com/api/docs-v5/en/)).
- **Version Focus**: Resolve order placement failures due to decimal precision and `tgtCcy` errors, ensuring compatibility with `KAITO-USDT-SWAP`.

## Development Journey

### Initial Challenges
1. **Decimal Precision Issues**:
   - **Problem**: Orders failed with errors like `"Invalid quantity"` when `sz` included decimals (e.g., `sz="15.00"`). The user noted that rounding to 0 decimals (e.g., `sz="15"`) resolved earlier failures.
   - **Cause**: OKX’s API requires `sz` to align with `lotSz` (lot size), which is often an integer (e.g., `lotSz=1`) for swap contracts when using `tgtCcy="quote_ccy"`.
   - **Initial Approach**: In early iterations, `sz` was rounded to 2 decimals (`round(size, 2)`), leading to errors for instruments requiring integer sizes.

2. **tgtCcy Error**:
   - **Problem**: Logs showed `"Order placement failed: All operations failed (code: 1, details: The instrument corresponding to this KAITO-USDT-SWAP does not support the tgtCcy parameter)"` at 2025-08-09 15:18:17 and 15:18:30.
   - **Cause**: `KAITO-USDT-SWAP` does not support `tgtCcy="quote_ccy"` (USDT-based sizing). OKX requires `sz` in base currency (KAITO contracts) for this pair.
   - **Impact**: The bot couldn’t place orders directly in USDT, necessitating a conversion to contracts.

### Solutions Implemented
1. **Dynamic Instrument Details**:
   - Added `get_instrument_details` to fetch `ctVal` (contract value in KAITO), `lotSz`, and `minSz` from `/api/v5/public/instruments`.
   - Defaults: `CT_VAL=1`, `LOT_SIZE=1`, `MIN_SIZE=1` if the API call fails.
   - Validation: If `ctVal`, `lotSz`, or `minSz` ≤ 0, revert to defaults and log a warning.

2. **Contract-Based Sizing**:
   - Removed `tgtCcy="quote_ccy"` from the order payload and `attachAlgoOrds`.
   - Calculated contracts: `contracts = math.floor(size_usdt / (entry_price * CT_VAL / LOT_SIZE)) * LOT_SIZE`.
   - Ensured `contracts ≥ MIN_SIZE` and aligned with `lotSz`.
   - Example: For `size_usdt=15`, `entry_price=1.2328`, `CT_VAL=1`, `LOT_SIZE=1`, `contracts = math.floor(15 / 1.2328) = 12`.

3. **Decimal Fix**:
   - Used `math.floor` for `contracts`, `tp1_contracts`, `tp2_contracts`, and `sl_contracts` to ensure integer or `lotSz`-aligned sizes.
   - Adjusted TP2 size to ensure `tp1_contracts + tp2_contracts = contracts`, avoiding mismatches.

4. **Enhanced Error Logging**:
   - Added `sMsg` extraction in `place_order` to log detailed error messages (e.g., “Invalid quantity”).
   - Logged actual USDT value (`contracts * entry_price * CT_VAL`) for transparency.

5. **Balance Check**:
   - Retained margin check: `margin = POSITION_SIZE_USDT / LEVERAGE` (e.g., $15 / 15 = $1).
   - Ensured balance supports two positions for hedging ($2 total margin).

### Key Code Changes
- **New Parameters**:
  - `CT_VAL`, `LOT_SIZE`, `MIN_SIZE` for contract sizing.
- **get_instrument_details**:
  - Fetches `ctVal`, `lotSz`, `minSz` at startup.
  - Logs: `Instrument details for KAITO-USDT-SWAP: ctVal=1, lotSz=1, minSz=1`.
- **place_order**:
  - Converts USDT to contracts: `contracts = math.floor(size_usdt / (entry_price * CT_VAL / LOT_SIZE)) * LOT_SIZE`.
  - Removed `tgtCcy` from payload.
  - TP1/TP2 sizes: `tp1_contracts = math.floor(TP1_SIZE_RATIO * contracts / LOT_SIZE) * LOT_SIZE`.
  - Logs actual USDT: `actual_usdt = contracts * entry_price * CT_VAL`.
- **Error Handling**:
  - Logs `sMsg` for detailed API errors.

### Lessons Learned
- **Instrument-Specific Rules**: Not all OKX swap pairs support `tgtCcy="quote_ccy"`. Always check `/public/instruments` for `tgtCcy` support.
- **Decimal Precision**: Use `math.floor` and `lotSz` to ensure valid sizes, especially for contracts.
- **Error Logging**: Detailed logging (`sMsg`) is critical for debugging API errors like `code: 1`.
- **Balance Management**: A tight balance ($3.45 vs. $2.00 required) requires careful sizing to avoid failures.
- **Flexibility**: Contract-based sizing with `ctVal` makes the bot adaptable to any pair, even without `tgtCcy="quote_ccy"`.

### Testing and Validation
- **Dry Run**: Recommended setting `x-simulated-trading: '1'` to test in OKX’s demo mode.
- **Log Checks**:
  - Verify `ctVal`, `lotSz`, `minSz` in `trading_bot.log`.
  - Confirm `contracts` and `actual_usdt` in order logs.
  - Check for `sMsg` if errors persist.
- **Balance**: Ensured $3.45 covers $2.00 margin for hedging.
- **Symbol Test**: Suggested testing with `DOGE-USDT-SWAP` to validate `ctVal` handling.

### Known Limitations
- **Rounding**: Flooring contracts may result in slightly smaller positions (e.g., 14.79 USDT vs. 15).
- **Balance Tightness**: $3.45 is close to $2.00 margin, risking failures if rounded sizes increase margin.
- **API Dependency**: Relies on `/public/instruments` for `ctVal`. Defaults may cause sizing errors if the API fails.

### Next Steps
- **Version v3.1.8**: Enhance error handling or add a new feature (e.g., dynamic TP/SL based on volatility).
- **Balance**: Consider lowering `POSITION_SIZE_USDT` to `10` or funding the account.
- **Testing**: Validate with other pairs to ensure `ctVal` and `lotSz` handling.

## Conclusion
**Pytomade_Bot-v3.1.7** successfully implemented USDT-based sizing by converting to contracts, resolving the `tgtCcy` error and decimal issues for `KAITO-USDT-SWAP`. The bot is now robust for pairs not supporting `tgtCcy="quote_ccy"`, with precise sizing and detailed logging. This version is ready for testing, and upon confirmation, we can proceed to v3.1.8 for further improvements.