# Pytomade_Bot-v3.1.6: Key Changes and Lessons Learned

### 1. Dual Take Profit Levels
- **What**: Added TP1 (5% PnL, 51% of position) and TP2 (7% PnL, 49%) for KAITO-USDT-SWAP.
- **Why**: Enables partial profit-taking for improved risk management.

### 2. Updated `place_order` Function
- **What**: Calculates TP1/TP2/SL prices and splits sizes; uses isolated margin for hedging.
- **Why**: Supports dual TPs and allows long/short positions to coexist.

### 3. New Global Variables
- **What**: Added `TP1_PNL=0.05`, `TP2_PNL=0.07`, `TP1_SIZE_RATIO=0.51`.
- **Why**: Defines TP levels and size allocation, adjustable for strategy tuning.

### 4. Instrument Change
- **What**: Switched to `KAITO-USDT-SWAP`, `CONTRACT_SIZE=1`, `LEVERAGE=21`, `MARGIN_COST_USD=1.5`.
- **Why**: Adapted bot for a new trading pair with updated parameters.

### 5. Lessons Learned
- **Sizing Issues**: Contract sizes caused API rejections (e.g., 0.136 vs. 0.14 for DOGE). KAITO uses integer sizes (`round(contracts, 0)`), but TP1/TP2/SL precision is inconsistent (0 vs. 5 decimals).
- **USDT-Based Sizing**: OKX supports `tgtCcy="quote_ccy"` for USDT-based sizing (e.g., $31.50), avoiding contract precision issues. Plan for v3.1.7.
- **Precision Rules**: DOGE requires 2 decimals (min 0.1 contracts), KAITO/PENGU likely 0 decimals. Must align `sz` and `attachAlgoOrds` sizes.
- **Debugging**: Need detailed API error logging to diagnose "All operations failed."
- **Testing**: Use demo mode (`x-simulated-trading: '1'`) to validate sizing.

### 6. Testing Note
- **What**: Test in OKX demo mode to verify order placement for KAITO-USDT-SWAP.
- **Why**: Ensures correct sizing and API compatibility before live trading.

---

### Quick Recap
Version 3.1.6 adds dual TPs for KAITO-USDT-SWAP, but contract sizing precision caused API errors. USDT-based sizing (`tgtCcy="quote_ccy"`) is a promising solution for v3.1.7. Test in demo mode and log API responses.