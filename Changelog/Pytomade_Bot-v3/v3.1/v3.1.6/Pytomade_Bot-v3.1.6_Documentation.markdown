# Pytomade_Bot-v3.1.6: Key Changes

### 1. Multiple Take Profit Levels (TP1 & TP2)
- **What**: Added TP1 (7% PnL, 71% of position) and TP2 (15% PnL, 29% of position).
- **Why**: Enables partial profit-taking for better risk management and flexibility.

### 2. Updated `place_order` Function
- **What**: Modified to calculate TP1/TP2 prices and split position sizes; sets SL for the full position.
- **Why**: Supports the new TP strategy while maintaining stop loss protection.

### 3. New Global Variables
- **What**: Added `TP1_PNL = 0.07`, `TP2_PNL = 0.15`, `TP1_SIZE_RATIO = 0.71`.
- **Why**: Controls TP levels and size allocation, adjustable for strategy tuning.

### 4. Kept Core Features
- **What**: Preserved hedging and `calculate_tp_sl` function.
- **Why**: Ensures compatibility with existing functionality.

### 5. Precision & Error Handling
- **What**: Rounded prices to 5 decimals, sizes to 2 decimals per OKX requirements.
- **Why**: Prevents errors and ensures API compatibility.

### 6. Testing Note
- **What**: Recommend testing TP1/TP2 in OKX demo mode.
- **Why**: Verify functionality before live use.

---

### Quick Recap
Version 3.1.6 adds dual take profit levels, refines the `place_order` function, and introduces new variables, all while keeping the bot’s core intact. Test in demo mode first!