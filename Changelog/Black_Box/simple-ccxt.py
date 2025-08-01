import ccxt.async_support as ccxt
import asyncio
import pandas as pd
import ta

# Konfigurasi bot
exchange = ccxt.okx({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET',
    'password': 'YOUR_PASSWORD',
    'enableRateLimit': True,
})
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LEVERAGE = 15
MARGIN_PER_POSITION = 5  # $5 per posisi
TP_PNL = 0.05  # Take Profit 5% PNL
SL_PNL = 0.03  # Stop Loss 3% PNL

# Variabel global untuk melacak posisi
current_positions = {'long': None, 'short': None}

async def fetchOHLCV(symbol, timeframe):
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

async def calculate_indicators(df):
    df['ma'] = ta.trend.sma_indicator(df['close'], window=20)
    df['atr'] = ta.volatility.atr(df['high'], df['low'], df['close'], window=14)
    return df

def calculate_tp_sl(entry_price, direction):
    if direction == 'long':
        tp_price = entry_price * (1 + TP_PNL)
        sl_price = entry_price * (1 - SL_PNL)
    else:  # short
        tp_price = entry_price * (1 - TP_PNL)
        sl_price = entry_price * (1 + SL_PNL)
    return tp_price, sl_price

async def check_positions(symbol):
    positions = await exchange.fetch_positions(symbol)
    global current_positions
    current_positions = {'long': None, 'short': None}
    for pos in positions:
        if pos['contracts'] > 0:  # Ada posisi terbuka
            direction = 'long' if pos['side'] == 'buy' else 'short'
            current_positions[direction] = pos

async def place_order(symbol, direction, entry_price):
    await check_positions(symbol)
    if current_positions[direction]:  # Sudah ada posisi di arah yang sama
        print(f"Posisi {direction} sudah ada, tidak membuka posisi baru.")
        return
    
    amount = (MARGIN_PER_POSITION * LEVERAGE) / entry_price
    tp_price, sl_price = calculate_tp_sl(entry_price, direction)
    
    order_type = 'buy' if direction == 'long' else 'sell'
    order = await exchange.create_order(
        symbol=symbol,
        type='market',
        side=order_type,
        amount=amount,
        params={'leverage': LEVERAGE}
    )
    print(f"Opened {direction} position: {order['id']}")
    
    # Set TP dan SL
    tp_order = await exchange.create_order(
        symbol=symbol,
        type='limit',
        side='sell' if direction == 'long' else 'buy',
        amount=amount,
        price=tp_price,
        params={'reduceOnly': True}
    )
    sl_order = await exchange.create_order(
        symbol=symbol,
        type='stop',
        side='sell' if direction == 'long' else 'buy',
        amount=amount,
        price=sl_price,
        params={'reduceOnly': True}
    )
    current_positions[direction] = order

async def trading_logic():
    await exchange.load_markets()
    while True:
        df = await fetchOHLCV(SYMBOL, TIMEFRAME)
        df = await calculate_indicators(df)
        last_row = df.iloc[-1]
        current_price = last_row['close']
        
        # Contoh logika sederhana berdasarkan MA
        if last_row['close'] > last_row['ma'] and not current_positions['long']:
            await place_order(SYMBOL, 'long', current_price)
        elif last_row['close'] < last_row['ma'] and not current_positions['short']:
            await place_order(SYMBOL, 'short', current_price)
        
        await asyncio.sleep(60)  # Tunggu 1 menit sebelum iterasi berikutnya

async def main():
    await trading_logic()

if __name__ == "__main__":
    asyncio.run(main())
