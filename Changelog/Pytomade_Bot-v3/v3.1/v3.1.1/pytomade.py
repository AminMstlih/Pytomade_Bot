
import time
import hmac
import base64
import requests
import pandas as pd
import numpy as np
import json
import logging
import sys
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('advanced_trading.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    timestamp: float
    symbol: str
    side: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    indicators: Dict

@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    margin_used: float
    timestamp: float

@dataclass
class Trade:
    id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: float
    exit_time: Optional[float]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    commission: float
    status: str  # 'open', 'closed', 'cancelled'

class OKXClient:
    def __init__(self, api_key: str, secret_key: str, passphrase: str, demo: bool = False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = "https://www.okx.com"
        self.demo = demo
        self.session = requests.Session()
        self.rate_limiter = deque(maxlen=100)
        
    def _rate_limit(self):
        """Simple rate limiting"""
        now = time.time()
        self.rate_limiter.append(now)
        if len(self.rate_limiter) >= 100:
            time_diff = now - self.rate_limiter[0]
            if time_diff < 2:  # 100 requests per 2 seconds
                time.sleep(2 - time_diff)
    
    def _get_server_time(self):
        try:
            response = self.session.get(f"{self.base_url}/api/v5/public/time")
            response.raise_for_status()
            return str(float(response.json()["data"][0]["ts"]) / 1000.0)
        except Exception as e:
            logger.error(f"Error fetching server time: {e}")
            return str(int(time.time()))
    
    def _generate_headers(self, method: str, request_path: str, body: str = ''):
        timestamp = self._get_server_time()
        message = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode(), 
                message.encode(), 
                digestmod='sha256'
            ).digest()
        ).decode()
        
        return {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json',
            'x-simulated-trading': "1" if self.demo else "0"
        }
    
    def _request(self, method: str, endpoint: str, params: dict = None, private: bool = False):
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = {}
        data = ''
        
        if private:
            if method == 'GET' and params:
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                endpoint += f"?{query_string}"
            elif method in ['POST', 'PUT', 'DELETE'] and params:
                data = json.dumps(params)
            
            headers = self._generate_headers(method, endpoint, data)
        
        try:
            if method == 'GET':
                response = self.session.get(url, headers=headers, params=params if not private else None)
            else:
                response = self.session.request(method, url, headers=headers, data=data)
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('code') != '0':
                logger.error(f"API Error: {result.get('msg', 'Unknown error')}")
                return None
            
            return result.get('data', [])
            
        except Exception as e:
            logger.error(f"Request failed: {method} {endpoint} - {e}")
            return None

class MarketDataManager:
    def __init__(self, client: OKXClient):
        self.client = client
        self.cache = {}
        self.cache_expiry = {}
        
    def get_candles(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV data with caching"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        now = time.time()
        
        # Check cache
        if cache_key in self.cache and now < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        endpoint = f"/api/v5/market/candles"
        params = {
            'instId': symbol,
            'bar': timeframe,
            'limit': str(limit)
        }
        
        data = self.client._request('GET', endpoint, params)
        if not data:
            return None
        
        # Reverse to get chronological order
        data.reverse()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'volCcy', 'volCcyQuote', 'confirm'
        ])
        
        # Convert to appropriate types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        
        # Cache for 30 seconds
        self.cache[cache_key] = df
        self.cache_expiry[cache_key] = now + 30
        
        return df
    
    def get_order_book(self, symbol: str, depth: int = 20) -> Optional[Dict]:
        """Get order book data"""
        endpoint = f"/api/v5/market/books"
        params = {'instId': symbol, 'sz': str(depth)}
        
        data = self.client._request('GET', endpoint, params)
        if not data:
            return None
        
        book = data[0]
        return {
            'bids': [[float(p), float(s)] for p, s, _, _ in book['bids']],
            'asks': [[float(p), float(s)] for p, s, _, _ in book['asks']],
            'timestamp': int(book['ts'])
        }
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current ticker data"""
        endpoint = f"/api/v5/market/ticker"
        params = {'instId': symbol}
        
        data = self.client._request('GET', endpoint, params)
        if not data:
            return None
        
        ticker = data[0]
        return {
            'symbol': ticker['instId'],
            'last': float(ticker['last']),
            'bid': float(ticker['bidPx']),
            'ask': float(ticker['askPx']),
            'volume24h': float(ticker['vol24h']),
            'change24h': float(ticker['chgUtc'])
        }

class TechnicalAnalysis:
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

class SignalGenerator:
    def __init__(self, market_data: MarketDataManager):
        self.market_data = market_data
        self.ta = TechnicalAnalysis()
    
    def multi_timeframe_analysis(self, symbol: str) -> Dict[str, TradeSignal]:
        """Analyze multiple timeframes for confluence"""
        timeframes = ['1m', '5m', '15m', '1H']
        signals = {}
        
        for tf in timeframes:
            df = self.market_data.get_candles(symbol, tf, 100)
            if df is None or len(df) < 50:
                continue
            
            signal = self._generate_signal(df, tf, symbol)
            if signal:
                signals[tf] = signal
        
        return signals
    
    def _generate_signal(self, df: pd.DataFrame, timeframe: str, symbol: str) -> Optional[TradeSignal]:
        """Generate trading signal based on multiple indicators"""
        if len(df) < 50:
            return None
        
        # Calculate indicators
        df['sma_20'] = self.ta.sma(df['close'], 20)
        df['sma_50'] = self.ta.sma(df['close'], 50)
        df['ema_12'] = self.ta.ema(df['close'], 12)
        df['ema_26'] = self.ta.ema(df['close'], 26)
        df['rsi'] = self.ta.rsi(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.ta.bollinger_bands(df['close'])
        df['stoch_k'], df['stoch_d'] = self.ta.stochastic(df['high'], df['low'], df['close'])
        df['atr'] = self.ta.atr(df['high'], df['low'], df['close'])
        df['vwap'] = self.ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = self.ta.ema(df['macd'], 9)
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Signal conditions
        bullish_signals = 0
        bearish_signals = 0
        
        # Trend following
        if latest['sma_20'] > latest['sma_50']:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # MACD
        if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            bullish_signals += 2
        elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            bearish_signals += 2
        
        # RSI
        if 30 < latest['rsi'] < 70:
            if latest['rsi'] > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Bollinger Bands
        if latest['close'] < latest['bb_lower']:
            bullish_signals += 1
        elif latest['close'] > latest['bb_upper']:
            bearish_signals += 1
        
        # Stochastic
        if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] < 80:
            bullish_signals += 1
        elif latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] > 20:
            bearish_signals += 1
        
        # Price vs VWAP
        if latest['close'] > latest['vwap']:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Determine signal
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            return None
        
        if bullish_signals > bearish_signals and bullish_signals >= 4:
            side = 'long'
            confidence = bullish_signals / total_signals
        elif bearish_signals > bullish_signals and bearish_signals >= 4:
            side = 'short'
            confidence = bearish_signals / total_signals
        else:
            return None
        
        # Calculate stop loss and take profit based on ATR
        atr_value = latest['atr']
        entry_price = latest['close']
        
        if side == 'long':
            stop_loss = entry_price - (2 * atr_value)
            take_profit = entry_price + (3 * atr_value)
        else:
            stop_loss = entry_price + (2 * atr_value)
            take_profit = entry_price - (3 * atr_value)
        
        return TradeSignal(
            timestamp=time.time(),
            symbol=symbol,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe=timeframe,
            indicators={
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']),
                'atr': atr_value,
                'volume_ratio': latest['volume'] / df['volume'].rolling(20).mean().iloc[-1]
            }
        )

class RiskManager:
    def __init__(self, max_portfolio_risk: float = 0.02, max_position_risk: float = 0.01):
        self.max_portfolio_risk = max_portfolio_risk  # 2% of portfolio
        self.max_position_risk = max_position_risk    # 1% per position
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_positions = 5
        self.correlation_limit = 0.7
        
    def calculate_position_size(self, signal: TradeSignal, account_balance: float, 
                               current_positions: List[Position]) -> float:
        """Calculate optimal position size based on Kelly Criterion and risk limits"""
        
        # Risk per trade based on stop loss
        risk_per_unit = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        
        # Kelly Criterion (simplified)
        win_rate = min(signal.confidence, 0.7)  # Cap at 70%
        avg_win = abs(signal.take_profit - signal.entry_price) / signal.entry_price
        avg_loss = risk_per_unit
        
        if avg_loss == 0:
            kelly_fraction = 0
        else:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Conservative Kelly (use 25% of full Kelly)
        kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.1))
        
        # Position size limits
        max_risk_amount = account_balance * self.max_position_risk
        position_value = max_risk_amount / risk_per_unit
        
        # Apply Kelly sizing
        kelly_position_value = account_balance * kelly_fraction
        
        # Use the smaller of the two
        final_position_value = min(position_value, kelly_position_value)
        
        # Check portfolio limits
        current_exposure = sum(pos.size * pos.current_price for pos in current_positions)
        max_total_exposure = account_balance * 3  # 3x leverage limit
        
        if current_exposure + final_position_value > max_total_exposure:
            final_position_value = max(0, max_total_exposure - current_exposure)
        
        return final_position_value
    
    def should_close_position(self, position: Position, current_price: float) -> bool:
        """Determine if position should be closed based on risk rules"""
        
        # Update unrealized PnL
        if position.side == 'long':
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        # Close if daily loss limit exceeded
        if pnl_pct < -self.daily_loss_limit:
            return True
        
        # Trailing stop loss (simple implementation)
        if position.side == 'long' and pnl_pct > 0.05:  # If 5% profit
            trailing_stop = current_price * 0.97  # 3% trailing stop
            if current_price < trailing_stop:
                return True
        elif position.side == 'short' and pnl_pct > 0.05:
            trailing_stop = current_price * 1.03
            if current_price > trailing_stop:
                return True
        
        return False

class PerformanceAnalyzer:
    def __init__(self):
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[float, float]] = []  # (timestamp, equity)
        
    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        
    def add_equity_point(self, timestamp: float, equity: float):
        self.equity_curve.append((timestamp, equity))
        
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        closed_trades = [t for t in self.trades if t.status == 'closed' and t.pnl is not None]
        if not closed_trades:
            return {}
        
        pnls = [t.pnl for t in closed_trades]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        metrics = {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'total_pnl': sum(pnls),
            'avg_win': statistics.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': statistics.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'profit_factor': abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        }
        
        # Calculate Sharpe ratio from equity curve
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1][1]
                curr_equity = self.equity_curve[i][1]
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            if returns:
                avg_return = statistics.mean(returns)
                if len(returns) > 1:
                    std_return = statistics.stdev(returns)
                    metrics['sharpe_ratio'] = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
                else:
                    metrics['sharpe_ratio'] = 0
        
        return metrics

class DatabaseManager:
    def __init__(self, db_path: str = 'trading_data.db'):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                size REAL,
                entry_price REAL,
                exit_price REAL,
                entry_time REAL,
                exit_time REAL,
                pnl REAL,
                pnl_pct REAL,
                commission REAL,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                symbol TEXT,
                side TEXT,
                confidence REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                timeframe TEXT,
                indicators TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                equity REAL,
                total_pnl REAL,
                open_positions INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade: Trade):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO trades 
            (id, symbol, side, size, entry_price, exit_price, entry_time, exit_time, 
             pnl, pnl_pct, commission, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.id, trade.symbol, trade.side, trade.size, trade.entry_price,
            trade.exit_price, trade.entry_time, trade.exit_time, trade.pnl,
            trade.pnl_pct, trade.commission, trade.status
        ))
        
        conn.commit()
        conn.close()
    
    def save_signal(self, signal: TradeSignal):
        """Save signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals 
            (timestamp, symbol, side, confidence, entry_price, stop_loss, take_profit, 
             timeframe, indicators)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp, signal.symbol, signal.side, signal.confidence,
            signal.entry_price, signal.stop_loss, signal.take_profit,
            signal.timeframe, json.dumps(signal.indicators)
        ))
        
        conn.commit()
        conn.close()

class AdvancedTradingBot:
    def __init__(self, config: Dict):
        # Initialize components
        self.client = OKXClient(
            config['api_key'],
            config['secret_key'], 
            config['passphrase'],
            config.get('demo', True)
        )
        
        self.market_data = MarketDataManager(self.client)
        self.signal_generator = SignalGenerator(self.market_data)
        self.risk_manager = RiskManager(
            config.get('max_portfolio_risk', 0.02),
            config.get('max_position_risk', 0.01)
        )
        self.performance_analyzer = PerformanceAnalyzer()
        self.db_manager = DatabaseManager()
        
        # Configuration
        self.config = config
        self.symbols = config.get('symbols', ['DOGE-USDT-SWAP'])
        self.is_running = False
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Dict] = {}
        
        # Performance tracking
        self.start_equity = 0
        self.current_equity = 0
        self.last_performance_update = 0
        
    def start(self):
        """Start the trading bot"""
        logger.info("Starting Advanced Trading Bot")
        self.is_running = True
        
        # Initialize starting equity
        account_info = self.get_account_info()
        if account_info:
            self.start_equity = account_info.get('total_equity', 0)
            self.current_equity = self.start_equity
            logger.info(f"Starting equity: ${self.start_equity:.2f}")
        
        # Start main trading loop
        with ThreadPoolExecutor(max_workers=4) as executor:
            try:
                while self.is_running:
                    # Submit tasks for parallel execution
                    futures = []
                    
                    # Update positions
                    futures.append(executor.submit(self.update_positions))
                    
                    # Process signals for each symbol
                    for symbol in self.symbols:
                        futures.append(executor.submit(self.process_symbol, symbol))
                    
                    # Update performance metrics
                    futures.append(executor.submit(self.update_performance))
                    
                    # Wait for all tasks to complete
                    for future in futures:
                        try:
                            future.result(timeout=30)
                        except Exception as e:
                            logger.error(f"Task failed: {e}")
                    
                    # Sleep before next iteration
                    time.sleep(self.config.get('polling_interval', 10))
                    
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
            finally:
                self.shutdown()
    
    def process_symbol(self, symbol: str):
        """Process trading logic for a single symbol"""
        try:
            # Generate signals
            signals = self.signal_generator.multi_timeframe_analysis(symbol)
            
            # Find the best signal (highest confidence)
            best_signal = None
            best_confidence = 0
            
            for timeframe, signal in signals.items():
                if signal.confidence > best_confidence:
                    best_signal = signal
                    best_confidence = signal.confidence
            
            if not best_signal:
                return
            
            # Save signal to database
            self.db_manager.save_signal(best_signal)
            
            # Check if we should trade
            current_position = self.positions.get(symbol)
            
            if current_position is None:
                # No position, consider opening one
                if best_confidence > self.config.get('min_signal_confidence', 0.6):
                    self.open_position(best_signal)
            else:
                # Have position, check if we should close or adjust
                ticker = self.market_data.get_ticker(symbol)
                if ticker:
                    current_price = ticker['last']
                    
                    # Check risk management rules
                    if self.risk_manager.should_close_position(current_position, current_price):
                        self.close_position(symbol, "Risk management")
                    
                    # Check for signal reversal
                    elif (current_position.side != best_signal.side and 
                          best_confidence > self.config.get('reversal_confidence', 0.7)):
                        self.close_position(symbol, "Signal reversal")
                        time.sleep(1)  # Brief pause before opening new position
                        self.open_position(best_signal)
                        
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    def open_position(self, signal: TradeSignal):
        """Open a new position based on signal"""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return
            
            # Calculate position size
            position_value = self.risk_manager.calculate_position_size(
                signal, 
                account_info['available_balance'],
                list(self.positions.values())
            )
            
            if position_value < self.config.get('min_position_value', 10):
                logger.info(f"Position value too small: ${position_value:.2f}")
                return
            
            # Place order via OKX API
            order_result = self.place_market_order(
                symbol=signal.symbol,
                side=signal.side,
                size=position_value / signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order_result:
                logger.info(f"Opened {signal.side} position for {signal.symbol}: ${position_value:.2f}")
                
                # Create trade record
                trade = Trade(
                    id=order_result['order_id'],
                    symbol=signal.symbol,
                    side=signal.side,
                    size=order_result['size'],
                    entry_price=signal.entry_price,
                    exit_price=None,
                    entry_time=time.time(),
                    exit_time=None,
                    pnl=None,
                    pnl_pct=None,
                    commission=order_result.get('fee', 0),
                    status='open'
                )
                
                self.db_manager.save_trade(trade)
                self.performance_analyzer.add_trade(trade)
                
        except Exception as e:
            logger.error(f"Error opening position: {e}")
    
    def close_position(self, symbol: str, reason: str):
        """Close an existing position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            # Close position via OKX API
            result = self.close_market_position(symbol)
            
            if result:
                logger.info(f"Closed position for {symbol}. Reason: {reason}")
                
                # Update trade record
                for trade in self.performance_analyzer.trades:
                    if trade.symbol == symbol and trade.status == 'open':
                        trade.exit_price = result['exit_price']
                        trade.exit_time = time.time()
                        trade.pnl = result['pnl']
                        trade.pnl_pct = result['pnl_pct']
                        trade.status = 'closed'
                        
                        self.db_manager.save_trade(trade)
                        break
                
                # Remove from positions
                del self.positions[symbol]
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        data = self.client._request('GET', '/api/v5/account/balance', private=True)
        if not data:
            return None
        
        account = data[0]
        total_equity = float(account.get('totalEq', 0))
        available_balance = float(account.get('availBal', 0))
        
        return {
            'total_equity': total_equity,
            'available_balance': available_balance,
            'margin_used': total_equity - available_balance
        }
    
    def update_positions(self):
        """Update current positions"""
        try:
            data = self.client._request('GET', '/api/v5/account/positions', private=True)
            if not data:
                return
            
            current_positions = {}
            
            for pos_data in data:
                if float(pos_data['pos']) != 0:
                    symbol = pos_data['instId']
                    
                    position = Position(
                        symbol=symbol,
                        side=pos_data['posSide'],
                        size=float(pos_data['pos']),
                        entry_price=float(pos_data['avgPx']),
                        current_price=float(pos_data['markPx']),
                        unrealized_pnl=float(pos_data['upl']),
                        unrealized_pnl_pct=float(pos_data['uplRatio']),
                        margin_used=float(pos_data['margin']),
                        timestamp=time.time()
                    )
                    
                    current_positions[symbol] = position
            
            self.positions = current_positions
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def update_performance(self):
        """Update performance metrics"""
        try:
            now = time.time()
            if now - self.last_performance_update < 60:  # Update every minute
                return
            
            account_info = self.get_account_info()
            if account_info:
                self.current_equity = account_info['total_equity']
                
                # Add equity point
                self.performance_analyzer.add_equity_point(now, self.current_equity)
                
                # Calculate and log metrics
                metrics = self.performance_analyzer.calculate_metrics()
                if metrics:
                    logger.info(f"Performance - Total PnL: ${metrics.get('total_pnl', 0):.2f}, "
                              f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%, "
                              f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                
                self.last_performance_update = now
                
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def place_market_order(self, symbol: str, side: str, size: float, 
                          stop_loss: float, take_profit: float) -> Optional[Dict]:
        """Place a market order with stop loss and take profit"""
        # This is a simplified implementation
        # In practice, you'd use the actual OKX order placement API
        
        params = {
            'instId': symbol,
            'tdMode': 'cross',
            'side': 'buy' if side == 'long' else 'sell',
            'posSide': side,
            'ordType': 'market',
            'sz': str(round(size, 6)),
            'attachAlgoOrds': [
                {
                    'algoOrdType': 'oco',
                    'tpTriggerPx': str(round(take_profit, 6)),
                    'tpOrdPx': '-1',
                    'slTriggerPx': str(round(stop_loss, 6)),
                    'slOrdPx': '-1',
                    'tpTriggerPxType': 'last',
                    'slTriggerPxType': 'last'
                }
            ]
        }
        
        result = self.client._request('POST', '/api/v5/trade/order', params, private=True)
        if result:
            return {
                'order_id': result[0]['ordId'],
                'size': size,
                'fee': 0  # Would be calculated from actual response
            }
        return None
    
    def close_market_position(self, symbol: str) -> Optional[Dict]:
        """Close market position"""
        position = self.positions.get(symbol)
        if not position:
            return None
        
        params = {
            'instId': symbol,
            'mgnMode': 'cross',
            'posSide': position.side
        }
        
        result = self.client._request('POST', '/api/v5/trade/close-position', params, private=True)
        if result:
            return {
                'exit_price': position.current_price,
                'pnl': position.unrealized_pnl,
                'pnl_pct': position.unrealized_pnl_pct
            }
        return None
    
    def shutdown(self):
        """Shutdown the bot gracefully"""
        logger.info("Shutting down trading bot")
        self.is_running = False
        
        # Close all positions if configured to do so
        if self.config.get('close_positions_on_shutdown', True):
            for symbol in list(self.positions.keys()):
                self.close_position(symbol, "Bot shutdown")
        
        # Final performance report
        metrics = self.performance_analyzer.calculate_metrics()
        if metrics:
            logger.info("Final Performance Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")

def main():
    """Main function to run the advanced trading bot"""
    config = {
        'api_key': "",
        'secret_key': "",
        'passphrase': "",
        'demo': False,  # Set to True for demo trading
        'symbols': ['DOGE-USDT-SWAP'],
        'max_portfolio_risk': 0.02,
        'max_position_risk': 0.01,
        'min_signal_confidence': 0.6,
        'reversal_confidence': 0.7,
        'min_position_value': 10,
        'polling_interval': 10,
        'close_positions_on_shutdown': True
    }
    
    bot = AdvancedTradingBot(config)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        bot.shutdown()

if __name__ == "__main__":
    main()
