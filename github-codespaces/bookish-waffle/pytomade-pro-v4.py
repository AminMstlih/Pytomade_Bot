import time
import hmac
import hashlib
import base64
import requests
import datetime
import json
import os
from dotenv import load_dotenv

# Load API credentials from .env file
load_dotenv()
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

BASE_URL = "https://www.okx.com"
SYMBOL = "BTC-USDT-SWAP"
LEVERAGE = 15
ORDER_SIZE = 0.01
SHORT_MA = 13
LONG_MA = 21

def get_server_time():
    """Get server time in ISO 8601 format"""
    endpoint = "/api/v5/public/time"
    try:
        response = requests.get(BASE_URL + endpoint, timeout=5)
        response.raise_for_status()
        # Return timestamp in milliseconds as string (OKX API format)
        return response.json()["data"][0]["ts"]
    except Exception as e:
        print(f"Error getting server time: {e}")
        # Fallback to local time in ISO 8601 format
        return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')

def generate_signature(timestamp, method, request_path, body=""):
    """Generate signature according to OKX docs"""
    try:
        if not isinstance(SECRET_KEY, str):
            raise ValueError("SECRET_KEY must be a string")
            
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            SECRET_KEY.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')
    except Exception as e:
        print(f"Signature generation failed: {e}")
        raise

def send_request(method, endpoint, body=None):
    """Send authenticated request to OKX API"""
    try:
        timestamp = get_server_time()
        body_json = json.dumps(body) if body else ""
        
        headers = {
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": generate_signature(timestamp, method, endpoint, body_json),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/json",
            "x-simulated-trading": "1"  # 1 for testnet, 0 for live
        }
        
        url = BASE_URL + endpoint
        response = requests.request(
            method,
            url,
            headers=headers,
            data=body_json if body else None,
            timeout=10
        )
        
        # Debugging response
        print(f"Request to {endpoint} - Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response content: {response.text}")
            
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Error response: {e.response.text}")
        return {"error": str(e)}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": str(e)}

def set_leverage():
    """Set leverage for the trading pair"""
    endpoint = "/api/v5/account/set-leverage"
    body = {
        "instId": SYMBOL,
        "lever": str(LEVERAGE),
        "mgnMode": "cross"
    }
    response = send_request("POST", endpoint, body)
    print("Leverage Response:", response)
    return response

def get_prices():
    """Get historical prices with improved error handling"""
    endpoint = f"/api/v5/market/candles?instId={SYMBOL}&bar=1m&limit={LONG_MA}"
    response = send_request("GET", endpoint)
    
    if "data" not in response:
        print(f"Error getting prices: {response}")
        return []
        
    try:
        return [float(candle[4]) for candle in response["data"]]
    except (IndexError, ValueError) as e:
        print(f"Error parsing prices: {e}")
        return []

def moving_average(prices, period):
    """Calculate moving average with safety checks"""
    if not prices or len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def place_order(side, pos_side):
    """Place order with improved error handling"""
    endpoint = "/api/v5/trade/order"
    order_data = {
        "instId": SYMBOL,
        "tdMode": "cross",
        "side": side,
        "posSide": pos_side,
        "ordType": "market",
        "sz": str(ORDER_SIZE * LEVERAGE)
    }
    response = send_request("POST", endpoint, body=order_data)
    print("Order Response:", response)
    
    if "data" in response and response["data"][0]["sCode"] == "0":
        return True
    return False

def check_open_positions():
    """Check positions with detailed response parsing"""
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    
    if "data" not in response:
        print(f"Error checking positions: {response}")
        return False
        
    for pos in response["data"]:
        if pos["instId"] == SYMBOL and float(pos["pos"]) > 0:
            return True
    return False

def close_all_positions():
    """Close all positions with confirmation"""
    if not check_open_positions():
        print("No open positions to close.")
        return True
        
    print("Closing all positions...")
    
    # Get current positions first
    endpoint = "/api/v5/account/positions"
    response = send_request("GET", endpoint)
    
    if "data" not in response:
        print("Failed to get positions for closing")
        return False
        
    success = True
    for pos in response["data"]:
        if pos["instId"] == SYMBOL and float(pos["pos"]) > 0:
            side = "buy" if pos["posSide"] == "short" else "sell"
            if not place_order(side, pos["posSide"]):
                success = False
                
    return success

def ma_cross_strategy():
    """Enhanced MA Cross strategy with better logging"""
    set_leverage()
    position = None
    
    while True:
        try:
            prices = get_prices()
            if not prices:
                print("Failed to get prices, retrying...")
                time.sleep(60)
                continue

            short_ma = moving_average(prices, SHORT_MA)
            long_ma = moving_average(prices, LONG_MA)
            
            if short_ma is None or long_ma is None:
                time.sleep(60)
                continue
                
            print(f"{datetime.datetime.now()} - Short MA: {short_ma:.2f}, Long MA: {long_ma:.2f}")

            if short_ma > long_ma and (not position or position == "long"):
                print("SHORT signal detected")
                if close_all_positions():
                    if place_order("sell", "short"):
                        position = "short"
                        
            elif short_ma < long_ma and (not position or position == "short"):
                print("LONG signal detected")
                if close_all_positions():
                    if place_order("buy", "long"):
                        position = "long"
            
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\nStopping bot...")
            close_all_positions()
            break
        except Exception as e:
            print(f"Error in strategy: {e}")
            time.sleep(60)

if __name__ == "__main__":
    print("Starting OKX Trading Bot")
    try:
        # Test API connection first
        test_response = send_request("GET", "/api/v5/account/balance")
        if "data" in test_response:
            print("API connection successful")
            ma_cross_strategy()
        else:
            print("API connection failed. Check credentials and network.")
    except Exception as e:
        print(f"Failed to start bot: {e}")
