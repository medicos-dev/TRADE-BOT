import os
import sys
import subprocess

try:
    import psutil
    os.nice(10) if os.name != 'nt' else psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
except ImportError:
    pass

# =========================================================
# 1. AUTO-BOOTSTRAP & ENV
# =========================================================
if sys.prefix == sys.base_prefix:
    print("[BOOTSTRAP] System Python detected. Creating Virtual Environment...")
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    if not os.path.exists(venv_dir):
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    python_exe = os.path.join(venv_dir, "Scripts", "python.exe") if os.name == 'nt' else os.path.join(venv_dir, "bin", "python")
    subprocess.check_call([python_exe, "-m", "pip", "install", "pandas", "scikit-learn", "python-dotenv", "aiohttp", "aiosqlite", "lightgbm", "numba", "psutil"], stdout=subprocess.DEVNULL)
    sys.exit(subprocess.call([python_exe] + sys.argv))

import json
import time
import logging
import csv
import decimal
import math
import hmac
import hashlib
import urllib.parse
import requests
import numpy as np
import pandas as pd
import aiohttp
import asyncio
import aiosqlite
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator

from numba import prange

@njit(parallel=True, fastmath=True)
def fast_features_numba(closes, highs, lows, volumes, rsi_length):
    n = len(closes)
    out = np.zeros((n, 24), dtype=np.float64)
    kf_x = closes[0]; kf_p = 1.0; kf_q = 1e-5; kf_r = 0.01
    ema_12 = closes[0]; ema_26 = closes[0]; macd_signal = 0.0; atr_ema = 0.0
    ema_up = 0.0; ema_down = 0.0
    
    # SuperTrend state
    st_ub = 0.0; st_lb = 0.0; st_trend = 1
    
    # ADX state
    dm_pos_ema = 0.0; dm_neg_ema = 0.0; adx_ema = 0.0
    
    kalmans = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        c = closes[i]; pc = closes[i-1]; h = highs[i]; l = lows[i]; v = volumes[i]
        ph = highs[i-1]; pl = lows[i-1]
        
        # Kalman Filter
        kf_p += kf_q
        k = kf_p / (kf_p + kf_r)
        kf_x += k * (c - kf_x)
        kf_p *= (1.0 - k)
        kalmans[i] = kf_x

        # MACD
        ema_12 = c * (2/13) + ema_12 * (1 - 2/13)
        ema_26 = c * (2/27) + ema_26 * (1 - 2/27)
        macd = ema_12 - ema_26
        macd_signal = macd * (2/10) + macd_signal * (1 - 2/10)
        macdh = macd - macd_signal
        
        # SMA & BB
        if i >= 19:
            w20 = closes[i-19:i+1]
            sma_20 = np.mean(w20)
            std20 = np.std(w20)
        else:
            sma_20 = c; std20 = 0.0
            
        bb_u = sma_20 + std20 * 2
        bb_l = sma_20 - std20 * 2
        bb_w = (bb_u - bb_l) / sma_20 if sma_20 != 0 else 0
        
        sma_50 = np.mean(closes[i-49:i+1]) if i >= 49 else c
            
        # ATR
        tr = max(h - l, max(abs(h - pc), abs(l - pc)))
        alpha_rsi = 1.0 / rsi_length
        atr_ema = tr * alpha_rsi + atr_ema * (1 - alpha_rsi)
        
        # Returns & Volatility
        ret = (c - pc) / pc if pc != 0 else 0.0
        if i >= 9:
            rets = np.zeros(10)
            for j in range(10): 
                pr_c = closes[i-10+j]
                rets[j] = (closes[i-9+j] - pr_c) / pr_c if pr_c != 0 else 0.0
            volatility = np.std(rets)
        else:
            volatility = 0.0
            
        momentum = c - closes[i-5] if i >= 5 else 0.0
        trend_strength = sma_20 - sma_50
        kalman_slope = kalmans[i] - kalmans[i-3] if i >= 3 else 0.0
        
        vol_ratio = 0.0
        if i >= 19:
            avg_v = np.mean(volumes[i-19:i+1])
            if avg_v > 0: vol_ratio = v / avg_v

        # RSI Backup
        delta = c - pc
        up = delta if delta > 0 else 0.0
        down = -delta if delta < 0 else 0.0
        ema_up = up * alpha_rsi + ema_up * (1 - alpha_rsi)
        ema_down = down * alpha_rsi + ema_down * (1 - alpha_rsi)
        if ema_down == 0: rsi = 100.0
        else: rsi = 100.0 - (100.0 / (1 + (ema_up / ema_down)))

        # --- SuperTrend (Multiplier=3.0, Period=10) ---
        hl2 = (h + l) / 2
        basic_ub = hl2 + 3.0 * atr_ema
        basic_lb = hl2 - 3.0 * atr_ema
        
        if i == 1:
            st_ub = basic_ub
            st_lb = basic_lb
        else:
            if basic_ub < st_ub or closes[i-1] > st_ub: st_ub = basic_ub
            if basic_lb > st_lb or closes[i-1] < st_lb: st_lb = basic_lb
        
        if st_trend == 1:
            if c < st_lb:
                st_trend = -1
                st_val = st_ub
            else:
                st_val = st_lb
        else:
            if c > st_ub:
                st_trend = 1
                st_val = st_lb
            else:
                st_val = st_ub

        # --- ADX (14) ---
        ph_val = highs[i-1]
        pl_val = lows[i-1]
        dm_pos = max(h - ph_val, 0.0) if (h - ph_val) > (pl_val - l) else 0.0
        dm_neg = max(pl_val - l, 0.0) if (pl_val - l) > (h - ph_val) else 0.0
        alpha_adx = 1.0 / 14
        dm_pos_ema = dm_pos * alpha_adx + dm_pos_ema * (1 - alpha_adx)
        dm_neg_ema = dm_neg * alpha_adx + dm_neg_ema * (1 - alpha_adx)
        
        di_pos = 100.0 * dm_pos_ema / atr_ema if atr_ema != 0 else 0.0
        di_neg = 100.0 * dm_neg_ema / atr_ema if atr_ema != 0 else 0.0
        den = di_pos + di_neg
        dx = 100.0 * abs(di_pos - di_neg) / den if den != 0 else 0.0
        adx_ema = dx * alpha_adx + adx_ema * (1 - alpha_adx)
            
        out[i, 0] = kf_x; out[i, 1] = macd; out[i, 2] = macdh; out[i, 3] = macd_signal
        out[i, 4] = sma_20; out[i, 5] = sma_50; out[i, 6] = bb_u; out[i, 7] = bb_l
        out[i, 8] = atr_ema; out[i, 9] = c; out[i, 10] = ret; out[i, 11] = volatility
        out[i, 12] = momentum; out[i, 13] = trend_strength; out[i, 14] = bb_w
        out[i, 15] = kalman_slope; out[i, 16] = vol_ratio; out[i, 17] = rsi
        out[i, 18] = st_val; out[i, 19] = st_ub; out[i, 20] = st_lb
        out[i, 21] = adx_ema; out[i, 22] = di_pos; out[i, 23] = di_neg
        
    state = np.array([kf_x, kf_p, ema_12, ema_26, macd_signal, atr_ema, ema_up, ema_down, 
                      st_ub, st_lb, float(st_trend), dm_pos_ema, dm_neg_ema, adx_ema], dtype=np.float64)
    return out, state

@njit(parallel=True, fastmath=True)
def numba_tick_update(c, h, l, v, closes, highs, lows, vols, kalmans, state, rsi_length):
    ph_val = highs[-1]; pl_val = lows[-1]
    closes[:-1] = closes[1:]; closes[-1] = c
    highs[:-1] = highs[1:]; highs[-1] = h
    lows[:-1] = lows[1:]; lows[-1] = l
    vols[:-1] = vols[1:]; vols[-1] = v
    pc = closes[-2]
    
    kf_x = state[0]; kf_p = state[1]; ema_12 = state[2]; ema_26 = state[3]
    macd_signal = state[4]; atr_ema = state[5]; ema_up = state[6]; ema_down = state[7]
    st_ub = state[8]; st_lb = state[9]; st_trend = int(state[10])
    dm_pos_ema = state[11]; dm_neg_ema = state[12]; adx_ema = state[13]

    kf_p += 1e-5
    k = kf_p / (kf_p + 0.01)
    kf_x += k * (c - kf_x)
    kf_p *= (1.0 - k)
    kalmans[:-1] = kalmans[1:]; kalmans[-1] = kf_x

    ema_12 = c * (2/13) + ema_12 * (1 - 2/13)
    ema_26 = c * (2/27) + ema_26 * (1 - 2/27)
    macd = ema_12 - ema_26
    macd_signal = macd * (2/10) + macd_signal * (1 - 2/10)
    macdh = macd - macd_signal
    
    w20 = closes[-20:]
    sma_20 = np.mean(w20)
    std20 = np.std(w20)
    bb_u = sma_20 + std20 * 2
    bb_l = sma_20 - std20 * 2
    bb_w = (bb_u - bb_l) / sma_20 if sma_20 != 0 else 0
    
    sma_50 = np.mean(closes[-50:])
    
    tr = max(h - l, max(abs(h - pc), abs(l - pc)))
    alpha_rsi = 1.0 / rsi_length
    atr_ema = tr * alpha_rsi + atr_ema * (1 - alpha_rsi)
    
    ret = (c - pc) / pc if pc != 0 else 0.0
    rets = np.zeros(10)
    for j in range(10):
        pr_c = closes[-11+j]
        rets[j] = (closes[-10+j] - pr_c) / pr_c if pr_c != 0 else 0.0
    volatility = np.std(rets)
    
    momentum = c - closes[-6]
    trend_strength = sma_20 - sma_50
    kalman_slope = kalmans[-1] - kalmans[-4]
    
    v_mean = np.mean(vols[-20:])
    vol_ratio = v / v_mean if v_mean > 0 else 0.0

    delta = c - pc
    up = delta if delta > 0 else 0.0
    down = -delta if delta < 0 else 0.0
    ema_up = up * alpha_rsi + ema_up * (1 - alpha_rsi)
    ema_down = down * alpha_rsi + ema_down * (1 - alpha_rsi)
    rsi = 100.0 if ema_down == 0 else 100.0 - (100.0 / (1 + (ema_up / ema_down)))

    # SuperTrend
    hl2 = (h + l) / 2
    basic_ub = hl2 + 3.0 * atr_ema
    basic_lb = hl2 - 3.0 * atr_ema
    if basic_ub < st_ub or pc > st_ub: st_ub = basic_ub
    if basic_lb > st_lb or pc < st_lb: st_lb = basic_lb
    if st_trend == 1:
        if c < st_lb:
            st_trend = -1
            st_val = st_ub
        else:
            st_val = st_lb
    else:
        if c > st_ub:
            st_trend = 1
            st_val = st_lb
        else:
            st_val = st_ub

    # ADX
    dm_pos = max(h - ph_val, 0.0) if (h - ph_val) > (pl_val - l) else 0.0
    dm_neg = max(pl_val - l, 0.0) if (pl_val - l) > (h - ph_val) else 0.0
    alpha_adx = 1.0 / 14
    dm_pos_ema = dm_pos * alpha_adx + dm_pos_ema * (1 - alpha_adx)
    dm_neg_ema = dm_neg * alpha_adx + dm_neg_ema * (1 - alpha_adx)
    di_pos = 100.0 * dm_pos_ema / atr_ema if atr_ema != 0 else 0.0
    di_neg = 100.0 * dm_neg_ema / atr_ema if atr_ema != 0 else 0.0
    den = di_pos + di_neg
    dx = 100.0 * abs(di_pos - di_neg) / den if den != 0 else 0.0
    adx_ema = dx * alpha_adx + adx_ema * (1 - alpha_adx)

    state[0] = kf_x; state[1] = kf_p; state[2] = ema_12; state[3] = ema_26
    state[4] = macd_signal; state[5] = atr_ema; state[6] = ema_up; state[7] = ema_down
    state[8] = st_ub; state[9] = st_lb; state[10] = float(st_trend)
    state[11] = dm_pos_ema; state[12] = dm_neg_ema; state[13] = adx_ema
    
    out = np.empty(24, dtype=np.float64)
    out[0] = kf_x; out[1] = macd; out[2] = macdh; out[3] = macd_signal
    out[4] = sma_20; out[5] = sma_50; out[6] = bb_u; out[7] = bb_l
    out[8] = atr_ema; out[9] = c; out[10] = ret; out[11] = volatility
    out[12] = momentum; out[13] = trend_strength; out[14] = bb_w
    out[15] = kalman_slope; out[16] = vol_ratio; out[17] = rsi
    out[18] = st_val; out[19] = st_ub; out[20] = st_lb
    out[21] = adx_ema; out[22] = di_pos; out[23] = di_neg
    return out

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, '.env'))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] => %(message)s")

API_KEY = os.getenv('BINANCE_TESTNET_API_KEY', 'CcJYXKFDwceFxRAwh20bUmVrre3j39iM4MncJhVU1OD12TafHA1DvES2TD47FM8X')
API_SECRET = os.getenv('BINANCE_TESTNET_SECRET', 'A0zBLDzjfHiXBNNVSWzGcGtHij2QvantwTfXVN3J8lsxB0x3nu1Y8DO7N9HxDrPm')

# =========================================================
# STRATEGY PARAMETERS
# =========================================================
REST_BASE = "https://demo-fapi.binance.com"
INTERVAL = "15m"
KLINE_ENDPOINT_URL = f"{REST_BASE}/fapi/v1/klines"

# INTELLIGENCE PARAMETERS
TRADE_COOLDOWN_SECONDS = 90      # Increased cooldown to prevent overtrading
ATR_SL_MULTIPLIER = 1.2          # SL = entry ± (ATR × 1.2)
ATR_TP_MULTIPLIER = 2.5          # TP = entry ± (ATR × 2.5)
TIME_EXIT_SECONDS = 180           # Close stale trades after 3 minutes

# 🔧 Advanced Indicator Params
SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0
ADX_PERIOD = 14

# PRO EXIT PARAMETERS
TRAIL_ACTIVATION_ATR = 1.0       # Activate trailing after 1× ATR profit
TRAIL_STEP_ATR = 0.5             # Trail SL by 0.5× ATR increments
BREAKEVEN_ATR = 0.8              # Move SL to entry after 0.8× ATR profit
PARTIAL_TP_ATR = 1.5             # Close 50% at 1.5× ATR

# =========================================================
# 2. PRECISION (Binance LOT_SIZE / PRICE_FILTER compliance)
# =========================================================
decimal.getcontext().prec = 28

async def fetch_symbol_precision(symbol):
    """Fetch stepSize and tickSize from exchangeInfo — async"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REST_BASE}/fapi/v1/exchangeInfo") as r:
                data = await r.json()
                for s in data['symbols']:
                    if s['symbol'] == symbol:
                        step_size = tick_size = 0.001
                        for f in s['filters']:
                            if f['filterType'] == 'LOT_SIZE': step_size = float(f['stepSize'])
                            if f['filterType'] == 'PRICE_FILTER': tick_size = float(f['tickSize'])
                        return step_size, tick_size
    except:
        pass
    return 0.001, 0.10  # BTCUSDT defaults

def get_precision(size):
    s = f"{size:.10f}".rstrip('0')
    if '.' in s:
        return len(s.split('.')[1])
    return 0

def round_step(quantity, step_size):
    """Floor quantity to Binance stepSize precision safely"""
    return round(math.floor(quantity / step_size) * step_size, get_precision(step_size))

def round_tick(price, tick_size):
    """Round price to Binance tickSize precision safely"""
    return round(round(price / tick_size) * tick_size, get_precision(tick_size))

def fmt_qty(quantity, step_size):
    """Format quantity string safely"""
    decimals = get_precision(step_size)
    q = round_step(quantity, step_size)
    return f"{q:.{decimals}f}"

def fmt_price(price, tick_size):
    """Format price string safely"""
    decimals = get_precision(tick_size)
    p = round_tick(price, tick_size)
    return f"{p:.{decimals}f}"

def get_dynamic_threshold(atr, price):
    """Adaptive confidence threshold — lowered to 0.55 base for speed"""
    volatility = atr / price
    if volatility > 0.008:
        return 0.58  # High vol — faster entries
    elif volatility > 0.004:
        return 0.60  # Medium vol
    else:
        return 0.65  # Low vol — still tightest filter

async def get_top_gainers(limit=20):
    url = f"{REST_BASE}/fapi/v1/ticker/24hr"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            data = await r.json()
            usdt_pairs = [x for x in data if x['symbol'].endswith('USDT')]
            sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['priceChangePercent']), reverse=True)
            return sorted_pairs[:limit]

async def show_top_gainers():
    gainers = await get_top_gainers()
    print("\n🔥 TOP GAINERS (24H):\n")
    for i, coin in enumerate(gainers, 1):
        print(f"{i}. {coin['symbol']} | {coin['priceChangePercent']}%")
    return gainers

def get_safe_pairs():
    safe_list = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT",
        "SOLUSDT", "XRPUSDT", "ADAUSDT",
        "AVAXUSDT", "LINKUSDT", "XAUUSDT", "XAGUSDT", "PAXGUSDT"
    ]
    return safe_list

def show_safe_pairs():
    safe_list = get_safe_pairs()
    print("\n🛡️ SAFE PAIRS (Recommended):\n")
    for i, coin in enumerate(safe_list, 1):
        print(f"{i}. {coin}")
    return safe_list

def classify_regime(*args):
    """Classify market regime: TREND, RANGE, CHAOS, UNCERTAIN"""
    if len(args) == 1:
        df = args[0]
        latest = df.iloc[-1]
        atr = latest['atr_14']
        close_price = latest['close']
        sma_20 = latest['sma_20']
        sma_50 = latest['sma_50']
        bb_width = latest['bb_width']
    elif len(args) == 5:
        atr, close_price, sma_20, sma_50, bb_width = args
    else:
        raise ValueError("classify_regime accepts either 1 DataFrame or 5 numeric arguments")

    volatility = atr / close_price if close_price else 0
    trend = abs(sma_20 - sma_50) / close_price if close_price else 0

    if volatility < 0.003:
        return "CHAOS"   # dead market — no edge

    if trend > 0.002 and volatility > 0.004:
        return "TREND"   # strong directional move

    if bb_width < 0.01:
        return "RANGE"   # tight range — mean reversion territory

    return "UNCERTAIN"   # mixed signals — sit out or reduce size

def dynamic_risk(win_rate, volatility):
    """Dynamic position sizing — adjusts based on performance + volatility"""
    base = 0.02

    if win_rate > 0.6:
        base += 0.01    # winning streak — slightly more aggressive
    elif win_rate < 0.4:
        base -= 0.01    # losing streak — pull back

    if volatility > 0.01:
        base *= 0.5     # high vol cut

    return max(0.01, min(base, 0.05))

# =========================================================
# STEP 2: MULTI-STRATEGY ENGINES
# =========================================================
async def trend_signal(engine, features):
    """Trend engine — wraps existing LightGBM predict_with_confidence (async)"""
    signal, confidence = await engine.predict_with_confidence(features)
    return signal, confidence

def reversion_signal(price, bb_upper, bb_lower, rsi):
    """Mean-reversion engine — BB + RSI for range-bound markets"""
    if price <= bb_lower and rsi < 30:
        return 1, 0.65   # LONG at lower band + oversold

    if price >= bb_upper and rsi > 70:
        return -1, 0.65  # SHORT at upper band + overbought

    return 0, 0.0

async def micro_signal(client):
    """Microstructure engine — Order Book Imbalance directional edge"""
    obi = await calculate_obi(client)

    if obi > 0.3:
        return 1, 0.6    # Buy pressure dominant
    elif obi < -0.3:
        return -1, 0.6   # Sell pressure dominant

    return 0, 0.0

# =========================================================
# STEP 3: SIGNAL FUSION (Ensemble Voting)
# =========================================================
def combine_signals(regime, trend, rev, micro, st_signal, adx, htf_bias=0.0):
    """Weighted ensemble voting — routes to correct engines based on regime
    
    Incorporates SuperTrend and ADX strength into the voting process.
    """
    signals = []

    if regime == "TREND":
        # Weight trend signal more if ADX is high
        adx_boost = 1.0 + (adx / 100.0)
        signals.append((trend[0], trend[1] * adx_boost))
        signals.append(micro)
        if st_signal != 0:
            signals.append((st_signal, 0.4)) # Soft SuperTrend confirmation

    elif regime == "RANGE":
        signals.append(rev)
        signals.append(micro)
        signals.append(trend)

    else:
        return 0, 0

    score = sum([s[0] * s[1] for s in signals])
    score += htf_bias
    confidence = sum([s[1] for s in signals]) / len(signals)

    if score > 0:
        return 1, confidence
    elif score < 0:
        return -1, confidence
    return 0, 0

# =========================================================
# STEP 5: RL EXIT DECISION (Adaptive Trade Management)
# =========================================================
def rl_exit_decision(profit_atr, time_in_trade):
    """Adaptive exit logic based on profit-in-ATR and time held"""
    if profit_atr > 2.0:
        return "CLOSE"    # Lock big win

    if profit_atr > 1.0:
        return "TRAIL"    # Protect with trailing SL

    # Disabled max_hold_time per user request
    # if time_in_trade > 180:
    #     return "EXIT"     # Stale trade — cut it

    return "HOLD"         # Stay in

# =========================================================
# STEP 6: MULTI-COIN SCANNER (Localized Scoring v2.0)
# =========================================================
def score_market(df, symbol="UNKNOWN"):
    """Score a symbol's tradability using LOCALIZED, price-relative metrics.
    
    All components are normalized to the coin's own price scale so BTC's $67k
    price doesn't drown out ETH's $2k price.
    
    Sub-scores (max 100):
      Volatility : 0-30  (atr / close)
      Trend      : 0-30  (|sma_20 - sma_50| / close)
      Volume     : 0-20  (volume_ratio = current_vol / 20-period avg)
      Momentum   : 0-10  (|momentum| / close)
      BB Width   : 0-10  (already price-relative)
    """
    latest = df.iloc[-1]
    close_price = latest['close']
    if close_price <= 0:
        return 0.0

    # --- 1. Volatility (0-30) — ATR relative to price ---
    vol = latest['atr_14'] / close_price
    if vol > 0.010:
        vol_score = 30
    elif vol > 0.007:
        vol_score = 25
    elif vol > 0.004:
        vol_score = 18
    elif vol > 0.002:
        vol_score = 10
    else:
        vol_score = 0

    # --- 2. Trend Strength (0-30) — RELATIVE to price ---
    trend_rel = abs(latest['sma_20'] - latest['sma_50']) / close_price
    if trend_rel > 0.005:
        trend_score = 30
    elif trend_rel > 0.003:
        trend_score = 24
    elif trend_rel > 0.002:
        trend_score = 16
    elif trend_rel > 0.001:
        trend_score = 8
    else:
        trend_score = 0

    # --- 3. Volume Ratio (0-20) — already self-relative ---
    vol_ratio = latest['volume_ratio'] if 'volume_ratio' in df.columns else 0.0
    if vol_ratio > 2.0:
        volume_score = 20
    elif vol_ratio > 1.5:
        volume_score = 15
    elif vol_ratio > 1.1:
        volume_score = 10
    elif vol_ratio > 0.8:
        volume_score = 5
    else:
        volume_score = 0

    # --- 4. Momentum (0-10) — RELATIVE to price ---
    mom_rel = abs(latest['momentum']) / close_price if close_price > 0 else 0
    if mom_rel > 0.005:
        mom_score = 10
    elif mom_rel > 0.002:
        mom_score = 7
    elif mom_rel > 0.001:
        mom_score = 4
    else:
        mom_score = 0

    # --- 5. Bollinger Band Width (0-10) — already relative ---
    bb_w = latest['bb_width']
    if bb_w > 0.020:
        bb_score = 10
    elif bb_w > 0.012:
        bb_score = 7
    elif bb_w > 0.006:
        bb_score = 4
    else:
        bb_score = 0

    total = vol_score + trend_score + volume_score + mom_score + bb_score

    # --- Zero-Score Debug ---
    if total == 0:
        logging.warning(
            f"[SCAN DEBUG] {symbol} scored 0/100 | "
            f"Vol={vol:.5f}→{vol_score}, Trend={trend_rel:.5f}→{trend_score}, "
            f"VolRatio={vol_ratio:.2f}→{volume_score}, Mom={mom_rel:.5f}→{mom_score}, "
            f"BBW={bb_w:.5f}→{bb_score}"
        )
    elif total <= 10:
        logging.info(
            f"[SCAN DEBUG] {symbol} low score {total}/100 | "
            f"Vol={vol_score}, Trend={trend_score}, VolR={volume_score}, "
            f"Mom={mom_score}, BB={bb_score}"
        )

    return float(total)

async def scan_and_pick(data_api, ai_engine, symbols):
    """Scan multiple symbols, score them via LOCALIZED metrics, return ranked list.
    
    Each coin is scored independently on its own price-relative indicators.
    No pool-wide normalization is applied — scores come out of 100 naturally.
    """
    scores = []
    for symbol in symbols:
        try:
            klines = await data_api.fetch_klines(symbol, INTERVAL, limit=200)
            if not klines or len(klines) < 100:
                continue
            df_raw = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                    'ct', 'qav', 'not', 'tbb', 'tbq', 'ign'])
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
            df_raw.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_raw[col] = df_raw[col].astype(float)
            df_feat = ai_engine.feature_engineering(df_raw.copy(), rsi_length=ai_engine.best_rsi_len)
            # Regime is calculated fresh from THIS coin's df_feat — no state leakage
            regime = classify_regime(df_feat)
            sc = score_market(df_feat, symbol=symbol)
            scores.append((symbol, sc, regime))
            await asyncio.sleep(0.15)  # Rate limit
        except Exception as e:
            logging.warning(f"Scanner skip {symbol}: {e}")
    scores.sort(key=lambda x: x[1], reverse=True)
    # Scores are already 0-100 from localized sub-scores — no max-normalization needed
    return scores

async def get_top_volume_symbols(limit=50):
    """STEP 13: Fetch top N symbols by 24h quote volume."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REST_BASE}/fapi/v1/ticker/24hr") as r:
                data = await r.json()
                usdt_pairs = [x for x in data if x['symbol'].endswith('USDT')]
                sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
                return [p['symbol'] for p in sorted_pairs[:limit]]
    except:
        return get_safe_pairs()

async def rank_and_allocate(data_api, ai_engine, symbols, top_n=3):
    """STEP 13: Full market scan → rank → allocate capital weights to top N."""
    scores = await scan_and_pick(data_api, ai_engine, symbols)
    if not scores:
        return []
    # Filter to tradable regimes only
    tradable = [(sym, sc, reg) for sym, sc, reg in scores if reg in ("TREND", "RANGE")]
    if not tradable:
        tradable = scores[:top_n]  # fallback to best scored
    top = tradable[:top_n]
    # Capital allocation proportional to score
    total_score = sum(sc for _, sc, _ in top)
    if total_score == 0:
        total_score = 1
    allocations = []
    for sym, sc, reg in top:
        weight = sc / total_score
        allocations.append((sym, sc, reg, weight))
    return allocations

# =========================================================
# 3. RAW SIGNED HTTP CLIENT (Zero CCXT Dependency)
# =========================================================
class BinanceDemoClientAsync:
    """Direct HTTP client for demo-fapi.binance.com (Async)"""
    def __init__(self, symbol):
        self.symbol = symbol
        self.session = None

    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(headers={'X-MBX-APIKEY': API_KEY})

    async def close(self):
        if self.session:
            await self.session.close()

    def _sign(self, params: dict) -> str:
        params['timestamp'] = str(int(time.time() * 1000))
        params['recvWindow'] = '10000'
        qs = urllib.parse.urlencode(params)
        sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
        return f"{qs}&signature={sig}"

    async def get_balance_usdt(self) -> float:
        await self.init_session()
        signed = self._sign({})
        async with self.session.get(f"{REST_BASE}/fapi/v3/balance?{signed}") as r:
            r.raise_for_status()
            data = await r.json()
            for asset in data:
                if asset.get("asset") == "USDT":
                    # Safe fallback handling
                    balance = (
                        asset.get("walletBalance")
                        or asset.get("balance")
                        or asset.get("crossWalletBalance")
                        or 0
                    )
                    return float(balance)
        return 0.0

    async def get_position_risk(self) -> list:
        await self.init_session()
        signed = self._sign({'symbol': self.symbol})
        async with self.session.get(f"{REST_BASE}/fapi/v2/positionRisk?{signed}") as r:
            r.raise_for_status()
            return await r.json()

    async def get_position_amt(self) -> float:
        data = await self.get_position_risk()
        if data:
            pos = float(data[0]['positionAmt'])
            return pos
        return 0.0

    async def set_leverage(self, leverage: int):
        await self.init_session()
        signed = self._sign({'symbol': self.symbol, 'leverage': str(leverage)})
        async with self.session.post(f"{REST_BASE}/fapi/v1/leverage?{signed}") as r:
            return await r.json()

    async def change_margin_type(self, margin_type: str):
        await self.init_session()
        signed = self._sign({'symbol': self.symbol, 'marginType': margin_type})
        async with self.session.post(f"{REST_BASE}/fapi/v1/marginType?{signed}") as r:
            try: return await r.json()
            except: return {}

    async def place_market_order(self, side: str, quantity, reduce_only=False) -> dict:
        await self.init_session()
        params = {'symbol': self.symbol, 'side': side, 'type': 'MARKET', 'quantity': str(quantity)}
        if reduce_only:
            params['reduceOnly'] = 'true'
        signed = self._sign(params)
        async with self.session.post(f"{REST_BASE}/fapi/v1/order?{signed}") as r:
            return await r.json()

    async def place_algo_stop(self, side: str, trigger_price, quantity, order_type: str) -> dict:
        await self.init_session()
        params = {
            'symbol': self.symbol, 'side': side, 'algoType': 'CONDITIONAL',
            'type': order_type, 'triggerPrice': str(trigger_price),
            'quantity': str(quantity), 'reduceOnly': 'true'
        }
        signed = self._sign(params)
        async with self.session.post(f"{REST_BASE}/fapi/v1/algoOrder?{signed}") as r:
            return await r.json()

    async def get_open_orders(self) -> list:
        await self.init_session()
        try:
            signed = self._sign({'symbol': self.symbol})
            async with self.session.get(f"{REST_BASE}/fapi/v1/openOrders?{signed}") as r:
                r.raise_for_status()
                return await r.json()
        except:
            return []

    async def cancel_all_orders(self):
        await self.init_session()
        try:
            signed = self._sign({'symbol': self.symbol})
            async with self.session.delete(f"{REST_BASE}/fapi/v1/allOpenOrders?{signed}") as r: pass
        except: pass
        try:
            signed = self._sign({'symbol': self.symbol})
            async with self.session.delete(f"{REST_BASE}/fapi/v1/algoOpenOrders?{signed}") as r: pass
        except: pass

    async def cancel_order(self, order_id) -> dict:
        """Cancel a specific order by orderId."""
        await self.init_session()
        try:
            signed = self._sign({'symbol': self.symbol, 'orderId': str(order_id)})
            async with self.session.delete(f"{REST_BASE}/fapi/v1/order?{signed}") as r:
                return await r.json()
        except:
            return {}

    async def get_order(self, order_id) -> dict:
        """Fetch a specific order by orderId."""
        await self.init_session()
        try:
            signed = self._sign({'symbol': self.symbol, 'orderId': str(order_id)})
            async with self.session.get(f"{REST_BASE}/fapi/v1/order?{signed}") as r:
                r.raise_for_status()
                return await r.json()
        except:
            return {}

    async def place_limit_maker_order(self, side: str, quantity, price, time_in_force='GTX') -> dict:
        """Post-Only limit order (GTX) or Immediate-Or-Cancel (IOC)."""
        await self.init_session()
        params = {
            'symbol': self.symbol, 'side': side, 'type': 'LIMIT',
            'quantity': str(quantity), 'price': str(price),
            'timeInForce': time_in_force
        }
        signed = self._sign(params)
        async with self.session.post(f"{REST_BASE}/fapi/v1/order?{signed}") as r:
            return await r.json()

    async def get_order_book(self, limit=20) -> dict:
        """Fetch top N levels of bid/ask depth (public, no auth)."""
        await self.init_session()
        try:
            async with self.session.get(f"{REST_BASE}/fapi/v1/depth?symbol={self.symbol}&limit={limit}") as r:
                r.raise_for_status()
                return await r.json()
        except:
            return {'bids': [], 'asks': []}

    async def get_open_interest(self) -> float:
        """STEP 12: Fetch current Open Interest for the symbol."""
        await self.init_session()
        try:
            async with self.session.get(f"{REST_BASE}/fapi/v1/openInterest?symbol={self.symbol}") as r:
                r.raise_for_status()
                data = await r.json()
                return float(data.get('openInterest', 0))
        except:
            return 0.0

    async def get_open_interest_hist(self, period='5m', limit=2) -> list:
        """STEP 12: Fetch OI history to compute OI change."""
        await self.init_session()
        try:
            params = {'symbol': self.symbol, 'period': period, 'limit': limit}
            async with self.session.get(f"{REST_BASE}/futures/data/openInterestHist", params=params) as r:
                if r.status == 200:
                    return await r.json()
        except:
            pass
        return []

    async def get_agg_trades(self, limit=200) -> list:
        """STEP 12: Fetch recent aggregated trades for CVD calculation."""
        await self.init_session()
        try:
            async with self.session.get(f"{REST_BASE}/fapi/v1/aggTrades?symbol={self.symbol}&limit={limit}") as r:
                r.raise_for_status()
                return await r.json()
        except:
            return []

async def calculate_obi(client, levels=10):
    """Order Book Imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty). Range: -1.0 to +1.0"""
    try:
        book = await client.get_order_book(limit=levels)
        bid_qty = sum(float(b[1]) for b in book.get('bids', [])[:levels])
        ask_qty = sum(float(a[1]) for a in book.get('asks', [])[:levels])
        total = bid_qty + ask_qty
        if total == 0:
            return 0.0
        return (bid_qty - ask_qty) / total
    except:
        return 0.0

async def calculate_cvd(client, limit=200):
    """STEP 12: Cumulative Volume Delta from aggTrades. Positive = buy pressure, Negative = sell pressure."""
    try:
        trades = await client.get_agg_trades(limit=limit)
        if not trades:
            return 0.0
        buy_vol = sum(float(t['q']) for t in trades if not t.get('m', True))  # m=False means buyer is maker
        sell_vol = sum(float(t['q']) for t in trades if t.get('m', True))     # m=True means seller is maker
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total  # Normalized -1 to +1
    except:
        return 0.0

async def calculate_oi_change(client):
    """STEP 12: OI change ratio. Positive = new contracts opening, Negative = contracts closing."""
    try:
        hist = await client.get_open_interest_hist(period='5m', limit=2)
        if len(hist) >= 2:
            old_oi = float(hist[0].get('sumOpenInterest', 0))
            new_oi = float(hist[1].get('sumOpenInterest', 0))
            if old_oi > 0:
                return (new_oi - old_oi) / old_oi  # % change
        return 0.0
    except:
        return 0.0


# =========================================================
# 4. DATA API (Dynamic Timeframe-Aware Fetching)
# =========================================================

def _interval_metadata(interval):
    """Convert interval string to (human_name, minutes_per_candle, lookback_days).
    
    Lookback is tuned per timeframe to ensure enough candles for ML training
    while avoiding excessive API calls on short timeframes.
    """
    meta = {
        "1m":  ("1-Minute",   1,   7),     # 7 days  = ~10,080 candles
        "3m":  ("3-Minute",   3,   14),    # 14 days = ~6,720 candles
        "5m":  ("5-Minute",   5,   21),    # 21 days = ~6,048 candles
        "15m": ("15-Minute",  15,  90),    # 90 days = ~8,640 candles
        "30m": ("30-Minute",  30,  120),   # 120 days = ~5,760 candles
        "1h":  ("1-Hour",     60,  180),   # 180 days = ~4,320 candles
        "2h":  ("2-Hour",     120, 240),   # 240 days = ~2,880 candles
        "4h":  ("4-Hour",     240, 365),   # 365 days = ~2,190 candles
        "6h":  ("6-Hour",     360, 365),   # 365 days = ~1,460 candles
    }
    if interval in meta:
        return meta[interval]
    # Fallback: treat unknown as 15m-like
    return (interval.upper(), 15, 90)

class SmartBackoffAPI:
    def __init__(self):
        self.session = None
        self.ban_wait_until = 0

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_klines(self, symbol, interval, limit=1000, end_time=None):
        if self.session is None: self.session = aiohttp.ClientSession()
        if time.time() < self.ban_wait_until:
            await asyncio.sleep(self.ban_wait_until - time.time())
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time: params["endTime"] = end_time
        async with self.session.get(KLINE_ENDPOINT_URL, params=params) as res:
            if res.status in [429, 418]:
                retry_after = int(res.headers.get("Retry-After", 60))
                self.ban_wait_until = time.time() + retry_after
                return await self.fetch_klines(symbol, interval, limit, end_time)
            res.raise_for_status()
            return await res.json()

    async def get_historical_data(self, symbol, interval):
        human_name, mins_per_candle, lookback_days = _interval_metadata(interval)
        estimated_candles = (lookback_days * 24 * 60) // mins_per_candle
        logging.info(f"Fetching {lookback_days} days of {human_name} Klines for {symbol} (~{estimated_candles} candles)...")
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (lookback_days * 24 * 60 * 60 * 1000)
        all_klines = []
        current_end = now_ms
        while current_end > start_ms:
            batch = await self.fetch_klines(symbol, interval, limit=1000, end_time=current_end)
            if not batch: break
            all_klines = batch + all_klines
            current_end = batch[0][0] - 1
            await asyncio.sleep(0.2)
        if self.session: 
            await self.session.close()
            self.session = None
        logging.info(f"Fetched {len(all_klines)} {human_name} candles for {symbol}")
        return all_klines

# =========================================================
# 5. HFT BRAIN v2.0 (LightGBM + Confidence Gating + Walk-Forward)
# =========================================================

@njit
def _triple_barrier_numba(closes, tps, sls, time_limit):
    """Numba-compiled triple-barrier labeling loop with dynamic targets."""
    n = len(closes)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        tp = tps[i]
        sl = sls[i]
        end = min(i + time_limit, n)
        for j in range(i + 1, end):
            price = closes[j]
            if price >= tp:
                labels[i] = 1
                break
            elif price <= sl:
                labels[i] = -1
                break
    return labels

class AI_Brain_Module:
    def __init__(self):
        self.model = None
        self.best_rsi_len = 14
        self.features_list = [
            'kalman', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'sma_20', 'sma_50', 'bb_upper', 'bb_lower', 'atr_14', 'close',
            'returns', 'volatility', 'momentum', 'trend_strength',
            'bb_width', 'kalman_slope', 'volume_ratio', 'rsi',
            'supertrend', 'adx', 'di_plus', 'di_minus',
            'hour', 'day_of_week',
            'obi', 'cvd', 'oi_change',
            'htf_trend', 'htf_strength'
        ]

    def feature_engineering(self, df, rsi_length=14):
        c = df['close'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.zeros(len(c), dtype=np.float64)

        feats, state = fast_features_numba(c, h, l, v, rsi_length)

        cols = [
            'kalman', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'sma_20', 'sma_50', 'bb_upper', 'bb_lower', 'atr_14', 'close',
            'returns', 'volatility', 'momentum', 'trend_strength',
            'bb_width', 'kalman_slope', 'volume_ratio', 'rsi',
            'supertrend', 'supertrend_ub', 'supertrend_lb',
            'adx', 'di_plus', 'di_minus'
        ]

        import pandas as pd
        out_df = pd.DataFrame(feats, columns=cols, index=df.index[-len(feats):])
        
        # Merge back base columns
        out_df['open'] = df['open'].iloc[-len(feats):].values if 'open' in df.columns else np.nan
        out_df['high'] = df['high'].iloc[-len(feats):].values if 'high' in df.columns else np.nan
        out_df['low'] = df['low'].iloc[-len(feats):].values if 'low' in df.columns else np.nan
        out_df['volume'] = df['volume'].iloc[-len(feats):].values if 'volume' in df.columns else np.nan
        
        out_df['obi'] = df['obi'].iloc[-len(feats):].values if 'obi' in df.columns else 0.0
        out_df['cvd'] = df['cvd'].iloc[-len(feats):].values if 'cvd' in df.columns else 0.0
        out_df['oi_change'] = df['oi_change'].iloc[-len(feats):].values if 'oi_change' in df.columns else 0.0

        # 🔧 HTF (Higher Timeframe) features via wider-window SMAs
        # SMA(40) ≈ HTF SMA(20) at 2× timeframe, SMA(100) ≈ HTF SMA(50)
        close_series = out_df['close']
        htf_sma_20 = close_series.rolling(window=40, min_periods=1).mean()
        htf_sma_50 = close_series.rolling(window=100, min_periods=1).mean()
        out_df['htf_trend'] = (htf_sma_20 > htf_sma_50).astype(float)
        out_df['htf_strength'] = (htf_sma_20 - htf_sma_50) / close_series.replace(0, np.nan).fillna(1.0)
        
        # session-based features
        out_df['hour'] = out_df.index.hour
        out_df['day_of_week'] = out_df.index.dayofweek

        return out_df.iloc[50:].copy() if len(out_df) > 50 else out_df.copy()

    def label_data(self, df, time_limit=10):
        """STEP 10: Triple-Barrier Labeling — Dynamic ATR targets per row."""
        closes = df['close'].values.astype(np.float64)
        atrs = df['atr_14'].values.astype(np.float64)
        volatilities = atrs / (closes + 1e-9)
        
        # 🔧 Dynamic ATR multiplier: more volatility -> slightly wider targets to reduce noise
        # Scale multiplier between 1.2 and 2.5 based on volatility
        dynamic_mults = 1.2 + np.clip((volatilities - 0.004) * 150.0, 0.0, 1.3)
        
        tps = closes + atrs * dynamic_mults
        sls = closes - atrs * dynamic_mults
        
        labels = _triple_barrier_numba(closes, tps, sls, time_limit)
        df['target'] = labels
        # Map to binary for LightGBM classifier: 1 = profitable, 0 = not
        df['target'] = df['target'].map({1: 1, 0: 0, -1: 0})
        df.dropna(inplace=True)
        return df

    def walk_forward_validate(self, df, rsi_len, n_splits=5):
        """Walk-Forward Rolling Validation — train on past, test on future windows"""
        df = self.feature_engineering(df.copy(), rsi_length=rsi_len)
        df = self.label_data(df)

        X, y = df[self.features_list], df['target']
        total_len = len(X)
        
        if total_len < 500:
            return None, 0, 0, None
            
        split_size = total_len // (n_splits + 1)

        accuracies = []
        best_model = None
        best_acc = 0

        for i in range(n_splits):
            train_end = split_size * (i + 2)
            test_end = min(train_end + split_size, total_len)
            if test_end <= train_end: break

            X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
            X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

            model = LGBMClassifier(
                n_estimators=500, max_depth=10, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5,
                extra_trees=True, path_smooth=1.0,
                random_state=42, verbose=-1, n_jobs=-1
            )
            model.fit(X_train, y_train)

            acc = accuracy_score(y_test, model.predict(X_test))
            accuracies.append(acc)

            if acc > best_acc:
                best_acc = acc
                best_model = model

        avg_acc = np.mean(accuracies) if accuracies else 0
        return best_model, avg_acc, best_acc, df

    def train_model(self, klines_data, symbol, force_retrain=False):
        if not klines_data or len(klines_data) < 1000:
            logging.error("Not enough data to train LightGBM.")
            return False, None

        df_raw = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                    'ct', 'qav', 'not', 'tbb', 'tbq', 'ign'])
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
        df_raw.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']: df_raw[col] = df_raw[col].astype(float)

        logging.info(f"Raw rows: {len(df_raw)}")

        brain_file = f'hft_brain_{INTERVAL}_{symbol}.txt'
        if not force_retrain and os.path.exists(brain_file):
            import lightgbm as lgb
            logging.info("🧠 FAST START: Loading existing Brain directly from disk...")
            self.model = lgb.Booster(model_file=brain_file)
            self.best_rsi_len = 14  # Default for loaded model
            best_df = self.feature_engineering(df_raw.copy(), rsi_length=self.best_rsi_len)
            logging.info(f"✅ Brain Loaded! Indicators seeded securely in O(1) time.")
            return True, best_df

        logging.info("Initiating HFT BRAIN v2.0 Walk-Forward Grid Search (LightGBM)...")

        best_overall_acc = 0.0
        best_overall_model = None
        best_df = None

        for rsi_len in [7, 14, 21]:
            model, avg_acc, peak_acc, df = self.walk_forward_validate(df_raw.copy(), rsi_len)
            logging.info(f"RSI {rsi_len} | Avg WF Accuracy: {avg_acc*100:.2f}% | Peak: {peak_acc*100:.2f}%")

            if avg_acc > best_overall_acc:
                best_overall_acc = avg_acc
                best_overall_model = model
                self.best_rsi_len = rsi_len
                best_df = df

        logging.info(f"--- HFT BRAIN v2.0 LOCKED | Best Avg WF Accuracy: {best_overall_acc*100:.2f}% ---")
        
        if best_overall_model is None or best_df is None:
            logging.error("❌ Training failed — no valid model/data")
            return False, None

        self.model = best_overall_model
        self.model.booster_.save_model(f'hft_brain_{INTERVAL}_{symbol}.txt')
        logging.info(f"🧠 BRAIN SAVED: 'hft_brain_{INTERVAL}_{symbol}.txt'")
        
        return True, best_df

# 6. INTELLIGENT SCALPER v2.0 (Precision + Adaptive Threshold + Micro-Eval)
# =========================================================

class NumbaRollingCalculator:
    def __init__(self, ai, hist_df):
        self.ai = ai
        n = 100
        self.closes = hist_df['close'].values[-n:].astype(np.float64).copy()
        self.highs = hist_df['high'].values[-n:].astype(np.float64).copy()
        self.lows = hist_df['low'].values[-n:].astype(np.float64).copy()
        try: self.vols = hist_df['volume'].values[-n:].astype(np.float64).copy()
        except: self.vols = np.zeros(n, dtype=np.float64)
        
        c_all = hist_df['close'].values.astype(np.float64)
        h_all = hist_df['high'].values.astype(np.float64)
        l_all = hist_df['low'].values.astype(np.float64)
        try: v_all = hist_df['volume'].values.astype(np.float64)
        except: v_all = np.zeros(len(c_all), dtype=np.float64)
        
        feats, self.state = fast_features_numba(c_all, h_all, l_all, v_all, ai.best_rsi_len)
        # Seed Kalman to last real price — prevents 47k spikes on boot
        self.state[0] = c_all[-1]
        self.kalmans = feats[-n:, 0].copy()
        self.kalmans[:] = c_all[-1]  # Reset entire kalman buffer to real price
        
    def update(self, c, h, l, v):
        return numba_tick_update(c, h, l, v, self.closes, self.highs, self.lows, self.vols, self.kalmans, self.state, self.ai.best_rsi_len)

class LiveTradingEngine:
    def __init__(self, ai, history_df, symbol, leverage):
        self.leverage = leverage
        self.target_leverage = leverage
        self.symbol = symbol
        self.ws_url = f"wss://fstream.binancefuture.com/ws/{symbol.lower()}@kline_{INTERVAL}"
        self.ai = ai
        self.client = BinanceDemoClientAsync(self.symbol)

        import collections
        import sqlite3
        self.rolling_calc = NumbaRollingCalculator(ai, history_df)
        self.db_conn = sqlite3.connect(f'trading_state_{self.symbol}.db', isolation_level=None)
        self._init_db()

        self.step_size = 0.001
        self.tick_size = 0.10
        self.current_position = 0  # 1 (Long), -1 (Short), 0 (Flat)
        self.tick_sl = 0.0
        self.tick_tp = 0.0
        self.entry_price = 0.0
        self.last_qty = 0.0
        self.last_heartbeat = time.time()
        self.last_trade_time = 0
        self.last_micro_eval = 0
        self.last_msg_time = time.time()
        self.ws = None
        self.booted_trade = False
        self.trade_count = 0
        self.win_count = 0
        self.last_confidence = 0.0
        self.entry_time = 0
        self.realized_pnl = 0.0
        self.max_daily_loss_percent = 0.05
        self.start_balance = 0.0
        self.max_loss_amount = 0.0
        self.last_train_time = time.time()
        self.kill_switch_active = False   # STEP 8: Kill switch flag
        self.last_perf_log = 0.0          # Used for perf latency logging

        # PRO EXIT STATE
        self.trailing_active = False
        self.breakeven_locked = False
        self.partial_tp_done = False
        self.peak_price = 0.0
        self.initial_qty = 0.0
        self.initial_atr = 0.0
        self.emergency_shutdown = False  # Q-key emergency exit flag
        self.rsi_exhaustion_count = 0     # Consecutive ticks of extreme RSI against position

        # v4.0: SPREAD-AWARE + SAFETY GUARD + DEPTH STREAM + CSV LOGGING
        self.latest_price = 0.0           # Atomic price for safety guard
        self.live_orderbook = {'bids': [], 'asks': []}  # L2 depth stream cache
        self.entry_slippage = 0.0         # Slippage from mid-price entry
        self.entry_regime = ''            # Regime at entry time
        self.entry_intended_price = 0.0   # Mid-price we intended to fill at
        self.effective_leverage = leverage # Current effective leverage (may be reduced in high vol)
        self.csv_dir = os.path.join(SCRIPT_DIR, 'trade_logs')
        os.makedirs(self.csv_dir, exist_ok=True)

    def _init_db(self):
        c = self.db_conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            symbol TEXT, position INTEGER, entry_price REAL,
            tick_sl REAL, tick_tp REAL, last_qty REAL, initial_qty REAL,
            initial_atr REAL, peak_price REAL, trailing_active INTEGER,
            breakeven_locked INTEGER, partial_tp_done INTEGER, entry_time INTEGER
        )''')
        c.execute('INSERT OR IGNORE INTO state (id, position) VALUES (1, 0)')
        # Trade journal table
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, side TEXT,
            entry_price REAL, exit_price REAL,
            quantity REAL, pnl REAL,
            slippage REAL, fees_est REAL,
            confidence REAL, regime TEXT,
            duration_s REAL, exit_reason TEXT
        )''')

    def save_state(self):
        c = self.db_conn.cursor()
        c.execute('''UPDATE state SET 
            symbol=?, position=?, entry_price=?, tick_sl=?, tick_tp=?, 
            last_qty=?, initial_qty=?, initial_atr=?, peak_price=?, 
            trailing_active=?, breakeven_locked=?, partial_tp_done=?, entry_time=?
            WHERE id=1''', (
                self.symbol, self.current_position, self.entry_price, self.tick_sl, self.tick_tp,
                self.last_qty, self.initial_qty, self.initial_atr, self.peak_price,
                int(self.trailing_active), int(self.breakeven_locked), int(self.partial_tp_done), self.entry_time
            ))

    def load_state(self):
        c = self.db_conn.cursor()
        c.execute('SELECT * FROM state WHERE id=1')
        row = c.fetchone()
        if row and row[2] != 0 and row[1] == self.symbol:
            self.current_position = row[2]
            self.entry_price = row[3]
            self.tick_sl = row[4]
            self.tick_tp = row[5]
            self.last_qty = row[6]
            self.initial_qty = row[7]
            self.initial_atr = row[8]
            self.peak_price = row[9]
            self.trailing_active = bool(row[10])
            self.breakeven_locked = bool(row[11])
            self.partial_tp_done = bool(row[12])
            self.entry_time = row[13]
            return True
        return False

    async def setup(self):
        self.step_size, self.tick_size = await fetch_symbol_precision(self.symbol)
        logging.info(f"Precision loaded: stepSize={self.step_size} | tickSize={self.tick_size}")
        try:
            await self.client.set_leverage(self.leverage)
            logging.info(f"Leverage configured to {self.leverage}x")
            if self.leverage > 15:
                res = await self.client.change_margin_type('ISOLATED')
                if 'msg' in res and 'No need to change margin type' in res['msg']:
                    pass # Already ISOLATED
                logging.info(f"Margin mode switched to ISOLATED for > 15x leverage")
        except Exception as e:
            logging.error(f"Failed to set Leverage/Margin: {e}")

        # BOOT RECOVERY WITH SQLITE
        pos_amt = await self.client.get_position_amt()
        if pos_amt != 0:
            if self.load_state() and abs(self.last_qty - abs(pos_amt)) < self.step_size:
                logging.info(f"💾 BOOT RECOVERY: Restored active SQL Memory for {self.symbol} (Pos: {pos_amt}). Continuing perfectly without clearing orders!")
            else:
                logging.warning(f"⚠️ Binance Position ({pos_amt}) mismatch with DB! Resetting state & clearing orphaned orders...")
                await self.cleanup_orders()
                self.current_position = 0
                self.save_state()
        else:
            self.current_position = 0
            self.save_state()
            open_orders = await self.client.get_open_orders()
            if open_orders:
                logging.warning("⚠️ Found orphaned leftover orders on boot — clearing...")
                await self.cleanup_orders()

        self.start_balance = await self.client.get_balance_usdt()
        self.max_loss_amount = -(self.start_balance * self.max_daily_loss_percent)

    def log_trade(self, exit_price, exit_reason, regime=""):
        """Log completed trade to SQLite journal + CSV file."""
        try:
            duration = time.time() - self.entry_time if self.entry_time > 0 else 0
            side = "LONG" if self.current_position == 1 else "SHORT"
            if self.current_position == 1:
                pnl = (exit_price - self.entry_price) * self.last_qty
            else:
                pnl = (self.entry_price - exit_price) * self.last_qty
            slippage = getattr(self, 'entry_slippage', 0.0)
            fees_est = self.entry_price * self.last_qty * 0.0004
            entry_regime = getattr(self, 'entry_regime', regime)
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            c = self.db_conn.cursor()
            c.execute('''INSERT INTO trades (timestamp, symbol, side, entry_price, exit_price,
                quantity, pnl, slippage, fees_est, confidence, regime, duration_s, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (ts, self.symbol, side, self.entry_price, exit_price,
                 self.last_qty, pnl, slippage, fees_est, self.last_confidence,
                 entry_regime, duration, exit_reason))
            logging.info(f"📝 [TRADE LOG] {side} | PnL: ${pnl:.4f} | Fees: ${fees_est:.4f} | Slippage: ${slippage:.4f} | Duration: {duration:.0f}s | Reason: {exit_reason}")
            # Feature 5: CSV Performance Log
            self._log_trade_csv(ts, side, exit_price, pnl, slippage, fees_est, entry_regime, duration, exit_reason)
        except Exception as e:
            logging.warning(f"Trade log failed: {e}")

    def _log_trade_csv(self, ts, side, exit_price, pnl, slippage, fees_est, regime, duration, exit_reason):
        """Save each trade to a single trades.csv file."""
        try:
            csv_file = os.path.join(self.csv_dir, 'trades.csv')
            write_header = not os.path.exists(csv_file)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'timestamp', 'symbol', 'side', 'entry_price', 'exit_price',
                        'quantity', 'pnl', 'regime_at_entry', 'ml_confidence',
                        'slippage', 'fees_est', 'duration_s', 'exit_reason',
                        'effective_leverage'
                    ])
                writer.writerow([
                    ts, self.symbol, side, f"{self.entry_price:.6f}", f"{exit_price:.6f}",
                    f"{self.last_qty:.6f}", f"{pnl:.6f}", regime, f"{self.last_confidence:.4f}",
                    f"{slippage:.6f}", f"{fees_est:.6f}", f"{duration:.1f}", exit_reason,
                    f"{getattr(self, 'effective_leverage', self.leverage)}"
                ])
            logging.info(f"📊 [CSV] Trade #{self.trade_count} logged to {csv_file}")
        except Exception as e:
            logging.warning(f"CSV log failed: {e}")

    async def cleanup_orders(self):
        try:
            await self.client.cancel_all_orders()
            logging.info("✅ All leftover orders cancelled")
        except Exception as e:
            logging.warning(f"⚠️ Cleanup failed: {e}")

    async def print_heartbeat(self, c_price):
        try:
            data = await self.client.get_position_risk()
            actual_pos = 0.0
            exchange_pnl = 0.0
            if data:
                actual_pos = float(data[0]['positionAmt'])
                exchange_pnl = float(data[0]['unRealizedProfit'])

            if abs(actual_pos) < self.step_size:
                self.current_position = 0
                self.last_qty = 0
            else:
                self.current_position = 1 if actual_pos > 0 else -1
                self.last_qty = abs(actual_pos)

            self.save_state()

            usdt = await self.client.get_balance_usdt()
            side_str = "FLAT (Scanning)"
            if self.current_position == 1:
                side_str = "LONG"
            elif self.current_position == -1:
                side_str = "SHORT"

            pnl_sign = "+" if exchange_pnl >= 0 else ""
            wr = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
            logging.info(f"[WALLET] ${usdt:.2f} | [POS] {side_str} | [PNL] {pnl_sign}${exchange_pnl:.2f} | [TRADES] {self.trade_count} (WR: {wr:.0f}%)")
        except: pass

    def get_current_atr(self, df_feat):
        try:
            return float(df_feat['atr_14'].iloc[-1])
        except:
            return 50.0

    async def compute_wallet_qty(self, c_price, atr, confidence=None, regime=None):
        try:
            usdt_balance = await self.client.get_balance_usdt()
            
            # Historical Win Rate (Assume 50% for first 5 trades)
            win_rate = (self.win_count / self.trade_count) if self.trade_count >= 5 else 0.5
            # Risk/Reward Ratio
            rr_ratio = ATR_TP_MULTIPLIER / ATR_SL_MULTIPLIER if ATR_SL_MULTIPLIER > 0 else 2.0
            
            # Fractional Kelly Criterion (Half Kelly)
            kelly_pct = win_rate - ((1.0 - win_rate) / rr_ratio)
            kelly_pct = max(0.01, min(kelly_pct, 0.20)) # Cap between 1% and 20%
            fractional_kelly = kelly_pct * 0.5
            
            if self.leverage > 15:
                # 1. The "Bulldozer" Logic
                base_margin = 10.0
                calc_margin = base_margin
                volatility = atr / c_price
                
                if confidence is not None and regime is not None:
                    # Regime Check, Volatility Check, Profit Buffer Check
                    is_high_conf = confidence >= 0.95
                    is_trend = (regime == 'TREND')
                    is_vol_safe = volatility <= 0.008
                    is_profit_buffer = usdt_balance >= 4980.0
                    
                    if is_high_conf and is_trend and is_vol_safe and is_profit_buffer:
                        calc_margin = base_margin * 5.0 # Bulldozer Mode
                        logging.info(f"🚜 [BULLDOZER] High Conf ({confidence:.2f}) in Trend. Scaling UP to ${calc_margin:.2f}.")
                    elif regime == 'CHAOS':
                        calc_margin = base_margin * 0.5 # Safety Mode
                        logging.info(f"🎯 [SNIPER] Chaos detected. Scaling DOWN for safety to ${calc_margin:.2f}.")
                    else:
                        calc_margin = base_margin
                
                raw_qty = (calc_margin * self.leverage) / c_price
                qty = round_step(raw_qty, self.step_size)
                
                # 4. The "Safety Valve" Rule
                sl_distance = atr * ATR_SL_MULTIPLIER
                
                # Max allowed risk is the minimum of 1% of wallet and the Fractional Kelly percentage
                max_risk_pct = min(0.01, fractional_kelly)
                max_loss_usdt = usdt_balance * max_risk_pct
                max_qty_by_risk = max_loss_usdt / sl_distance
                
                if qty > max_qty_by_risk:
                    logging.info(f"🛡️ [SAFETY VALVE] Reduced size to cap loss at Kelly/1% of wallet (${max_loss_usdt:.2f}).")
                    qty = round_step(max_qty_by_risk, self.step_size)
            else:
                volatility = atr / c_price
                risk_per_trade = min(fractional_kelly, 0.05) # Cap risk at 5% max
                risk_amount = usdt_balance * risk_per_trade
                sl_distance = atr * ATR_SL_MULTIPLIER
                raw_qty = risk_amount / sl_distance
                qty = round_step(raw_qty, self.step_size)
                max_margin_usdt = usdt_balance * 0.05
                max_qty_allowed = (max_margin_usdt * self.leverage) / c_price
                if qty > max_qty_allowed:
                    qty = round_step(max_qty_allowed, self.step_size)
                 
            min_qty = round_step(11.0 / c_price + self.step_size, self.step_size)
            return max(qty, min_qty)
        except Exception as e:
            logging.warning(f"Wallet sync issue: {e}. Defaulting to 0.002")
            return 0.002

    def _predict_sync(self, features_dict):
        """Synchronous ML inference — called inside executor"""
        try:
            import pandas as pd
            
            # Create a 1-row DataFrame preserving exact feature names
            df = pd.DataFrame([{f: features_dict.get(f, 0.0) for f in self.ai.features_list}])
                
            if hasattr(self.ai.model, 'predict_proba'):
                proba = self.ai.model.predict_proba(df)[0]
            else:
                p_pos = self.ai.model.predict(df)[0]
                proba = [1.0 - p_pos, p_pos]
                
            confidence = max(proba)
            if 0.45 < proba[1] < 0.55:
                return 0, confidence
            signal = 1 if proba[1] > proba[0] else -1
            return signal, confidence
        except Exception as e:
            import traceback
            logging.error(f"[EXECUTOR] _predict_sync crashed: {e}\n{traceback.format_exc()}")
            return 0, 0.0

    async def predict_with_confidence(self, features_dict):
        """Async ML inference — runs prediction in executor to avoid blocking event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync, features_dict)

    def _get_raw_probas_sync(self, features_dict):
        try:
            import pandas as pd
            df = pd.DataFrame([{f: features_dict.get(f, 0.0) for f in self.ai.features_list}])
            if hasattr(self.ai.model, 'predict_proba'):
                proba = self.ai.model.predict_proba(df)[0]
                return proba[0], proba[1]
            else:
                p_pos = self.ai.model.predict(df)[0]
                return 1.0 - p_pos, p_pos
        except Exception as e:
            return 0.5, 0.5

    async def check_ml_exit(self, features_dict):
        """Logic-Based Exit: ML Flip, Confidence Decay, and RSI Exhaustion Override."""
        if self.current_position == 0: return
        try:
            # === RULE 1: ASYMMETRIC RSI OVERRIDE (Hard Exit) ===
            rsi = features_dict.get('rsi', 50.0)
            if self.current_position == -1 and rsi > 95:
                self.rsi_exhaustion_count += 1
            elif self.current_position == 1 and rsi < 5:
                self.rsi_exhaustion_count += 1
            else:
                self.rsi_exhaustion_count = 0

            if self.rsi_exhaustion_count > 5:
                # Minimum Profit Guard check for RSI Exhaustion
                latest_price = features_dict.get('close', self.entry_price)
                current_pnl = (latest_price - self.entry_price) * self.last_qty if self.current_position == 1 else (self.entry_price - latest_price) * self.last_qty
                estimated_fee = (self.entry_price * self.last_qty) * 0.0012
                min_profit = estimated_fee * 3.0
                
                if 0 < current_pnl < min_profit:
                    logging.info(f"[MIN-PROFIT GUARD] 🛡️ Blocked RSI Exhaustion. PnL (${current_pnl:.2f}) < 3x Fee (${min_profit:.2f})")
                    self.rsi_exhaustion_count = 0  # reset to avoid spamming
                    return

                logging.info(f"[RSI EXHAUSTION] 🚨 FORCE EXIT — RSI {rsi:.1f} against position for {self.rsi_exhaustion_count} consecutive ticks")
                pos_amt = await self.client.get_position_amt()
                exit_px = features_dict.get('close', self.entry_price)
                if pos_amt != 0:
                    close_side = 'SELL' if pos_amt > 0 else 'BUY'
                    close_qty = fmt_qty(abs(pos_amt), self.step_size)
                    await self.client.place_market_order(close_side, close_qty)
                    for _ in range(5):
                        new_pos = await self.client.get_position_amt()
                        if abs(new_pos) < self.step_size:
                            break
                        await asyncio.sleep(0.5)
                    if self.current_position == 1:
                        self.realized_pnl += (exit_px - self.entry_price) * self.last_qty
                    else:
                        self.realized_pnl += (self.entry_price - exit_px) * self.last_qty
                self.log_trade(exit_px, "RSI Exhaustion")
                self.last_exit_price = exit_px
                self.last_exit_direction = self.current_position
                self.last_exit_confidence = self.last_confidence
                await self.cleanup_orders()
                self.current_position = 0
                self.entry_time = 0
                self.last_trade_time = 0
                self.last_confidence = 0.0
                self.rsi_exhaustion_count = 0
                if hasattr(self, 'last_signal'): self.last_signal = 0
                if hasattr(self, 'strategy_decision'): self.strategy_decision = 0
                self.save_state()
                return  # Skip further ML checks

            # === RULE 2: DYNAMIC CONFIDENCE DECAY ===
            against_pct = abs(features_dict.get('close', self.entry_price) - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

            ml_flip_thresh = 0.90
            ml_decay_thresh = 0.35
            if against_pct >= 0.015:
                ml_flip_thresh *= 0.85   # 0.90 → 0.765
                ml_decay_thresh *= 0.80  # 0.35 → 0.28
                logging.info("[ANTI-STUBBORN] RSI/Price Divergence detected. Softening exit threshold.")

            loop = asyncio.get_event_loop()
            proba_short, proba_long = await loop.run_in_executor(None, self._get_raw_probas_sync, features_dict)
            
            hit_exit = False
            exit_reason = ""
            
            if self.current_position == 1:
                if proba_short > ml_flip_thresh:
                    hit_exit = True
                    exit_reason = f"ML Flip to SHORT (conf: {proba_short:.2f} > {ml_flip_thresh:.2f})"
                elif proba_long < ml_decay_thresh:
                    hit_exit = True
                    exit_reason = f"Confidence Decay (LONG conf: {proba_long:.2f} < {ml_decay_thresh:.2f})"
            elif self.current_position == -1:
                if proba_long > ml_flip_thresh:
                    hit_exit = True
                    exit_reason = f"ML Flip to LONG (conf: {proba_long:.2f} > {ml_flip_thresh:.2f})"
                elif proba_short < ml_decay_thresh:
                    hit_exit = True
                    exit_reason = f"Confidence Decay (SHORT conf: {proba_short:.2f} < {ml_decay_thresh:.2f})"
            
            if hit_exit and self.current_position != 0:
                # Minimum Profit Guard
                latest_price = features_dict.get('close', self.entry_price)
                current_pnl = (latest_price - self.entry_price) * self.last_qty if self.current_position == 1 else (self.entry_price - latest_price) * self.last_qty
                estimated_fee = (self.entry_price * self.last_qty) * 0.0012
                min_profit = estimated_fee * 3.0
                
                if 0 < current_pnl < min_profit:
                    logging.info(f"[MIN-PROFIT GUARD] 🛡️ Blocked {exit_reason}. PnL (${current_pnl:.2f}) < 3x Fee (${min_profit:.2f})")
                    return

                logging.info(f"[LOGIC-EXIT] 🧠 {exit_reason}. Executing Market Close.")
                pos_amt = await self.client.get_position_amt()
                exit_px = latest_price if 'latest_price' in locals() else self.entry_price
                if pos_amt != 0:
                    close_side = 'SELL' if pos_amt > 0 else 'BUY'
                    close_qty = fmt_qty(abs(pos_amt), self.step_size)
                    await self.client.place_market_order(close_side, close_qty)
                    for _ in range(5):
                        new_pos = await self.client.get_position_amt()
                        if abs(new_pos) < self.step_size:
                            break
                        await asyncio.sleep(0.5)
                    if self.current_position == 1:
                        self.realized_pnl += (exit_px - self.entry_price) * self.last_qty
                    else:
                        self.realized_pnl += (self.entry_price - exit_px) * self.last_qty
                
                self.log_trade(exit_px, f"ML-EXIT: {exit_reason}")
                self.last_exit_price = exit_px
                self.last_exit_direction = self.current_position
                self.last_exit_confidence = self.last_confidence

                await self.cleanup_orders()
                self.current_position = 0
                self.entry_time = 0
                self.last_trade_time = 0    # Bypass Cooldown for immediate Re-Entry
                self.last_confidence = 0.0  # Clear previous confidence locks
                self.rsi_exhaustion_count = 0
                if hasattr(self, 'last_signal'): self.last_signal = 0
                if hasattr(self, 'strategy_decision'): self.strategy_decision = 0
                self.save_state()
        except Exception as e:
            logging.error(f"ML Exit Check failed: {e}")

    def regime_filter_passed(self, latest_feat, c_price, atr):
        sma_20 = float(latest_feat['sma_20'].iloc[0])
        sma_50 = float(latest_feat['sma_50'].iloc[0])
        trend_threshold = atr * 0.30
        if abs(sma_20 - sma_50) < trend_threshold:
            return False, "Weak Trend (SMA20/50 too close)"
        if (atr / c_price) < 0.0005:
            return False, "Low Volatility (Dead zone)"
        return True, "Strong Regime"

    async def midprice_limit_entry(self, side, qty_str, c_price, signal):
        """SPREAD-AWARE ENTRY: Place GTX limit at mid-price, cancel after 2s if unfilled. No chasing."""
        try:
            book = await self.client.get_order_book(limit=5)
            if not book['bids'] or not book['asks']:
                logging.warning("[MID-PRICE] ⚠️ Empty order book — skipping entry")
                return False

            best_bid = float(book['bids'][0][0])
            best_ask = float(book['asks'][0][0])
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2.0
            mid_price_fmt = fmt_price(mid_price, self.tick_size)

            logging.info(f"[MID-PRICE] Bid: {best_bid} | Ask: {best_ask} | Spread: {spread:.4f} | Mid: {mid_price_fmt}")

            # Store intended price for slippage calculation
            self.entry_intended_price = float(mid_price_fmt)

            # Place GTX (Post-Only) limit order at mid-price
            entry_res = await self.client.place_limit_maker_order(side, qty_str, mid_price_fmt, time_in_force='GTX')

            if 'orderId' not in entry_res:
                logging.info(f"[MID-PRICE] ❌ GTX rejected (would cross as taker). No chase — waiting for next signal.")
                self.entry_slippage = 0.0
                return False

            order_id = entry_res['orderId']
            logging.info(f"[MID-PRICE] 📋 GTX Order placed @ {mid_price_fmt} | OrderID: {order_id} | Waiting 2s for fill...")

            # Wait exactly 2 seconds for fill
            await asyncio.sleep(2.0)

            # Check execution status
            order_status = await self.client.get_order(order_id)
            executed_qty = float(order_status.get('executedQty', 0.0))
            avg_price = float(order_status.get('avgPrice', 0.0)) if order_status.get('avgPrice') else float(mid_price_fmt)
            order_api_status = order_status.get('status', 'NEW')

            if executed_qty > 0:
                self.trade_count += 1
                # Calculate actual slippage
                self.entry_slippage = abs(avg_price - self.entry_intended_price)
                logging.info(f"[MID-PRICE] ✅ FILLED Qty: {executed_qty} @ {avg_price} | Slippage: {self.entry_slippage:.4f} | OrderID: {order_id}")
                return executed_qty
            else:
                # Cancel unfilled order — do NOT chase
                if order_api_status in ('NEW', 'PARTIALLY_FILLED'):
                    await self.client.cancel_order(order_id)
                self.entry_slippage = 0.0
                logging.info(f"[MID-PRICE] ⏳ NOT FILLED after 2s — cancelled. Waiting for next signal (no chase).")
                return False

        except Exception as e:
            logging.error(f"[MID-PRICE] Entry failed: {e}")
            self.entry_slippage = 0.0
            return False



    def check_orderbook_wall(self, signal):
        """Feature 4: Check L2 order book for walls blocking trade direction."""
        try:
            bids = self.live_orderbook.get('bids', [])
            asks = self.live_orderbook.get('asks', [])
            if not bids or not asks:
                return True, "No L2 data"

            top_bid_qty = sum(float(b[1]) for b in bids[:5])
            top_ask_qty = sum(float(a[1]) for a in asks[:5])

            if top_bid_qty == 0 or top_ask_qty == 0:
                return True, "Empty book side"

            ratio = top_bid_qty / top_ask_qty if top_ask_qty > 0 else 999

            if signal == -1 and ratio > 3.0:
                return False, f"BUY WALL detected (bid/ask ratio: {ratio:.1f}x) — blocking SHORT"

            if signal == 1 and (1.0 / ratio) > 3.0:
                return False, f"SELL WALL detected (ask/bid ratio: {1.0/ratio:.1f}x) — blocking LONG"

            return True, f"Book clear (bid/ask: {ratio:.2f})"
        except Exception as e:
            return True, f"Wall check error: {e}"

    async def purge_and_execute(self, signal, c_price, atr_value, confidence=None, regime=None):
        # STEP 7: SPREAD GUARD — skip trades during spread spikes
        try:
            book = await self.client.get_order_book(limit=5)
            if book['bids'] and book['asks']:
                best_bid = float(book['bids'][0][0])
                best_ask = float(book['asks'][0][0])
                spread = best_ask - best_bid
                if spread > atr_value * 2:
                    logging.warning(f"[SPREAD GUARD] ⛔ Spread ${spread:.2f} > 2×ATR — skipping trade")
                    return
        except:
            pass

        # Feature 4: ORDER BOOK WALL CHECK
        wall_passed, wall_info = self.check_orderbook_wall(signal)
        if not wall_passed:
            logging.warning(f"[WALL DETECT] ⛔ {wall_info} — skipping trade")
            return
        else:
            logging.info(f"[WALL DETECT] ✅ {wall_info}")

        try:
            await self.client.cancel_all_orders()
            pos_amt = await self.client.get_position_amt()
            if pos_amt != 0:
                if self.current_position == 1 and c_price > self.entry_price: self.win_count += 1
                elif self.current_position == -1 and c_price < self.entry_price: self.win_count += 1
                logging.info(f"Closing existing position ({pos_amt})...")
                close_qty = fmt_qty(abs(pos_amt), self.step_size)
                await self.client.place_market_order('SELL' if pos_amt > 0 else 'BUY', close_qty)
        except: pass

        if hasattr(self, 'target_leverage') and self.leverage != self.target_leverage:
            try:
                res = await self.client.set_leverage(self.target_leverage)
                if res and 'leverage' in res:
                    self.leverage = int(res['leverage'])
                    logging.info(f"🔧 [LEVERAGE APPLIED] Exchange confirmed {self.leverage}x for next trade")
                else:
                    logging.error(f"⚠️ Failed to apply leverage {self.target_leverage}x: {res}")
            except Exception as e:
                logging.error(f"⚠️ Exception applying leverage {self.target_leverage}x: {e}")

        qty = await self.compute_wallet_qty(c_price, atr_value, confidence, regime)
        qty_str = fmt_qty(qty, self.step_size)
        
        # 🔧 Dynamic ATR-based Targets: Scale targets based on market regime and confidence
        sl_mult = ATR_SL_MULTIPLIER
        tp_mult = ATR_TP_MULTIPLIER
        
        if regime == "TREND":
            tp_mult *= 1.2 # Let trends run
        elif regime == "RANGE":
            tp_mult *= 0.8 # Tighter targets in range
            
        if confidence and confidence > 0.90:
            tp_mult *= 1.1 # More confidence, slightly more target
        
        base_sl_dist = atr_value * sl_mult
        base_tp_dist = atr_value * tp_mult
        
        max_dist = c_price * 0.10
        if base_sl_dist > max_dist or base_sl_dist <= 0:
            sl_dist = c_price * 0.01
            tp_dist = c_price * 0.02
        else:
            sl_dist = max(base_sl_dist, self.tick_size * 5)
            tp_dist = max(base_tp_dist, self.tick_size * 10)

        conf = confidence if confidence else self.last_confidence
        
        obi = await calculate_obi(self.client)
        if signal == 1 and obi < -0.3: conf -= 0.05
        elif signal == -1 and obi > 0.3: conf -= 0.05

        # Store regime at entry for CSV logging
        self.entry_regime = regime if regime else ''

        if signal == 1:
            sl = round_tick(c_price - sl_dist, self.tick_size)
            tp = round_tick(c_price + tp_dist, self.tick_size)
            side, opp_side = 'BUY', 'SELL'
            self.current_position = 1
            logging.info(f"[HFT ASYNC] LONG @ {fmt_price(c_price, self.tick_size)} | SL: {fmt_price(sl, self.tick_size)} | TP: {fmt_price(tp, self.tick_size)} | {self.leverage}x")
        else:
            sl = round_tick(c_price + sl_dist, self.tick_size)
            tp = round_tick(c_price - tp_dist, self.tick_size)
            side, opp_side = 'SELL', 'BUY'
            self.current_position = -1
            logging.info(f"[HFT ASYNC] SHORT @ {fmt_price(c_price, self.tick_size)} | SL: {fmt_price(sl, self.tick_size)} | TP: {fmt_price(tp, self.tick_size)} | {self.leverage}x")

        self.tick_sl = sl
        self.tick_tp = tp
        self.entry_price = c_price
        self.last_qty = qty
        self.initial_qty = qty
        self.initial_atr = atr_value
        self.last_trade_time = time.time()
        self.entry_time = time.time()

        self.trailing_active = False
        self.breakeven_locked = False
        self.partial_tp_done = False
        self.peak_price = c_price
        
        self.save_state()

        try:
            # Feature 1: Mid-Price Entry (no chasing)
            filled_qty = await self.midprice_limit_entry(side, qty_str, c_price, signal)
            if not filled_qty:
                self.current_position = 0
                self.save_state()
                return

            if isinstance(filled_qty, float) and filled_qty > 0:
                qty_str = fmt_qty(filled_qty, self.step_size)
                self.last_qty = filled_qty
                self.initial_qty = filled_qty
                self.save_state()

            sl_str = fmt_price(sl, self.tick_size)
            await self.client.place_algo_stop(opp_side, sl_str, qty_str, 'STOP_MARKET')
            tp_str = fmt_price(tp, self.tick_size)
            await self.client.place_algo_stop(opp_side, tp_str, qty_str, 'TAKE_PROFIT_MARKET')
        except Exception as e: logging.error(f"Execution failed: {e}")

    async def evaluate_trade_signal(self, latest_feat_dict, c_price, atr):
        """STEP 9: Meta-Labeling — Strategy generates signals, ML filters them"""
        start_time = time.perf_counter()
        
        # Extracted directly from dict to avoid pandas overhead
        regime = classify_regime(latest_feat_dict['atr_14'], c_price, latest_feat_dict['sma_20'], latest_feat_dict['sma_50'], latest_feat_dict['bb_width'])

        t_sig = await trend_signal(self, latest_feat_dict)
        r_sig = reversion_signal(c_price, latest_feat_dict['bb_upper'], latest_feat_dict['bb_lower'], latest_feat_dict['rsi'])
        m_sig = await micro_signal(self.client)

        # Sub-signal debug logging
        logging.info(f"[SIGNALS] Trend: {t_sig} | Rev: {r_sig} | Micro: {m_sig} | Regime: {regime}")

        live_obi = await calculate_obi(self.client)
        live_cvd = await calculate_cvd(self.client)
        live_oi_change = await calculate_oi_change(self.client)
        
        latest_feat_dict['obi'] = live_obi
        latest_feat_dict['cvd'] = live_cvd
        latest_feat_dict['oi_change'] = live_oi_change

        # 🔧 SuperTrend & ADX extraction
        st_val = latest_feat_dict.get('supertrend', c_price)
        st_signal = 1 if c_price > st_val else -1
        adx = latest_feat_dict.get('adx', 0.0)

        # 🔧 HTF Bias: Soft directional influence from higher timeframe
        htf_bias = 0.2 if latest_feat_dict.get('htf_trend', 0.5) > 0.5 else -0.2
        signal, base_conf = combine_signals(regime, t_sig, r_sig, m_sig, st_signal, adx, htf_bias=htf_bias)

        # 🔧 Momentum Override: Detect fast dumps/pumps and force directional bias
        returns = latest_feat_dict.get('returns', 0.0)
        vol = latest_feat_dict.get('volatility', 0.0)
        momentum_override = 0
        if returns < -0.008 and vol > 0.005:
            momentum_override = -1
            logging.info(f"[MOMENTUM OVERRIDE] 📉 Fast dump | Ret: {returns:.4f} | Vol: {vol:.4f} → SHORT bias")
        elif returns > 0.008 and vol > 0.005:
            momentum_override = 1
            logging.info(f"[MOMENTUM OVERRIDE] 📈 Fast pump | Ret: {returns:.4f} | Vol: {vol:.4f} → LONG bias")

        if momentum_override != 0 and signal == 0:
            signal = momentum_override
            base_conf = max(base_conf, 0.60)
            logging.info(f"[MOMENTUM OVERRIDE] ⚡ Forced signal={signal} with conf={base_conf:.2f}")
        elif momentum_override != 0 and signal == momentum_override:
            base_conf = min(base_conf + 0.10, 1.0)
            logging.info(f"[MOMENTUM BOOST] 🚀 Momentum confirms signal — conf boosted to {base_conf:.2f}")

        final_signal = 0
        final_confidence = 0.0
        if signal != 0:
            ml_signal, ml_conf = await self.predict_with_confidence(latest_feat_dict)
            if ml_signal == signal and ml_conf > 0.6:
                final_signal = signal
                final_confidence = (base_conf + ml_conf) / 2
                logging.info(f"[META-LABEL] ✅ ML CONFIRMS | Regime: {regime} | Strategy: {signal} | ML: {ml_signal} ({ml_conf:.2f}) | Final: {final_confidence:.2f} | HTF: {'BULL' if htf_bias > 0 else 'BEAR'}")
            else:
                # If momentum override was active, allow with reduced confidence
                if momentum_override != 0 and ml_conf > 0.45:
                    final_signal = signal
                    final_confidence = base_conf * 0.85
                    logging.info(f"[MOMENTUM FORCE] ⚡ ML weak ({ml_conf:.2f}) but momentum override active — entering with {final_confidence:.2f}")
                else:
                    logging.info(f"[META-LABEL] ❌ ML VETOES | Regime: {regime} | Strategy: {signal} | ML: {ml_signal} ({ml_conf:.2f}) — skipping")
        else:
            # HIGH-CONFIDENCE ML SOLO: Strategy is neutral but ML may have strong conviction
            ml_signal, ml_conf = await self.predict_with_confidence(latest_feat_dict)
            if ml_conf >= 0.85 and ml_signal != 0 and regime != "CHAOS":
                final_signal = ml_signal
                final_confidence = ml_conf
                logging.info(f"[ML-SOLO] 🔥 High-Conf ML Override | Signal: {ml_signal} | Conf: {ml_conf:.2f} | Regime: {regime}")

        # === RULE 3: REGIME SENSITIVITY — Block entries in CHAOS ===
        rsi = latest_feat_dict.get('rsi', 50.0)
        if regime == "CHAOS":
            if final_signal == -1 and rsi > 70:
                logging.info(f"[REGIME BLOCK] ⛔ CHAOS + RSI {rsi:.1f} > 70 — blocking SHORT entry")
                final_signal = 0; final_confidence = 0.0
            elif final_signal == 1 and rsi < 30:
                logging.info(f"[REGIME BLOCK] ⛔ CHAOS + RSI {rsi:.1f} < 30 — blocking LONG entry")
                final_signal = 0; final_confidence = 0.0

        calc_time = (time.perf_counter() - start_time) * 1000
        if final_signal != 0 or time.time() - getattr(self, "last_perf_log", 0) > 60:
            logging.info(f"[PERF] Brain Latency: {calc_time:.2f}ms")
            self.last_perf_log = time.time()

        return final_signal, final_confidence, regime

    async def on_message(self, message):
        self.last_msg_time = time.time()
        import json
        data = json.loads(message)
        if 'k' not in data: return
        kline = data['k']
        c_price = float(kline['c'])
        self.latest_price = c_price  # Feature 3: Atomic update for safety guard

        if self.kill_switch_active or self.emergency_shutdown: return

        exchange_ts = int(kline.get('T', 0))
        latency_ms = 0
        if exchange_ts > 0:
            latency_ms = int(time.time() * 1000) - exchange_ts

        # Numba Array logic, entirely avoiding Pandas DataFrames!
        loop = asyncio.get_event_loop()
        feat_vals = await loop.run_in_executor(
            None, 
            self.rolling_calc.update, 
            c_price, float(kline['h']), float(kline['l']), float(kline['v'])
        )
        
        # Zip features list into a clean dictionary
        latest_feat_dict = {
            'kalman': feat_vals[0], 'MACD_12_26_9': feat_vals[1], 'MACDh_12_26_9': feat_vals[2], 'MACDs_12_26_9': feat_vals[3],
            'sma_20': feat_vals[4], 'sma_50': feat_vals[5], 'bb_upper': feat_vals[6], 'bb_lower': feat_vals[7],
            'atr_14': feat_vals[8], 'close': feat_vals[9], 'returns': feat_vals[10], 'volatility': feat_vals[11],
            'momentum': feat_vals[12], 'trend_strength': feat_vals[13], 'bb_width': feat_vals[14], 'kalman_slope': feat_vals[15],
            'volume_ratio': feat_vals[16], 'rsi': feat_vals[17],
            'supertrend': feat_vals[18], 'adx': feat_vals[21], 'di_plus': feat_vals[22], 'di_minus': feat_vals[23]
        }
        # session-based features
        import datetime
        now_dt = datetime.datetime.now()
        latest_feat_dict['hour'] = now_dt.hour
        latest_feat_dict['day_of_week'] = now_dt.weekday()
        
        # 🔧 HTF Features: Compute from rolling buffer (wider-window SMAs)
        # SMA(40) ≈ HTF SMA(20) at 2× timeframe, SMA(100) ≈ HTF SMA(50)
        rc = self.rolling_calc
        htf_sma_20 = float(np.mean(rc.closes[-40:]))
        htf_sma_50 = float(np.mean(rc.closes[-100:]))
        latest_feat_dict['htf_trend'] = 1.0 if htf_sma_20 > htf_sma_50 else 0.0
        latest_feat_dict['htf_strength'] = (htf_sma_20 - htf_sma_50) / c_price if c_price > 0 else 0.0
        
        atr = feat_vals[8]
        
        if not self.booted_trade:
            self.booted_trade = True
            # Seed Kalman to first real WS price — eliminates startup spikes
            self.rolling_calc.state[0] = c_price
            self.rolling_calc.kalmans[:] = c_price
            logging.info(f"🎯 [KALMAN SEED] Initialized to first WS tick: {c_price}")
            asyncio.create_task(self._async_brain_calculation(latest_feat_dict, c_price, atr, is_boot=True, latency_ms=latency_ms))
            self.last_heartbeat = time.time()
            return

        now = time.time()
        if not kline['x']:
            if now - self.last_heartbeat >= 15.0:
                await self.print_heartbeat(c_price)
                self.last_heartbeat = now

            # 5-Minute Buffer Health Log
            if now - getattr(self, "last_buffer_health_log", 0) >= 300:
                self.last_buffer_health_log = now
                logging.info(f"🩺 [BUFFER HEALTH] Kalman: {latest_feat_dict['kalman']:.4f} | RSI: {latest_feat_dict['rsi']:.2f}")

            if self.current_position != 0:
                # Add ML Exit periodic check (every 1 second)
                if now - getattr(self, "last_ml_exit_check", 0) > 1.0:
                    self.last_ml_exit_check = now
                    asyncio.create_task(self.check_ml_exit(latest_feat_dict))

                atr = self.initial_atr if self.initial_atr > 0 else self.get_current_atr(df_feat)
                unrealized_dist = c_price - self.entry_price if self.current_position == 1 else self.entry_price - c_price
                if self.current_position == 1 and c_price > self.peak_price: self.peak_price = c_price
                if self.current_position == -1 and c_price < self.peak_price: self.peak_price = c_price

                state_changed = False
                
                # DYNAMIC BREAK-EVEN
                pnl_percent = unrealized_dist / self.entry_price if self.entry_price > 0 else 0
                if not self.breakeven_locked and pnl_percent >= 0.0015:
                    if self.current_position == 1:
                        new_sl = round_tick(self.entry_price * 1.0002, self.tick_size)
                    else:
                        new_sl = round_tick(self.entry_price * 0.9998, self.tick_size)
                    
                    self.tick_sl = new_sl
                    self.breakeven_locked = True
                    state_changed = True
                    logging.info(f"[DYNAMIC BREAK-EVEN] 🔒 +0.15% Hit! SL moved to {fmt_price(new_sl, self.tick_size)}")

                # PARTIAL TP
                if not self.partial_tp_done and unrealized_dist >= (PARTIAL_TP_ATR * atr):
                    half_qty = round_step(self.initial_qty / 2, self.step_size)
                    if half_qty > 0:
                        opp_side = 'SELL' if self.current_position == 1 else 'BUY'
                        # BG Task Order (No Blocking)
                        asyncio.create_task(self.client.place_market_order(opp_side, fmt_qty(half_qty, self.step_size)))
                        partial_pnl = unrealized_dist * half_qty
                        self.last_qty = round_step(self.last_qty - half_qty, self.step_size)
                        self.realized_pnl += partial_pnl
                        self.partial_tp_done = True
                        state_changed = True
                        logging.info(f"[PARTIAL-TP] 💰 Closed 50% | Locked +${partial_pnl:.2f}")

                # TRAILING SL
                if unrealized_dist >= (TRAIL_ACTIVATION_ATR * atr):
                    if not self.trailing_active:
                        self.trailing_active = True
                        state_changed = True
                        logging.info(f"[TRAILING] 🏃 Activated!")
                    trail_dist = TRAIL_STEP_ATR * atr
                    if self.current_position == 1:
                        new_trail_sl = round_tick(self.peak_price - trail_dist, self.tick_size)
                        if new_trail_sl > self.tick_sl:
                            self.tick_sl = new_trail_sl
                            state_changed = True
                            logging.info(f"[TRAILING] ⬆️ SL raised to {fmt_price(new_trail_sl, self.tick_size)}")
                    else:
                        new_trail_sl = round_tick(self.peak_price + trail_dist, self.tick_size)
                        if new_trail_sl < self.tick_sl:
                            self.tick_sl = new_trail_sl
                            state_changed = True
                            logging.info(f"[TRAILING] ⬇️ SL lowered to {fmt_price(new_trail_sl, self.tick_size)}")

                if state_changed:
                    self.save_state()

                # STEP 5: RL EXIT DECISION (replaces fixed TIME_EXIT_SECONDS)
                profit_atr = unrealized_dist / atr if atr > 0 else 0
                time_in_trade = now - self.entry_time if self.entry_time > 0 else 0
                exit_decision = rl_exit_decision(profit_atr, time_in_trade)

                is_sl_tp = False
                hit_exit = False
                if self.current_position == 1 and (c_price >= self.tick_tp or c_price <= self.tick_sl): 
                    hit_exit = True
                    is_sl_tp = True
                elif self.current_position == -1 and (c_price <= self.tick_tp or c_price >= self.tick_sl): 
                    hit_exit = True
                    is_sl_tp = True

                # RL overrides
                if not is_sl_tp and (exit_decision == "CLOSE" or exit_decision == "EXIT"):
                    current_pnl = (c_price - self.entry_price) * self.last_qty if self.current_position == 1 else (self.entry_price - c_price) * self.last_qty
                    estimated_fee = (self.entry_price * self.last_qty) * 0.0012
                    min_profit = estimated_fee * 3.0
                    
                    if 0 < current_pnl < min_profit:
                        logging.info(f"[MIN-PROFIT GUARD] 🛡️ Blocked RL-EXIT. PnL (${current_pnl:.2f}) < 3x Fee (${min_profit:.2f})")
                    else:
                        hit_exit = True
                        logging.info(f"[RL-EXIT] 🧠 Decision: {exit_decision} | Profit ATR: {profit_atr:.2f} | Time: {time_in_trade:.0f}s")
                elif exit_decision == "TRAIL" and not self.trailing_active:
                    self.trailing_active = True
                    state_changed = True
                    logging.info(f"[RL-EXIT] 🧠 Force-activated trailing at {profit_atr:.2f}× ATR")

                if hit_exit:
                    self.last_exit_price = c_price
                    self.last_exit_direction = self.current_position
                    self.last_exit_confidence = self.last_confidence
                    
                    pnl = unrealized_dist * self.last_qty
                    if pnl > 0: self.win_count += 1
                    
                    exit_reason = f"SL/TP Hit" if exit_decision == "HOLD" else f"RL-EXIT: {exit_decision}"
                    
                    pos_amt = await self.client.get_position_amt()
                    if pos_amt != 0:
                        close_side = 'SELL' if pos_amt > 0 else 'BUY'
                        close_qty = fmt_qty(abs(pos_amt), self.step_size)
                        await self.client.place_market_order(close_side, close_qty)
                        for _ in range(5):
                            new_pos = await self.client.get_position_amt()
                            if abs(new_pos) < self.step_size:
                                break
                            await asyncio.sleep(0.5)

                    # Log trade AFTER market order and confirmation
                    self.log_trade(c_price, exit_reason)

                    await self.cleanup_orders()
                    self.current_position = 0
                    self.entry_time = 0
                    self.last_trade_time = 0    # Bypass Cooldown for immediate Re-Entry
                    self.last_confidence = 0.0  # Clear previous confidence locks
                    self.rsi_exhaustion_count = 0
                    if hasattr(self, 'last_signal'): self.last_signal = 0
                    if hasattr(self, 'strategy_decision'): self.strategy_decision = 0
                    self.save_state()
                    self.realized_pnl += pnl

                    # STEP 8: KILL SWITCH
                    if self.realized_pnl <= self.max_loss_amount:
                        logging.error(f"🚨 KILL SWITCH: Daily loss ${self.realized_pnl:.2f} exceeded limit ${self.max_loss_amount:.2f}. STOPPING BOT.")
                        self.kill_switch_active = True
                        await self.cleanup_orders()
                        return
            else:
                # We are FLAT. Evaluate signals intra-candle to prevent stalling.
                if now - self.last_trade_time >= TRADE_COOLDOWN_SECONDS:
                    # 🔧 Fast Scan: Detect momentum spike / volatility expansion for instant evaluation
                    ret = latest_feat_dict.get('returns', 0.0)
                    vol = latest_feat_dict.get('volatility', 0.0)
                    is_momentum_spike = abs(ret) > 0.005 and vol > 0.003
                    scan_interval = 0.2 if is_momentum_spike else 1.0  # 200ms during spikes, 1s normal
                    if now - getattr(self, "last_scan_time", 0) >= scan_interval:
                        self.last_scan_time = now
                        if is_momentum_spike:
                            logging.info(f"[FAST SCAN] ⚡ Momentum spike | Ret: {ret:.4f} | Vol: {vol:.4f} — instant eval")
                        asyncio.create_task(self._async_brain_calculation(latest_feat_dict, c_price, atr, is_boot=False, latency_ms=latency_ms))
            return

        # === CANDLE CLOSE: MULTI-STRATEGY SIGNAL FUSION ===
        if time.time() - self.last_trade_time < TRADE_COOLDOWN_SECONDS: return

        # Dispatch Brain to a separate task immediately so WS loop is not blocked by HTTP/ML overhead!
        asyncio.create_task(self._async_brain_calculation(latest_feat_dict, c_price, atr, is_boot=False, latency_ms=latency_ms))

    async def _async_brain_calculation(self, latest_feat, c_price, atr, is_boot=False, latency_ms=0):
        """Runs the entire prediction cycle in the background without blocking the WS tick consumer."""
        try:
            signal, confidence, regime = await self.evaluate_trade_signal(latest_feat, c_price, atr)
            self.last_confidence = confidence
            threshold = get_dynamic_threshold(atr, c_price)

            # Dynamic Sniper Threshold (Safety Logic)
            if self.leverage >= 30:
                threshold = max(threshold, 0.80)
            elif self.leverage >= 15:
                threshold = max(threshold, 0.75)

            if is_boot:
                logging.info(f"--- HFT ENGINE v3.0 ONLINE (MULTI-STRATEGY + FUSION) | Regime: {regime} | Thresh: {threshold*100:.0f}% ---")

            if signal == 0: return

            # ML VETO: Check Latency Guard AFTER calculating ml confidence
            if latency_ms > 3000:
                if confidence >= 0.80:
                    logging.info(f"[LATENCY GUARD VETO] 🔥 {latency_ms}ms delay ignored due to HIGH ML Conviction ({confidence*100:.1f}%)")
                else:
                    logging.warning(f"[LATENCY GUARD] ⚠️ {latency_ms}ms delay & Conf {confidence:.2f} < 0.80 — skipping trade")
                    return

            if self.current_position == 0:
                if confidence >= threshold:
                    # Price Gap Rule: Anti-Whipsaw for Re-Entries
                    if hasattr(self, 'last_exit_direction') and getattr(self, 'last_exit_direction', 0) == signal:
                        is_better_price = False
                        if signal == 1 and c_price < getattr(self, 'last_exit_price', 0): is_better_price = True
                        elif signal == -1 and c_price > getattr(self, 'last_exit_price', float('inf')): is_better_price = True
                        
                        has_more_conf = confidence >= getattr(self, 'last_exit_confidence', 0) + 0.05
                        
                        if not (is_better_price or has_more_conf):
                            logging.info(f"[FEE-CHURN GUARD] ⛔ Skipped immediate re-entry. Price not better and Conf ({confidence:.2f}) not +5% > {getattr(self, 'last_exit_confidence', 0):.2f}.")
                            return

                    asyncio.create_task(self.purge_and_execute(signal, c_price, atr, confidence, regime))
            elif self.current_position != signal:
                rev_threshold = threshold + 0.02
                if confidence >= rev_threshold:
                    asyncio.create_task(self.purge_and_execute(signal, c_price, atr, confidence, regime))
        except Exception as e:
            import traceback
            logging.error(f"[ASYNC BRAIN] Inference task crashed: {e}\n{traceback.format_exc()}")

    async def emergency_exit(self):
        """Q-KEY: Close all positions, cancel all orders, full cleanup."""
        logging.info("\n🚨 [EMERGENCY EXIT] Initiated by user...")
        self.kill_switch_active = True
        try:
            await self.client.cancel_all_orders()
            logging.info("✅ All orders cancelled.")
        except Exception as e:
            logging.error(f"Cancel orders failed: {e}")

        try:
            pos_amt = await self.client.get_position_amt()
            if pos_amt != 0:
                close_side = 'SELL' if pos_amt > 0 else 'BUY'
                close_qty = fmt_qty(abs(pos_amt), self.step_size)
                resp = await self.client.place_market_order(close_side, close_qty, reduce_only=True)
                if 'code' in resp:
                    logging.error(f"Failed to close position: {resp}")
                else:
                    logging.info(f"✅ Position closed: {pos_amt} → 0")
            else:
                logging.info("✅ No open position.")
        except Exception as e:
            logging.error(f"Position close failed: {e}")

        try:
            await self.client.close()
        except:
            pass
        self.current_position = 0
        self.save_state()
        self.emergency_shutdown = True
        logging.info("✅ [EMERGENCY EXIT] Complete. Bot stopped.")

    async def _safety_guard_loop(self):
        """Feature 3: Independent safety guard — checks SL every 200ms even during brain computation."""
        logging.info("[SAFETY GUARD] 🛡️ Background SL monitor started (200ms interval)")
        while not self.emergency_shutdown:
            try:
                await asyncio.sleep(0.2)
                if self.current_position == 0 or self.latest_price <= 0:
                    continue

                price = self.latest_price
                sl_hit = False

                if self.current_position == 1 and price <= self.tick_sl:
                    sl_hit = True
                elif self.current_position == -1 and price >= self.tick_sl:
                    sl_hit = True

                if sl_hit:
                    logging.warning(f"[SAFETY GUARD] 🚨 SL BREACH @ {price:.2f} (SL: {self.tick_sl:.2f}) — EMERGENCY CLOSE")
                    try:
                        pos_amt = await self.client.get_position_amt()
                        if pos_amt != 0:
                            close_side = 'SELL' if pos_amt > 0 else 'BUY'
                            close_qty = fmt_qty(abs(pos_amt), self.step_size)
                            await self.client.place_market_order(close_side, close_qty)
                            for _ in range(5):
                                new_pos = await self.client.get_position_amt()
                                if abs(new_pos) < self.step_size:
                                    break
                                await asyncio.sleep(0.5)
                            self.log_trade(price, "SAFETY GUARD SL")
                        await self.cleanup_orders()
                        self.current_position = 0
                        self.entry_time = 0
                        self.save_state()
                    except Exception as e:
                        logging.error(f"[SAFETY GUARD] Close failed: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"[SAFETY GUARD] Error: {e}")

    async def _depth_stream_loop(self):
        """Feature 4: WebSocket L2 depth stream — continuously updates live order book."""
        depth_url = f"wss://fstream.binancefuture.com/ws/{self.symbol.lower()}@depth20@100ms"
        logging.info(f"[DEPTH STREAM] 📡 Connecting to L2 depth stream...")
        while not self.emergency_shutdown:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(depth_url, timeout=30) as ws:
                        logging.info("[DEPTH STREAM] ✅ L2 Order Book connected")
                        async for msg in ws:
                            if self.emergency_shutdown:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    self.live_orderbook = {
                                        'bids': data.get('b', data.get('bids', [])),
                                        'asks': data.get('a', data.get('asks', []))
                                    }
                                except:
                                    pass
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.emergency_shutdown:
                    logging.warning(f"[DEPTH STREAM] Reconnecting in 5s... ({e})")
                    await asyncio.sleep(5)

    async def _ws_ping_loop(self):
        """Asynchronous Heartbeat task to keep WS alive infinitely."""
        while not self.emergency_shutdown:
            await asyncio.sleep(60)
            if hasattr(self, 'ws') and self.ws and not self.ws.closed:
                try:
                    await self.ws.ping()
                except:
                    pass

    async def _keyboard_listener(self):
        """Async listener: Q+Enter = emergency exit, number+Enter = change leverage for next trade."""
        loop = asyncio.get_event_loop()
        while not self.emergency_shutdown:
            try:
                user_input = await loop.run_in_executor(None, lambda: input())
                cleaned = user_input.strip().upper()
                if cleaned == 'Q':
                    await self.emergency_exit()
                    return
                # Try to parse as leverage number
                try:
                    new_lev = int(user_input.strip())
                    if 1 <= new_lev <= 125:
                        old_lev = getattr(self, 'target_leverage', self.leverage)
                        self.target_leverage = new_lev
                        logging.info(f"🔧 [LEVERAGE QUEUED] {old_lev}x → {new_lev}x (will apply when flat before next trade)")
                    else:
                        logging.warning(f"⚠️ Invalid leverage {new_lev}. Must be 1-125.")
                except ValueError:
                    pass  # Not a number, not Q — ignore
            except (EOFError, OSError):
                await asyncio.sleep(1)
            except:
                await asyncio.sleep(1)

    async def run(self):
        await self.setup()
        logging.info("Subscribing to realtime HFT streams via aiohttp...")
        logging.info("🚨 Press Q + Enter to EMERGENCY EXIT | Type a number + Enter to change leverage (e.g. '5' = 5x)")

        # Start listeners in background
        kb_task = asyncio.create_task(self._keyboard_listener())
        ping_task = asyncio.create_task(self._ws_ping_loop())
        safety_task = asyncio.create_task(self._safety_guard_loop())   # Feature 3: Async Safety Guard
        depth_task = asyncio.create_task(self._depth_stream_loop())    # Feature 4: L2 Depth Stream

        while not self.emergency_shutdown:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.ws_url, timeout=30) as ws:
                        self.ws = ws
                        logging.info("WebSocket Connected Async!")
                        async for msg in ws:
                            if self.emergency_shutdown:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self.on_message(msg.data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logging.error(f"WS Error: {ws.exception()}")
                                break
            except Exception as e:
                if self.emergency_shutdown:
                    break
                logging.error(f"WS Crash: {e}")
            if not self.emergency_shutdown:
                logging.warning("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

        kb_task.cancel()
        ping_task.cancel()
        safety_task.cancel()
        depth_task.cancel()
        for t in [kb_task, ping_task, safety_task, depth_task]:
            try:
                await t
            except asyncio.CancelledError:
                pass

        # Properly close aiohttp client session to prevent "Unclosed client session" error
        try:
            await self.client.close()
            logging.info("✅ Client session closed cleanly.")
        except Exception as e:
            logging.warning(f"Client session close error: {e}")
class MarketTimingController:
    def __init__(self, ai):
        self.ai = ai

    def analyze_market(self, df):
        df_feat = self.ai.feature_engineering(df.copy(), rsi_length=self.ai.best_rsi_len)
        latest = df_feat.iloc[-1]

        price = latest['close']
        atr = latest['atr_14']
        volatility = atr / price

        trend = abs(latest['sma_20'] - latest['sma_50']) / price
        bb_width = latest['bb_width']
        momentum = abs(latest['momentum']) / price

        score = 0

        # Volatility
        if volatility > 0.007:
            score += 2
        elif volatility > 0.004:
            score += 1

        # Trend
        if trend > 0.002:
            score += 2
        elif trend > 0.001:
            score += 1

        # Momentum
        if momentum > 0.002:
            score += 1

        # Bollinger expansion
        if bb_width > 0.012:
            score += 1

        # Classification
        if score >= 5:
            return "BEST", score
        elif score >= 3:
            return "OKAY", score
        else:
            return "BAD", score


async def cli_flow():
    global INTERVAL
    logging.info("--- HFT Antigravity v3.0 Execution Sequence Started ---")

    print("\nSelect Timeframe (Analysis & Trading):")
    print("Standard options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h")
    print("Note: Custom intervals like 2m, 4m, 10m are mapped to closest Binance intervals.")
    tf_input = input("Enter timeframe (e.g., 15m, 1h) [Default: 15m]: ").strip().lower()
    
    tf_map = {
        "1m": "1m", "2m": "1m", "3m": "3m", "4m": "3m", "5m": "5m", 
        "10m": "15m", "15m": "15m", "30m": "30m", "1h": "1h", 
        "2h": "2h", "3h": "2h", "4h": "4h", "5h": "4h", "6h": "6h"
    }
    if tf_input:
        INTERVAL = tf_map.get(tf_input, tf_input) # Use mapped or fallback to raw input if valid for binance
    else:
        INTERVAL = "15m"
    print(f"✅ Timeframe set to: {INTERVAL}")

    print("\nSelect Coin Category:")
    print("1. 🔥 Top Gainers (24H Volatility)")
    print("2. 🛡️ Safe Pairs (Recommended)")
    print("3. 🔎 Auto-Scan (Pick best from safe pairs)")
    print("4. 🕸️ Spider Web (Top 50 async scan)")
    cat_choice = input("Enter choice: ")

    data_api = SmartBackoffAPI()
    ai_engine = AI_Brain_Module()

    if cat_choice == "4":
        # STEP 13: Spider Web — scan top 50 by volume
        print("\n🕸️ Scanning TOP 50 symbols by volume...")
        top_symbols = await get_top_volume_symbols(50)
        print(f"Found {len(top_symbols)} symbols. Analyzing...")
        allocations = await rank_and_allocate(data_api, ai_engine, top_symbols, top_n=3)
        if not allocations:
            print("❌ No tradable symbols found.")
            await data_api.close()
            return
        print("\n📊 SPIDER WEB RANKINGS:\n")
        for i, (sym, sc, regime, weight) in enumerate(allocations, 1):
            emoji = {"TREND": "📈", "RANGE": "📊", "CHAOS": "💀", "UNCERTAIN": "❓"}.get(regime, "❓")
            print(f"{i}. {sym:12s} | Score: {sc:.1f}/100 | Regime: {emoji} {regime} | Allocation: {weight*100:.0f}%")
        choice = int(input("\nSelect coin number: "))
        symbol = allocations[choice - 1][0]
    elif cat_choice == "3":
        # STEP 6: Multi-coin scanner
        print("\n🔎 Scanning safe pairs for best opportunity...")
        safe_symbols = get_safe_pairs()
        scores = await scan_and_pick(data_api, ai_engine, safe_symbols)
        if not scores:
            print("❌ No tradable symbols found.")
            await data_api.close()
            return
        print("\n📊 SCAN RESULTS:\n")
        for i, (sym, sc, regime) in enumerate(scores, 1):
            emoji = {"TREND": "📈", "RANGE": "📊", "CHAOS": "💀", "UNCERTAIN": "❓"}.get(regime, "❓")
            print(f"{i}. {sym:12s} | Score: {sc:.1f}/100 | Regime: {emoji} {regime}")
        choice = int(input("\nSelect coin number: "))
        symbol = scores[choice - 1][0]
    elif cat_choice == "2":
        safe_list = show_safe_pairs()
        choice = int(input("\nSelect coin number: "))
        symbol = safe_list[choice - 1]
    else:
        gainers = await show_top_gainers()
        choice = int(input("\nSelect coin number: "))
        symbol = gainers[choice - 1]['symbol']

    print(f"\n✅ Selected: {symbol}")

    print("\nSelect Leverage:")
    print("1. 2x  (Safe)")
    print("2. 5x  (Balanced)")
    print("3. 10x (Standard)")
    print("--- EXPERT TIERS ---")
    print("4. 15x | 5. 20x | 6. 25x | 7. 30x | 8. 35x")
    print("9. Custom (Enter any value)")

    choice = input("Enter choice (or type leverage directly, e.g. 14): ").strip()
    leverage_map = {"1": 2, "2": 5, "3": 10, "4": 15, "5": 20, "6": 25, "7": 30, "8": 35}
    if choice in leverage_map:
        leverage = leverage_map[choice]
    elif choice == "9":
        try:
            leverage = int(input("Enter custom leverage (ex: 50): "))
        except:
            leverage = 2
    else:
        # Direct numeric input (e.g. user typed "14" for 14x)
        try:
            leverage = int(choice)
            if leverage < 1 or leverage > 125:
                print(f"⚠️ Invalid leverage {leverage}. Defaulting to 2x.")
                leverage = 2
        except ValueError:
            print("⚠️ Invalid input. Defaulting to 2x.")
            leverage = 2

    print(f"✅ Leverage set to: {leverage}x")

    print("\nSelect Training Mode:")
    print("1. ⚡ Fast Start (Load existing Brain if available)")
    print(f"2. 🧠 Force Retrain (Run Walk-Forward Grid Search on {INTERVAL} data)")
    train_choice = input("Enter choice: ")
    force_retrain = (train_choice == "2")
    if force_retrain:
        print("✅ Mode: Force Retrain")
    else:
        print("✅ Mode: Fast Start")

    klines = await data_api.get_historical_data(symbol, INTERVAL)
    valid, hist_df = ai_engine.train_model(klines, symbol, force_retrain=force_retrain)
    
    if not valid or hist_df is None or len(hist_df) == 0:
        print("❌ Model training failed / No usable data. Exiting...")
        await data_api.close()
        return

    while True:
        regime = classify_regime(hist_df)
        regime_emoji = {"TREND": "📈", "RANGE": "📊", "CHAOS": "💀", "UNCERTAIN": "❓"}.get(regime, "❓")
        print(f"\n📊 Market Regime: {regime_emoji} {regime}")

        if regime in ("CHAOS", "UNCERTAIN"):
            print(f"❌ Regime '{regime}' — not ideal. Waiting 30 seconds to re-check...")
            await asyncio.sleep(30)
            
            recent_klines = await data_api.fetch_klines(symbol, INTERVAL, limit=1000)
            if recent_klines:
                df_raw = pd.DataFrame(recent_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                            'ct', 'qav', 'not', 'tbb', 'tbq', 'ign'])
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
                df_raw.set_index('timestamp', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']: df_raw[col] = df_raw[col].astype(float)
                hist_df = ai_engine.feature_engineering(df_raw, rsi_length=ai_engine.best_rsi_len)
            continue

        print(f"✅ Regime '{regime}' is tradable!")
        user = input("\nType START to trade: ").upper()
        if user == "START":
            live_engine = LiveTradingEngine(ai_engine, hist_df, symbol, leverage)
            try:
                await live_engine.run()
            except asyncio.CancelledError:
                pass
            finally:
                await live_engine.client.close()
            break
        else:
            print("Exiting.")
            break

    await data_api.close()

if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(cli_flow())
    except KeyboardInterrupt:
        pass

