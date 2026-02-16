#!/usr/bin/env python3
"""Trend Regime Sniper — EXPERIMENTAL CRYPTO GRID (no mercy)"""

from siton.sdk import *

# ===================================================================
# EXPERIMENTAL RANGES — huge but still crypto-native
# ===================================================================

# --- 1. EMA Cross (very short → medium-long) ---
trend_direction = ema_cross(
    fast=[3, 5, 8, 9, 12, 21],           # 6 values (3 is pure noise, but we want to test it)
    slow=[21, 34, 55, 89, 144, 200],     # 6 fib-style (classic crypto levels)
)  # 36 combos

# --- 2. ADX — catch trends earlier + stronger ones ---
trend = trend_direction.filter_by(
    adx(
        periods=[7, 14, 28],                 # 3 (classic + extremes)
        thresholds=[15, 20, 25, 30, 35],     # 5 (15 = very early, 35 = monster trend)
    )
)  # 15 combos

# --- 3. OBV confirmation — from fast to slow regime ---
confirmed = trend & obv(
    ma_periods=[10, 20, 50, 100],        # 4 (10 is aggressive, 100 catches macro shifts)
)

# --- 4. RSI filter — wide safety net ---
safe = confirmed.filter_by(~rsi(
    periods=[9, 14],                     # 2 (9 is crypto king)
    oversold=[20, 30],                   # 2
    overbought=[70, 80],                 # 2 → 8 combos total
))

# --- 5. SAR exit — very wide sensitivity ---
exit_signal = sar(
    accelerations=[0.01, 0.015, 0.02, 0.025, 0.03, 0.04],   # 6
    maximums=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40],          # 6 → 36 combos
)

# ===================================================================
# TOTALS
# ===================================================================
# Signal combinations : 36 × 15 × 4 × 8 × 36 = **622,080**
# Risk grid (9 SL × 10 TP)       = 90
# Total backtests                ≈ **56 million**

# This is the "we are experimenting" version. 
# On a decent machine (32–64 GB RAM) it still finishes in a few minutes.

STRATEGY = Strategy("TrendRegimeSniper_Experimental",
    entry=safe,
    exit=exit_signal,
    stop_loss=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],   # 9 values
    take_profit=[2, 3, 4, 5, 6, 8, 10, 12, 15, 20],             # 10 values
    fee=0.04,
    slippage=0.07,          # bumped a bit — crypto reality
)

if __name__ == "__main__":
    backtest(STRATEGY)