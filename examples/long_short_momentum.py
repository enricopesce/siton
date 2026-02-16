#!/usr/bin/env python3
"""Long/Short Momentum — dual-direction strategy built with the Siton SDK.

Goes LONG when trend + momentum + volume all align bullish.
Goes SHORT when they all align bearish.
Stays FLAT when signals disagree.

How it works:
  - EMA crossover:  +1 when fast > slow (bullish), -1 when fast < slow (bearish)
  - MACD crossover:  +1 when MACD > signal,        -1 when MACD < signal
  - OBV crossover:   +1 when OBV > its MA,          -1 when OBV < its MA
  - The & operator requires ALL three to agree on direction:
      all +1 → open long    |    all -1 → open short    |    mixed → stay flat
  - ADX filter blocks trades in choppy, directionless markets
  - RSI filter blocks longs into overbought and shorts into oversold

Usage:
    siton examples/long_short_momentum.py --demo
    siton examples/long_short_momentum.py -s ETH/USDT -t 1h --start 2024-01-01 --end 2025-01-01
"""

from siton.sdk import *

# ── Layer 1: Trend direction (Fibonacci EMA periods) ─────────
trend = ema_cross(
    fast=[5, 8, 13],                          # 3 values
    slow=[21, 34, 55],                        # 3 values → 9 combos
)

# ── Layer 2: Momentum confirms direction ─────────────────────
momentum = macd_signal(
    fast=[8, 12],                             # 2 values
    slow=[21, 26],                            # 2 values
    signal_period=[7, 9],                     # 2 values → 8 combos
)

# ── Layer 3: Volume confirms direction ───────────────────────
volume = obv(ma_periods=[14, 21])             # 2 combos

# ── Combine: all three must agree (long OR short) ────────────
entry_raw = trend & momentum & volume         # 9 x 8 x 2 = 144

# ── Layer 4: Only trade in trending regimes ──────────────────
entry_trending = entry_raw.filter_by(
    adx(periods=[14, 21], thresholds=[20, 25])  # 2 x 2 = 4 combos
)                                             # 144 x 4 = 576

# ── Layer 5: Avoid exhausted moves ──────────────────────────
#   Blocks longs when overbought, blocks shorts when oversold
safe = entry_trending.filter_by(~rsi(
    periods=[7, 14],                          # 2 values
    oversold=[25, 30],                        # 2 values
    overbought=[70, 75],                      # 2 values → 8 combos
))                                            # 576 x 8 = 4,608

# ── Exit: SAR reversal closes the position ───────────────────
exit_signal = sar(
    accelerations=[0.02, 0.03],               # 2 values
    maximums=[0.1, 0.2],                      # 2 values → 4 combos
)

# ── Strategy ─────────────────────────────────────────────────
# Signal combos:  4,608 x 4 = 18,432
# PM grid:        3 SL x 3 TP x 2 trailing = 18
# Total:          18,432 x 18 = 331,776 backtests

STRATEGY = Strategy("LongShortMomentum",
    entry=safe,
    exit=exit_signal,
    stop_loss=[1, 2, 4],
    take_profit=[3, 6, 10],
    trailing_stop=[1, 2],
    fee=0.04,
    slippage=0.05,
)

if __name__ == "__main__":
    backtest(STRATEGY)
