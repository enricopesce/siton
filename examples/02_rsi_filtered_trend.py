"""Example 2 — RSI-Filtered EMA Trend (Easy)
============================================
Add a mean-reversion filter on top of the trend signal: only trade when
RSI is NOT in overbought/oversold territory.

Concepts introduced:
  - rsi() oscillator signal
  - ~signal  (where_flat) — inverts to "NOT in extreme zone"
  - signal.filter_by(other) — gate one signal with another
  - long_only=True — ignore short signals (crypto-friendly)
  - Grid: 2 × 2 × 2 × 2 × 2 = 32 combinations

Run:
    python examples/02_rsi_filtered_trend.py --demo
"""

from siton.sdk import ema_cross, rsi, Strategy, backtest

# ── Signals ───────────────────────────────────────────────────────────────────
trend = ema_cross(fast=[8, 12], slow=[26, 50])

# rsi() emits +1 when oversold (buy zone) and -1 when overbought (sell zone).
# ~rsi(...) flips it to a binary filter: 1 where RSI is NEUTRAL, 0 at extremes.
# Passing it to filter_by() keeps the trend signal only in calm RSI conditions.
rsi_neutral = ~rsi(periods=[14], oversold=[25, 30], overbought=[70, 75])

# Combined signal: trend fires only when RSI is not at an extreme.
signal = trend.filter_by(rsi_neutral)

# ── Strategy ──────────────────────────────────────────────────────────────────
STRATEGY = Strategy(
    "EMA_RSI_Filter",
    signal=signal,
    long_only=True,     # crypto spot: skip short signals
    top=5,
    sort="sharpe_ratio",
)

if __name__ == "__main__":
    backtest(STRATEGY)