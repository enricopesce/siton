"""Example 1 — EMA Crossover (Beginner)
======================================
The simplest possible strategy: go long when the fast EMA is above
the slow EMA, go short when it's below.

Concepts introduced:
  - ema_cross() signal constructor
  - Strategy() with a single signal=
  - backtest() CLI runner (--demo, --csv, -s/-t flags)

Run:
    python examples/01_ema_crossover.py --demo
    siton examples/01_ema_crossover.py --demo
    siton examples/01_ema_crossover.py -s BTC/USDT -t 1h --start 2024-01-01
"""

from siton.sdk import ema_cross, Strategy, backtest

# ── Signal ────────────────────────────────────────────────────────────────────
# ema_cross sweeps all (fast, slow) pairs from the cartesian product.
# Here: 2 × 2 = 4 combinations.
signal = ema_cross(fast=[8, 12], slow=[26, 50])

# ── Strategy ──────────────────────────────────────────────────────────────────
STRATEGY = Strategy(
    "EMA_Crossover",
    signal=signal,
    top=5,          # show top 5 results
    sort="sharpe_ratio",
    long_only=True
)

if __name__ == "__main__":
    backtest(STRATEGY)