"""Example 3 — Trend + Momentum Confluence (Intermediate)
==========================================================
Require two independent momentum signals to agree before entering, then
gate everything behind ADX to trade only in trending markets.

Concepts introduced:
  - macd_signal() momentum signal
  - signal & other  — both must agree (long+long → long, else flat)
  - signal.filter_by(adx(...)) — only trade when trend is strong
  - Grid arithmetic: 2×2 EMA × 1×1×1 MACD × 2×2 ADX = 16 combinations

Run:
    python examples/03_trend_momentum_confluence.py --demo
"""

from siton.sdk import ema_cross, macd_signal, adx, Strategy, backtest

# ── Signals ───────────────────────────────────────────────────────────────────

# Trend: fast/slow EMA crossover
trend = ema_cross(fast=[8, 12], slow=[26, 50])

# Momentum: MACD line vs signal line
momentum = macd_signal(fast=[12], slow=[26], signal_period=[9])

# Confluence: both must point the same direction
confluence = trend & momentum

# Regime filter: only enter when ADX confirms a strong trend
strong_trend = adx(periods=[14, 20], thresholds=[20, 25])

# Final: confluence signal gated by trend strength
signal = confluence.filter_by(strong_trend)

# ── Strategy ──────────────────────────────────────────────────────────────────
STRATEGY = Strategy(
    "TrendMomentum",
    signal=signal,
    top=5,
    sort="sharpe_ratio",
)

if __name__ == "__main__":
    backtest(STRATEGY)