"""Example 6 — Risk-Per-Trade Position Sizing
============================================
By default Siton sizes every trade as a fixed fraction of equity:

    trade_value = equity × fraction          # e.g. 100% of equity

This deploys the same capital regardless of where the stop is placed.
A tight 1% stop and a wide 4% stop both commit 100% of equity, so the
dollar risk per trade varies wildly:

    stop=1%  → risk  1% of equity   (tight stop, small R loss)
    stop=4%  → risk  4% of equity   (wide stop, 4× bigger R loss)

The PROFESSIONAL standard is to fix the dollar risk first and
back-calculate position size from the stop distance:

    trade_value = equity × risk_per_trade / stop_distance

Every trade then risks exactly risk_per_trade of equity when stopped out.
fraction becomes a LEVERAGE CAP (max position as fraction of equity).

WORKED EXAMPLE
==============
  capital          = $10,000
  risk_per_trade   = 1%    (risk $100 per trade)
  stop_loss        = 2%    (2% below entry price)

  trade_value  = $10,000 × 1% / 2% = $5,000
  loss if hit  = $5,000 × 2%       = $100  = exactly 1% of capital ✓

  With stop_loss = 1%:
  trade_value  = $10,000 × 1% / 1% = $10,000
  (capped by fraction=1.0 → $10,000 — full equity, tight stop)

  With stop_loss = 4%:
  trade_value  = $10,000 × 1% / 4% = $2,500
  loss if hit  = $2,500 × 4%       = $100  = exactly 1% of capital ✓

The optimizer now sweeps stop width freely. Wider stops get smaller
positions; tighter stops get larger positions — but the P&L exposure per
stop-out is always the same dollar amount.

SIZING FORMULAS
===============
%-based stops (stop_loss / trailing_stop as % of price):

    if risk_per_trade > 0 and sl_pct > 0:
        trade_value = equity × risk_per_trade / sl_pct
        trade_value = min(trade_value, equity × fraction)   # cap leverage
    else:
        trade_value = equity × fraction

ATR-based stops (stop level = entry ± sl_mult × ATR):

    stop_dist   = sl_mult × ATR[entry_bar]
    trade_value = equity × risk_per_trade × fill / stop_dist
    trade_value = min(trade_value, equity × fraction)

STRATEGY
========
Same regime-adaptive signal as Example 4 (ADX gate, EMA+MACD trend arm,
BB+RSI+CCI mean-reversion arm). Three sizing variants are exported so you
can compare them side by side:

  STRATEGY_FIXED    — baseline: 100% equity per trade, no risk budget
  STRATEGY_RPT      — risk_per_trade=1%, variable position size per stop
  STRATEGY_RPT_ATR  — risk_per_trade=1%, ATR-based stops + ATR sizing

Grid per variant:
  3,456 signal combos × 4 SL × 4 TP = 55,296 backtests (same as Ex04)

Run:
    python examples/06_risk_per_trade_sizing.py --demo
    python examples/06_risk_per_trade_sizing.py -s BTC/USDT -t 4h \\
        --start 2023-01-01 --end 2023-12-31
"""

from siton.sdk import (
    ema_cross, macd_signal, obv,
    bollinger_bands, rsi, cci,
    adx,
    Strategy, backtest,
)

# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION — identical to Examples 04 & 05
# ═══════════════════════════════════════════════════════════════════════════════

trending = adx(periods=[14], thresholds=[25])
ranging  = ~adx(periods=[14], thresholds=[25])


# ═══════════════════════════════════════════════════════════════════════════════
# ARM 1 — TREND FOLLOWING  (active only when ADX > 25)
# ═══════════════════════════════════════════════════════════════════════════════

trend_ema = ema_cross(
    fast=[8, 13, 21],
    slow=[55, 89, 144],
)

trend_macd = macd_signal(
    fast=[8, 12],
    slow=[21, 26],
    signal_period=[9],
)

trend_obv = obv(ma_periods=[20])

trend_arm = (trend_ema & trend_macd & trend_obv).filter_by(trending)


# ═══════════════════════════════════════════════════════════════════════════════
# ARM 2 — MEAN REVERSION  (active only when ADX ≤ 25)
# ═══════════════════════════════════════════════════════════════════════════════

mr_bb = bollinger_bands(
    windows=[14, 20, 26],
    num_std=[1.5, 2.0],
)

mr_rsi = rsi(
    periods=[9, 14],
    oversold=[28, 32],
    overbought=[68, 72],
)

mr_cci = cci(
    periods=[14, 20],
    oversold=[-100],
    overbought=[100],
)

mr_arm = (mr_bb & mr_rsi & mr_cci).filter_by(ranging)


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED ENTRY — mutually exclusive by ADX gate
# ═══════════════════════════════════════════════════════════════════════════════

entry = trend_arm | mr_arm


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT 1 — Baseline: fixed fraction, no risk budget
# ═══════════════════════════════════════════════════════════════════════════════
# Every trade deploys fraction=1.0 (100%) of equity regardless of stop width.
# A 1% stop risks 1% of equity; a 4% stop risks 4% of equity.
# The P&L impact of stop-outs scales with stop width — unpredictable risk.

STRATEGY_FIXED = Strategy(
    "RegimeAdaptive_FixedFraction",
    signal=entry,
    stop_loss=[1.0, 1.5, 2.0, 4.0],
    take_profit=[2.0, 3.0, 5.0, 8.0],
    fraction=1.0,
    top=20,
    sort="sharpe_ratio",
)


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT 2 — Risk-per-trade with fixed % stops
# ═══════════════════════════════════════════════════════════════════════════════
# risk_per_trade=0.01 → each stop-out costs exactly 1% of equity.
# fraction=1.0  → hard cap: never deploy more than 100% (no leverage).
#
# How stop width affects position size:
#   stop=1%  → trade_value = equity × 1% / 1% = 100% equity  (capped at 100%)
#   stop=2%  → trade_value = equity × 1% / 2% =  50% equity
#   stop=4%  → trade_value = equity × 1% / 4% =  25% equity
#
# The optimizer can now find that a wider stop (fewer false exits) with a
# proportionally smaller position still beats a tight stop with full size.

STRATEGY_RPT = Strategy(
    "RegimeAdaptive_RiskPerTrade",
    signal=entry,
    stop_loss=[1.0, 1.5, 2.0, 4.0],
    take_profit=[2.0, 3.0, 5.0, 8.0],
    fraction=1.0,          # max position = 100% equity
    risk_per_trade=0.01,   # risk 1% of equity per stop-out
    top=20,
    sort="sharpe_ratio",
)


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT 3 — Risk-per-trade with ATR-based stops (most adaptive)
# ═══════════════════════════════════════════════════════════════════════════════
# Combines the volatility-adaptive stop placement of Example 5 with the
# consistent dollar-risk sizing of Variant 2.
#
# stop_dist = atr_sl_mult × ATR[entry_bar]          (absolute price distance)
# trade_value = equity × risk_per_trade × fill / stop_dist
#
# In a high-volatility regime (wide ATR):
#   → stop_dist is large → trade_value shrinks → position is small
# In a low-volatility regime (narrow ATR):
#   → stop_dist is small → trade_value grows  → position is large
#
# Both stop width AND position size adapt together, keeping dollar risk/trade
# constant across all market regimes and asset volatility profiles.

STRATEGY_RPT_ATR = Strategy(
    "RegimeAdaptive_RiskPerTradeATR",
    signal=entry,
    atr_period=14,
    atr_stop_mult=[1.0, 1.5, 2.0, 3.0],
    atr_tp_mult=[2.0, 3.0, 5.0, 8.0],
    fraction=1.0,          # max position = 100% equity
    risk_per_trade=0.01,   # risk 1% of equity per stop-out
    top=20,
    sort="sharpe_ratio",
)


# Default strategy exposed to the CLI
STRATEGY = STRATEGY_RPT


if __name__ == "__main__":
    backtest(STRATEGY)
