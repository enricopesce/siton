"""Example 5 — Regime-Adaptive Strategy with ATR-Normalized Stops
=================================================================
Same regime-adaptive framework as Example 4, but stop-loss and take-profit
levels scale with ATR(14) at entry time instead of using fixed percentages.

WHY ATR STOPS MATTER
====================
Fixed % stops fail across assets with different volatility profiles:

  Asset      4h ATR(14) / price    Ex04 SL=2%    Verdict
  ───────────────────────────────────────────────────────
  BTC/USDT      ~1-2 %              ≈ 1× ATR     OK
  ETH/USDT      ~1.5-3 %            ≈ 0.7-1.3×   borderline
  SOL/USDT      ~3-7 %              ≈ 0.3-0.7×   NOISE — hit constantly

  A 2% SL on SOL is smaller than one typical 4h bar. The stop fires on
  normal intraday noise before any trend can develop, generating a cascade
  of small losses that accumulate to -40% IS in Ex04.

HOW ATR STOPS WORK
==================
  Fixed:  stop_loss  = entry_price × (1 - sl_pct)
  ATR:    stop_loss  = entry_price - atr_sl_mult × ATR(14)[entry_bar]

  On BTC  with atr_sl_mult=2.0 → SL ≈ 2-4% below entry (matches Ex04)
  On SOL  with atr_sl_mult=2.0 → SL ≈ 6-14% below entry (appropriate!)
  On ETH  with atr_sl_mult=2.0 → SL ≈ 3-6% below entry (wider, fair)

  Same multiplier. Different asset. Correct width automatically.

ATR REGIME ADAPTATION
=====================
  Beyond cross-asset: ATR also adapts INTRA-asset across volatility regimes.
  BTC 2022 bear:  ATR(14) ≈ 3-5% → wider stops, less whipsaw
  BTC 2023 bull:  ATR(14) ≈ 1-2% → tighter stops, faster exits

STRATEGY ARCHITECTURE (same as Ex04)
=====================================
  Regime gate:   ADX(14) > 25  → TRENDING arm
                 ADX(14) ≤ 25  → RANGING  arm

  Trend arm:     EMA cross (Fibonacci pairs) & MACD & OBV volume confirm
  Ranging arm:   Bollinger Bands & RSI & CCI (all must agree direction)
  Combined:      trend_arm | mr_arm  (mutually exclusive by ADX gate)
  Mode:          signal= (position directly tracks signal, no SAR state)

Grid: 36 trend × 96 MR = 3,456 signal combos
       × 4 SL mults × 4 TP mults = 55,296 backtests (same as Ex04)

Run:
    python examples/05_entry_exit_with_atr_stops.py --demo
    python examples/05_entry_exit_with_atr_stops.py -s BTC/USDT -t 4h \\
        --start 2023-01-01 --end 2023-12-31 --validate
    python examples/05_entry_exit_with_atr_stops.py -s SOL/USDT -t 4h \\
        --start 2023-01-01 --end 2023-12-31 --validate
    python examples/05_entry_exit_with_atr_stops.py -s ETH/USDT -t 4h \\
        --start 2023-01-01 --end 2023-12-31 --validate
"""

from siton.sdk import (
    ema_cross, macd_signal, obv,
    bollinger_bands, rsi, cci,
    adx,
    Strategy, backtest,
)

# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
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
# COMBINED ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

entry = trend_arm | mr_arm


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY — ATR-normalized stops
# ═══════════════════════════════════════════════════════════════════════════════
# atr_period=14:    Wilder's ATR(14) — same period as the ADX regime gate.
#
# atr_sl_mult:      Stop-loss = entry_price - mult × ATR  (long)
#   1.0× ATR → tight; stops out on 1 average bar's range
#   2.0× ATR → standard; requires 2 bars of adverse move to exit
#   3.0× ATR → wide; only stops out on exceptional adverse moves
#
# atr_tp_mult:      Take-profit = entry_price + mult × ATR  (long)
#   Trend trades target 5-8×ATR (ride the move)
#   MR trades target 2-3×ATR (snap-back)
#   Combined grid lets the optimizer select the best R:R per regime

STRATEGY = Strategy(
    "RegimeAdaptiveATR_v1",
    signal=entry,
    atr_period=14,
    atr_stop_mult=[1.0, 1.5, 2.0, 3.0],
    atr_tp_mult=[2.0, 3.0, 5.0, 8.0],
    top=20,
    sort="sharpe_ratio",
)

if __name__ == "__main__":
    backtest(STRATEGY)
