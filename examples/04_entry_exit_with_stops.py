"""Example 4 — Regime-Adaptive Strategy: Trend-Following ↔ Mean Reversion
=========================================================================
The market alternates between two states. This strategy detects which one
it's in and switches approach automatically:

  ADX(14) > 25  → TRENDING  → activate Trend-Following arm
  ADX(14) ≤ 25  → RANGING   → activate Mean-Reversion arm

The two arms are MUTUALLY EXCLUSIVE: filter_by(adx) and filter_by(~adx)
use the same ADX boundary, so only one arm can fire on any given bar.
At ADX = 25 exactly, the threshold is strict (>), so the ranging arm wins.

ARM 1 — Trend Following (ADX > 25)
  • Fibonacci EMA crosses  — globally watched MA pairs in crypto
  • MACD confirmation       — momentum agrees with price trend
  • OBV gate                — volume confirms the move is real
  Gated by filter_by(adx): entire arm goes silent in ranging markets

ARM 2 — Mean Reversion (ADX ≤ 25)
  • Bollinger Bands         — price statistically stretched from mean
  • RSI confirmation        — oscillator agrees price is at extreme
  • CCI confirmation        — H/L/C momentum also at extreme
  Gated by filter_by(~adx): entire arm goes silent in trending markets

Exit architecture — signal=, NOT entry=/exit=SAR:
  Uses Strategy(signal=combined) so position DIRECTLY follows the signal.
  When conditions disappear (BB/RSI/CCI go neutral, or EMA cross flips),
  signal → 0 and position closes immediately. No SAR state machine needed.
  SL/TP act as hard floors — positions are cut if price moves adversely.

  WHY NOT SAR: The entry=/exit=SAR state machine holds positions even after
  entry conditions vanish, waiting for SAR to flip. With slow SAR (accel=0.01)
  losing MR positions were held for days — the primary cause of -37% IS in v1.

Grid: 36 trend × 96 MR × 4 SL × 4 TP = 55,296 backtests (10× faster)

Timeframe recommendation — 4h NOT 1h:
  On 1h BTC, lagging indicators (EMA, MACD, ADX) generate hundreds of whipsaw
  trades because a 1.5% SL is hit by normal 2-3 bar noise. 4h bars filter out
  intraday noise: 75 trades per year (vs 236 on 1h) with 36% WR and positive
  OOS. The 2% SL and 8% TP align with 4h BTC volatility (~0.5-1%/bar).

Run:
    python examples/04_entry_exit_with_stops.py --demo
    python examples/04_entry_exit_with_stops.py -s BTC/USDT -t 4h \\
        --start 2023-01-01 --end 2023-12-31 --validate
"""

from siton.sdk import (
    ema_cross, macd_signal, obv,
    bollinger_bands, rsi, cci,
    adx,
    Strategy, backtest,
)

# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION — single ADX(14) boundary at 25
# ═══════════════════════════════════════════════════════════════════════════════
# adx(14, 25) → +1 where ADX>25 AND close>SMA14 (uptrend)
#                -1 where ADX>25 AND close<SMA14 (downtrend)
#                 0 where ADX≤25 (ranging/flat)
#
# ~adx(14, 25) → 1 where ADX≤25 (ranging), 0 where ADX>25 (trending)
#
# Threshold 25 is Wilder's original recommendation and the industry standard.
# NOT swept as a parameter — the regime boundary is a structural design choice,
# not something to overfit. Fixing it keeps the grid clean and interpretable.

trending = adx(periods=[14], thresholds=[25])    # gate: non-zero when trending
ranging  = ~adx(periods=[14], thresholds=[25])   # gate: 1 when flat/ranging


# ═══════════════════════════════════════════════════════════════════════════════
# ARM 1 — TREND FOLLOWING  (active only when ADX > 25)
# ═══════════════════════════════════════════════════════════════════════════════

# Fibonacci EMA pairs: all 9 combos are valid (8 < 55, 13 < 55, ... 21 < 144).
# Fibonacci numbers (8,13,21,55,89,144) are globally watched reference points.
trend_ema = ema_cross(
    fast=[8, 13, 21],
    slow=[55, 89, 144],
)

# MACD: 4 valid combos (8<21, 8<26, 12<21, 12<26). Signal period fixed at 9.
trend_macd = macd_signal(
    fast=[8, 12],
    slow=[21, 26],
    signal_period=[9],
)

# OBV: volume must confirm trend direction (fixed MA — not the primary variable).
trend_obv = obv(ma_periods=[20])

# ARM 1 composite: all three must agree direction, then ADX gate silences
# the arm entirely when the market is ranging.
# Grid keys: ema_fast[3] × ema_slow[3] × macd_fast[2] × macd_slow[2]
#            × macd_signal_period[1] × obv_ma_period[1]
#            × adx_period[1] × adx_threshold[1]  →  36 combos
trend_arm = (trend_ema & trend_macd & trend_obv).filter_by(trending)


# ═══════════════════════════════════════════════════════════════════════════════
# ARM 2 — MEAN REVERSION  (active only when ADX ≤ 25)
# ═══════════════════════════════════════════════════════════════════════════════

# Bollinger Bands: primary signal. Price below lower band = statistically cheap.
# 1.5σ fires often (frequent smaller extremes); 2.0σ fires rarely but cleanly.
mr_bb = bollinger_bands(
    windows=[14, 20, 26],
    num_std=[1.5, 2.0],
)

# RSI: confirms the extreme is real (standard 30/70 thresholds ± small range).
mr_rsi = rsi(
    periods=[9, 14],
    oversold=[28, 32],
    overbought=[68, 72],
)

# CCI: second confirmation using H/L/C average — different calculation method
# than RSI so it adds genuinely independent signal quality.
mr_cci = cci(
    periods=[14, 20],
    oversold=[-100],
    overbought=[100],
)

# ARM 2 composite: all three must agree direction, then ~ADX gate silences
# the arm entirely when the market is trending.
# Grid keys: bb_window[3] × bb_num_std[2] × rsi_period[2]
#            × rsi_oversold[2] × rsi_overbought[2]
#            × cci_period[2] × cci_oversold[1] × cci_overbought[1]
#            × adx_period_2[1] × adx_threshold_2[1]  →  96 combos
# (adx keys renamed _2 by _merge_flat to avoid collision with ARM1's adx keys)
mr_arm = (mr_bb & mr_rsi & mr_cci).filter_by(ranging)


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED ENTRY — mutually exclusive OR
# ═══════════════════════════════════════════════════════════════════════════════
# sig_or logic:
#   ADX > 25: trend_arm fires (+1 or -1), mr_arm = 0  → trend_arm wins
#   ADX ≤ 25: trend_arm = 0, mr_arm fires (+1 or -1) → mr_arm wins
#   Both 0:   entry = 0 (flat)
# The two arms cannot both be nonzero simultaneously — their ADX gates are
# complementary. No dead zone: at ADX = 25 exactly, ranging arm is active.
#
# Grid: 36 trend × 96 MR = 3,456 entry combos
entry = trend_arm | mr_arm


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY — signal= mode (position directly follows the combined signal)
# ═══════════════════════════════════════════════════════════════════════════════
# KEY CHANGE FROM v1: signal= instead of entry=/exit=SAR.
#
# In entry=/exit= mode the state machine HOLDS position until SAR flips —
# even after entry conditions vanish. With accel=0.01 this bled -37% in v1.
#
# In signal= mode position = signal value each bar:
#   MR trade:    closes naturally when BB/RSI/CCI return to neutral (snap done)
#   Trend trade: closes naturally when EMA cross or MACD reverses
#   SL/TP:       hard floors — cut immediately if price moves adversely
#
# Stop-loss 1–3%:   hard cut if trade goes wrong before signal clears.
# Take-profit 2–8%: MR targets 2-3% snap-back; trend targets 5-8% ride.
#
# Total: 3,456 signal combos × 4 SL × 4 TP = 55,296 backtests (9× faster)
STRATEGY = Strategy(
    "RegimeAdaptive_v2",
    signal=entry,
    stop_loss=[1.0, 1.5, 2.0, 3.0],
    take_profit=[2.0, 3.0, 5.0, 8.0],
    top=20,
    sort="sharpe_ratio",
)

if __name__ == "__main__":
    backtest(STRATEGY)
