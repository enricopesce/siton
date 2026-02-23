# Siton User Manual
### Building, Backtesting and Evaluating Trading Strategies

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
   - [4.6 Spot vs. Derivatives Markets](#46-spot-vs-derivatives-markets)
5. [Signal Constructors Reference](#5-signal-constructors-reference)
6. [Signal Composition Operators](#6-signal-composition-operators)
7. [The Strategy Class](#7-the-strategy-class)
   - [7.3 long_only Mode](#73-long_only-mode)
8. [Position Management](#8-position-management)
   - [8.4 Position Sizing Decision Tree](#84-position-sizing-decision-tree)
   - [8.5 Capital and Compounding](#85-capital-and-compounding)
9. [Data Sources](#9-data-sources)
10. [Running a Backtest](#10-running-a-backtest)
11. [Reading the Results](#11-reading-the-results)
12. [Purged K-Fold Cross-Validation](#12-purged-k-fold-cross-validation)
13. [Strategy Patterns Cookbook](#13-strategy-patterns-cookbook)
14. [Grid Sizing Guide](#14-grid-sizing-guide)
15. [Common Pitfalls](#15-common-pitfalls)
16. [Execution Costs and Realistic Backtesting](#16-execution-costs-and-realistic-backtesting)
17. [Performance Metrics Deep Dive](#17-performance-metrics-deep-dive)
18. [Multi-Strategy Runs and Capital Allocation](#18-multi-strategy-runs-and-capital-allocation)

---

## 1. Overview

Siton is a Python framework for backtesting systematic trading strategies on
historical cryptocurrency OHLCV data. Its two distinguishing features are:

- **Fluent SDK** — strategies are built from ~25 technical indicator constructors
  wired together with composable operators (`&`, `|`, `~`, `.filter_by()`, etc.)
  in a few readable lines of code.
- **Parallel engine** — a Numba-JIT engine evaluates every parameter combination
  in the cartesian product of your indicator grids simultaneously across all CPU
  cores, making thousands of backtests run in seconds.

The workflow is:

```
Strategy file → loads OHLCV data → SDK expands parameter grid
    → generates signal arrays → engine backtests all combos in parallel
    → ranks and prints results
```

---

## 2. Installation

The package is not yet published on PyPI. Install directly from the GitHub
repository.

### Step 1 — TA-Lib C library

The TA-Lib C library must be present on the system before installing the Python
package:

```bash
# Ubuntu / Debian
sudo apt-get install libta-lib-dev

# macOS
brew install ta-lib
```

### Step 2 — Clone and install

```bash
git clone https://github.com/enricopesce/siton.git
cd siton

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

`pip install -e .` installs siton in editable mode from the local clone,
including the `siton` CLI entry point. All dependencies (`polars`, `numpy≥2.0`,
`ccxt`, `ta-lib`, `numba`) are pulled in automatically.

**Required Python:** ≥ 3.10

---

## 3. Quick Start

Create a file `my_strategy.py`:

```python
from siton.sdk import ema_cross, rsi, Strategy, backtest

# 1. Build signals
trend = ema_cross(fast=[8, 12], slow=[26, 50])
filter_ = ~rsi(periods=[14], oversold=[30], overbought=[70])

# 2. Compose: trade the trend only when RSI is neutral
signal = trend.filter_by(filter_)

# 3. Package into a Strategy
STRATEGY = Strategy(
    "MyFirstStrategy",
    signal=signal,
    top=5,
    sort="sharpe_ratio",
)

# 4. Entry point
if __name__ == "__main__":
    backtest(STRATEGY)
```

Run it:

```bash
# Synthetic data (no internet needed)
python my_strategy.py --demo

# Real data from Binance
siton my_strategy.py -s BTC/USDT -t 1h --start 2024-01-01
```

---

## 4. Core Concepts

### 4.1 Ternary Signals

Every indicator produces an array with exactly three possible values:

| Value | Meaning |
|-------|---------|
| `+1.0` | Long (bullish) |
|  `0.0` | Flat (no opinion) |
| `-1.0` | Short (bearish) |

### 4.2 Parameter Grids

Every indicator constructor accepts lists of values instead of single scalars.
Siton automatically computes the **cartesian product** of all parameter lists and
backtests every combination.

```python
# This creates 2 × 2 = 4 combinations:
# (fast=8, slow=26), (fast=8, slow=50), (fast=12, slow=26), (fast=12, slow=50)
ema_cross(fast=[8, 12], slow=[26, 50])
```

When signals are composed, their grids are **merged flat** into a single
combined grid. The total number of backtests is the **product** of all leaf
indicator grid sizes.

### 4.3 Human-Readable Parameter Names

Each indicator prefixes its grid keys with its label so the output is
self-documenting. For example `ema_cross` produces keys `ema_fast` and
`ema_slow`; `adx` produces `adx_period` and `adx_threshold`. You never see
anonymous `param_0`, `param_1` columns.

### 4.4 Signal vs Entry/Exit Modes

**`signal=` mode** (recommended): The position directly follows the signal value
each bar. When the signal returns to 0, the position closes immediately.

**`entry=/exit=` mode**: A state machine opens a position when `entry` fires and
holds it until `exit` fires — even if the entry conditions have disappeared.
Use this only when you want an explicit, asymmetric exit indicator like SAR.

### 4.5 Execution Convention

Siton uses the professional standard:

```
Signal from close[i-1]  →  execute at open[i]  →  mark-to-market at close[i]
```

Intrabar stops (stop-loss, take-profit, trailing stop) use `high[i]` / `low[i]`
of the entry/holding bar to detect breaches.

### 4.6 Spot vs. Derivatives Markets

The two most common crypto trading contexts require different strategy designs.

#### Spot markets (buy / sell only)

You purchase the actual underlying asset. You can only **profit from price increases**
(long positions). When you want to exit, you sell what you hold. Shorting is not
available without an external lending mechanism.

**Examples:** Binance Spot, Coinbase, Kraken Spot.

**Siton setting:** always add `long_only=True` to your `Strategy`. This causes all
`-1` (short) signals to be treated as `0` (flat) before any position is opened.

```python
STRATEGY = Strategy("SpotOnly", signal=my_signal, long_only=True)
```

#### Derivatives markets (long + short)

A perpetual futures contract tracks the spot price but lets you take either side
with leverage. You can:

- **Go long (+1)** — profit from rising prices.
- **Go short (-1)** — profit from falling prices by borrowing the asset.

Siton backtests both directions by default (no flag needed).

**Examples:** Binance Futures (BTCUSDT perpetual), Bybit, OKX.

**Default Siton behavior** (`long_only=False`): both `+1` and `-1` signals are
executed. This is correct for derivatives accounts.

#### Side-by-side summary

| | Spot | Derivatives |
|---|---|---|
| Long (+1 signal) | Yes — buy the asset | Yes — open long contract |
| Short (-1 signal) | **No** — signal ignored | Yes — open short contract |
| Siton flag | `long_only=True` | (default, no flag needed) |
| Benchmark | Buy & Hold is fair comparison | Buy & Hold is only a partial comparison |
| Stops / TP | Same mechanics | Same mechanics |

#### How long_only affects your backtest

Setting `long_only=True` silently converts every `-1` signal to `0` before
processing. Practical effects:

- **Trade count roughly halves** — only bullish setups fire.
- **Returns in a bull market often improve** — short trades that would have lost
  money are removed.
- **Returns in a bear market often suffer** — profitable short trades are dropped.
- Always compare both modes before deciding which matches your deployment account.

```python
# Run both to compare before deploying
STRATEGY_SPOT   = Strategy("SpotVersion",  signal=signal, long_only=True,  top=5)
STRATEGY_FUTURES = Strategy("FuturesVersion", signal=signal, long_only=False, top=5)
STRATEGY = [STRATEGY_SPOT, STRATEGY_FUTURES]
```

---

## 5. Signal Constructors Reference

All constructors return a `Signal` object. Parameters that accept lists sweep
the full grid; single values produce a single combination.

### 5.1 Moving Average Crossovers

All crossover indicators: **long** when fast MA > slow MA, **short** when
fast MA < slow MA.

| Constructor | Description | Key params |
|-------------|-------------|------------|
| `ema_cross(fast, slow)` | Exponential MA crossover | `fast`, `slow` |
| `sma_cross(fast, slow)` | Simple MA crossover | `fast`, `slow` |
| `dema_cross(fast, slow)` | Double EMA crossover | `fast`, `slow` |
| `tema_cross(fast, slow)` | Triple EMA crossover | `fast`, `slow` |
| `wma_cross(fast, slow)` | Weighted MA crossover | `fast`, `slow` |
| `kama_cross(fast, slow)` | Kaufman Adaptive MA | `fast`, `slow` |
| `trima_cross(fast, slow)` | Triangular MA crossover | `fast`, `slow` |

All crossover constructors silently return flat (`0`) for any combo where
`fast >= slow`.

```python
from siton.sdk import ema_cross, sma_cross

trend = ema_cross(fast=[8, 12, 21], slow=[50, 100, 200])
# → 3 × 3 = 9 combinations (only valid pairs where fast < slow)
```

### 5.2 Oscillators (Mean Reversion)

All oscillator indicators: **long** when indicator < `oversold`, **short** when
indicator > `overbought`, **flat** in between.

| Constructor | Description | Default thresholds |
|-------------|-------------|-------------------|
| `rsi(periods, oversold, overbought)` | Relative Strength Index | 30 / 70 |
| `cmo(periods, oversold, overbought)` | Chande Momentum Oscillator | -50 / 50 |
| `cci(periods, oversold, overbought)` | Commodity Channel Index | -100 / 100 |
| `willr(periods, oversold, overbought)` | Williams %R | -80 / -20 |
| `mfi(periods, oversold, overbought)` | Money Flow Index (uses volume) | 20 / 80 |
| `stoch_rsi(periods, fastk, fastd, oversold, overbought)` | Stochastic RSI | 20 / 80 |
| `ultimate_osc(p1, p2, p3, oversold, overbought)` | Ultimate Oscillator | 30 / 70 |

```python
from siton.sdk import rsi, cci

# RSI: long when oversold (<30), short when overbought (>70)
momentum = rsi(periods=[9, 14], oversold=[25, 30], overbought=[70, 75])

# CCI: long when extremely oversold (<-100)
extreme = cci(periods=[14, 20], oversold=[-100], overbought=[100])
```

### 5.3 Momentum (Zero-Line Crossovers)

**Long** when indicator > 0, **short** when indicator < 0.

| Constructor | Description | Key params |
|-------------|-------------|------------|
| `macd_signal(fast, slow, signal_period)` | MACD line vs signal line | `fast`, `slow`, `signal_period` |
| `apo(fast, slow)` | Absolute Price Oscillator | `fast`, `slow` |
| `ppo(fast, slow)` | Percentage Price Oscillator | `fast`, `slow` |
| `mom(periods)` | Price Momentum (price vs N bars ago) | `period` |
| `roc(periods)` | Rate of Change | `period` |
| `trix(periods)` | Triple smoothed EMA ROC | `period` |

```python
from siton.sdk import macd_signal

# MACD: 4 valid fast/slow pairs × 1 signal period = 4 combos
momentum = macd_signal(fast=[8, 12], slow=[21, 26], signal_period=[9])
```

### 5.4 Trend Strength

These indicators fire only when a trend is detected above a threshold.

| Constructor | Description | Output logic |
|-------------|-------------|-------------|
| `adx(periods, thresholds)` | Average Directional Index | `+1` when ADX > threshold AND price > SMA (uptrend); `-1` when downtrend; `0` when ranging |
| `adxr(periods, thresholds)` | Smoothed ADX | same as `adx` |
| `dx(periods, thresholds)` | Directional Movement Index | same as `adx` |
| `di_cross(periods)` | +DI vs −DI crossover | `+1` when +DI > −DI |
| `dm_cross(periods)` | +DM vs −DM crossover | `+1` when +DM > −DM |

```python
from siton.sdk import adx

# ADX gate: non-zero only when trend is strong (ADX > 25)
trending = adx(periods=[14], thresholds=[25])

# Ranging gate: non-zero when market is flat (ADX ≤ 25)
ranging = ~adx(periods=[14], thresholds=[25])
```

### 5.5 Volume

| Constructor | Description | Output logic |
|-------------|-------------|-------------|
| `obv(ma_periods)` | On-Balance Volume vs SMA(OBV) | `+1` when OBV > SMA(OBV) |
| `ad(ma_periods)` | Accumulation/Distribution vs SMA | `+1` when AD > SMA(AD) |
| `adosc(fast, slow)` | ADOSC zero-line cross | `+1` when ADOSC > 0 |

```python
from siton.sdk import obv

# Volume confirmation: positive money flow
vol_confirm = obv(ma_periods=[20, 50])
```

### 5.6 Volatility

| Constructor | Description | Output logic |
|-------------|-------------|-------------|
| `bollinger_bands(windows, num_std)` | Bollinger Band squeeze | `+1` when price < lower band; `-1` when price > upper band |
| `atr_breakout(periods, lookbacks)` | ATR expanding + direction | `+1` when ATR > SMA(ATR) and price rising |
| `natr_breakout(periods, lookbacks)` | NATR expanding + direction | same as `atr_breakout` |
| `channel_breakout(periods)` | Donchian channel breakout | `+1` when close > previous N-bar high |

```python
from siton.sdk import bollinger_bands

# Bollinger: +1 below lower band, -1 above upper band
bb = bollinger_bands(windows=[14, 20], num_std=[1.5, 2.0])
```

### 5.7 Exit Signals (SAR family)

| Constructor | Description | Key params |
|-------------|-------------|------------|
| `sar(accelerations, maximums)` | Parabolic SAR | `acceleration`, `maximum` |
| `sar_ext(accel_inits, accelerations, maximums)` | Extended SAR with symmetric long/short params | `accel_init`, `acceleration`, `maximum` |

SAR signals are most useful in `entry=/exit=` mode where the SAR flip acts as
a trailing exit.

```python
from siton.sdk import sar

exit_signal = sar(accelerations=[0.02, 0.03], maximums=[0.2])
```

### 5.8 Candlestick Patterns

```python
from siton.sdk import candlestick_pattern

# Any TA-Lib CDL function name
engulfing = candlestick_pattern("CDLENGULFING")
hammer    = candlestick_pattern("CDLHAMMER")
doji      = candlestick_pattern("CDLDOJI")

# Patterns that accept a penetration parameter
piercing  = candlestick_pattern("CDLPIERCING", penetration_values=[0.3, 0.5])
```

The output maps TA-Lib's `100` → `+1.0`, `-100` → `-1.0`, `0` → `0.0`.

### 5.9 Custom Signals

For any indicator not natively available, use `custom()`:

```python
from siton.sdk import custom
import numpy as np

def my_signal(data, period, threshold):
    close = data["close"]
    # ... compute your indicator ...
    return np.where(indicator > threshold, 1.0, np.where(indicator < -threshold, -1.0, 0.0))

sig = custom(my_signal, label="mysig", period=[10, 20], threshold=[0.5, 1.0])
```

`data` is a dict with NumPy float64 arrays: `open`, `high`, `low`, `close`, `volume`.
The function must return a `float64` array with values in `{-1.0, 0.0, +1.0}`.

---

## 6. Signal Composition Operators

This is the heart of the SDK. Once you have individual indicator signals, you
wire them together with operators to express your trading logic in plain,
readable code. Siton provides eight operators, each capturing a distinct
real-world concept: confirmation, fallback, negation, gating, directional
agreement, temporal confirmation, voting, and inversion.

### What "composition" means

Every indicator returns a ternary array — a value of `+1`, `0`, or `-1` for
every bar in your dataset. Composition takes two (or more) of those arrays and
combines them bar-by-bar to produce a new array. The result is itself a `Signal`
object, so you can keep chaining operators indefinitely.

```
bar 1   bar 2   bar 3   bar 4   bar 5  ...
  +1      +1      0       -1      +1       ← EMA crossover
  +1       0      0       -1       0       ← MACD signal
  +1       0      0       -1       0       ← EMA & MACD  (both must agree)
```

All compositions also **merge the parameter grids**. If EMA has 4 combos and
MACD has 2, the composed signal has 4 × 2 = 8 combos. Every combination is
backtested independently.

---

### 6.1 `&` — AND: Both Must Agree

**What it means:** Only trade when two independent indicators point in the same
direction. If either one is silent or they disagree, stay flat.

**When to use it:** Adding a confirmation layer. You have a trend signal but
you only want to act when a momentum indicator also agrees — reducing false
entries on choppy markets.

**Bar-by-bar logic:**

| Signal A | Signal B | Result |
|----------|----------|--------|
| +1 (long)  | +1 (long)  | +1 (long) |
| -1 (short) | -1 (short) | -1 (short) |
| +1 (long)  | -1 (short) | 0 (flat — conflict) |
| +1 (long)  |  0 (flat)  | 0 (flat — A alone is not enough) |
| -1 (short) |  0 (flat)  | 0 (flat) |
|  0 (flat)  |  0 (flat)  | 0 (flat) |

**Example — require trend and momentum to agree:**

```python
from siton.sdk import ema_cross, macd_signal

trend    = ema_cross(fast=[8, 12], slow=[26, 50])   # 4 combos
momentum = macd_signal(fast=[12], slow=[26], signal_period=[9])  # 1 combo

# Only go long when BOTH EMA is bullish AND MACD is above its signal line
confluence = trend & momentum   # 4 × 1 = 4 backtests
```

**Chaining multiple `&` operators:**

```python
# All three must agree — each added & halves the number of signals that pass
signal = ema_cross(fast=[12], slow=[50]) & macd_signal(fast=[12], slow=[26], signal_period=[9]) & obv(ma_periods=[20])
```

> **Tip for beginners:** `&` makes your strategy more selective. Fewer trades
> will fire, but each one has more evidence behind it. If your strategy is
> generating too many bad trades, add another `&` condition.

---

### 6.2 `|` — OR: First Non-Flat Wins

**What it means:** Use signal A if it has an opinion; otherwise fall back to
signal B. If both are active but disagree, stay flat (treat the conflict as
uncertainty).

**When to use it:** Building **mutually exclusive regime arms**. If you have a
trend-following strategy and a mean-reversion strategy, and you want exactly one
of them active at any time, gate each arm with an ADX filter and combine with
`|`.

**Bar-by-bar logic:**

| Signal A | Signal B | Result | Reason |
|----------|----------|--------|--------|
| +1 (long)  |  0 (flat)  | +1 | A has opinion, B is silent |
|  0 (flat)  | +1 (long)  | +1 | A is silent, B takes over |
| -1 (short) |  0 (flat)  | -1 | A has opinion, B is silent |
|  0 (flat)  | -1 (short) | -1 | A is silent, B takes over |
| +1 (long)  | -1 (short) |  0 | Both active but conflict — stay flat |
| -1 (short) | +1 (long)  |  0 | Both active but conflict — stay flat |
|  0 (flat)  |  0 (flat)  |  0 | Neither has opinion |

**Example — two regime arms that can never both fire:**

```python
from siton.sdk import ema_cross, macd_signal, bollinger_bands, rsi, adx

# Gate each arm with complementary ADX filters
trending = adx(periods=[14], thresholds=[25])   # non-zero when ADX > 25
ranging  = ~adx(periods=[14], thresholds=[25])  # non-zero when ADX ≤ 25

trend_arm = (ema_cross(fast=[8, 13], slow=[55, 89]) & macd_signal(fast=[12], slow=[26], signal_period=[9])).filter_by(trending)
mr_arm    = (bollinger_bands(windows=[20], num_std=[2.0]) & rsi(periods=[14], oversold=[30], overbought=[70])).filter_by(ranging)

# When ADX > 25: only trend_arm can fire (mr_arm is silenced by its gate)
# When ADX ≤ 25: only mr_arm can fire (trend_arm is silenced by its gate)
# The two arms are structurally mutually exclusive
combined = trend_arm | mr_arm
```

> **Tip for beginners:** Think of `|` as a selector switch. `trend_arm | mr_arm`
> reads as "use the trend strategy, or if it has nothing to say, use the
> mean-reversion strategy instead". When paired with proper ADX gating, exactly
> one arm fires on every bar.

---

### 6.3 `~` — Invert to Binary Gate

**What it means:** Produce a binary "permission" signal that is `+1` wherever
the original signal is **flat (0)**, and `0` everywhere else.

**Critical distinction — this is NOT a long/short flip:**

| Original | `~` result | Meaning |
|----------|-----------|---------|
| +1 (long)  | 0 | "not allowed" (indicator is active, so gate is closed) |
| -1 (short) | 0 | "not allowed" (indicator is active, so gate is closed) |
|  0 (flat)  | 1 | "allowed" (indicator is quiet, so gate is open) |

`~` answers the question: **"Is this indicator currently calm or neutral?"**

It is designed to be used directly inside `.filter_by()`. The idiomatic pattern
is `signal.filter_by(~oscillator(...))`, which reads as: "trade this signal only
when the oscillator is NOT at an extreme".

**Example — avoid entering during overbought/oversold conditions:**

```python
from siton.sdk import ema_cross, rsi

trend = ema_cross(fast=[8, 12], slow=[26, 50])

# rsi() fires +1 when oversold (< 30), -1 when overbought (> 70), 0 when neutral
# ~rsi() fires +1 when NEUTRAL, 0 when at extremes
rsi_calm = ~rsi(periods=[14], oversold=[30], overbought=[70])

# Only take trend trades when RSI is in the neutral zone
safe_trend = trend.filter_by(rsi_calm)
```

**Example — range gate (inverse of ADX):**

```python
from siton.sdk import adx

# adx() fires when the market IS trending
# ~adx() fires when the market is NOT trending (ranging / flat)
ranging = ~adx(periods=[14], thresholds=[25])

# Mean-reversion strategy only active in ranging markets
mr_signal = bollinger_bands(windows=[20], num_std=[2.0]).filter_by(ranging)
```

> **Common mistake:** `~signal` does NOT mean "go short when you would go long".
> For that, use `.negate()`. The `~` operator is purely a gate: it tells other
> signals whether they are allowed to fire.

**If you want to use `~` for long/short inversion (rare):**

```python
# This would flip a trend signal into a contrarian signal
contrarian = trend.negate()   # use .negate(), NOT ~
```

---

### 6.4 `.filter_by(other)` — Activity Gate

**What it means:** The primary signal passes through unchanged wherever `other`
is **non-zero**. Where `other` is flat (`0`), the primary signal is suppressed
to `0` regardless of what it says.

**Key point:** `.filter_by()` does **not** check direction. As long as `other`
has any opinion (long or short), the primary signal is allowed through.

| Primary signal | Gate (other) | Result |
|----------------|--------------|--------|
| +1 | +1 | +1 (primary passes) |
| +1 | -1 | +1 (primary passes — gate is non-zero regardless of direction) |
| +1 |  0 | 0  (primary suppressed — gate is flat) |
| -1 | +1 | -1 (primary passes) |
| -1 | -1 | -1 (primary passes) |
| -1 |  0 | 0  (primary suppressed) |
|  0 | anything | 0  (primary was already flat) |

**When to use it:** Regime filters. You want a trend signal to be active only
when the market is in trend mode, or a mean-reversion signal to be active only
when the market is ranging.

```python
from siton.sdk import ema_cross, adx

# ADX fires (+1 or -1) when a trend is strong enough
# ema_cross is allowed through only during those bars
gated = ema_cross(fast=[8, 12], slow=[26, 50]).filter_by(
    adx(periods=[14], thresholds=[20, 25])
)
```

**Chaining multiple `.filter_by()` calls:**

```python
# Three independent gates: all must be non-zero for the signal to pass
signal = (
    ema_cross(fast=[12], slow=[50])
    .filter_by(adx(periods=[14], thresholds=[25]))   # trending market
    .filter_by(~rsi(periods=[14], oversold=[30], overbought=[70]))  # RSI not extreme
    .filter_by(obv(ma_periods=[20]))                 # volume confirms
)
```

> **`.filter_by()` vs `&`:** Both restrict when trades fire. The difference is
> subtle but important. `a & b` requires A and B to agree on **direction** —
> if A is long and B is short, the result is flat. `a.filter_by(b)` only checks
> whether B is active at all — it lets A's direction through regardless of B's
> direction. For regime filters (ADX, volume), use `.filter_by()`. For
> directional confirmation (MACD confirming EMA), use `&`.

---

### 6.5 `.agree(other)` — Directional Gate

**What it means:** Like `.filter_by()`, but stricter: the primary signal only
passes where `other` is non-zero **AND pointing the same direction**.

| Primary | Other | `.filter_by()` | `.agree()` |
|---------|-------|---------------|-----------|
| +1 | +1 | +1 | +1 |
| +1 | -1 | +1 | 0  ← suppressed (directions differ) |
| +1 |  0 |  0 |  0 |
| -1 | -1 | -1 | -1 |
| -1 | +1 | -1 |  0 ← suppressed (directions differ) |
| -1 |  0 |  0 |  0 |

**When to use it:** When you want a confirmation indicator to not only be active
but to specifically agree with the direction you are trading. This is more
conservative than `.filter_by()`.

```python
from siton.sdk import ema_cross, di_cross

# +DI vs -DI cross: +1 when buyers dominate, -1 when sellers dominate
# Only allow EMA signals where DI cross agrees on direction
directional = ema_cross(fast=[8, 12], slow=[26, 50]).agree(
    di_cross(periods=[14])
)
```

**Practical difference — when to choose `.agree()` over `.filter_by()`:**

```python
trend = ema_cross(fast=[12], slow=[50])
volume = obv(ma_periods=[20])

# .filter_by(volume): goes long when EMA is bullish AND OBV is active (any direction)
# Allows long EMA + bearish OBV → might be wrong
trend.filter_by(volume)

# .agree(volume): goes long only when EMA is bullish AND OBV is also bullish
# More conservative — both must point the same way
trend.agree(volume)
```

> **Tip:** Use `.agree()` when the secondary indicator is genuinely directional
> (volume flow, DI cross). Use `.filter_by()` when the secondary indicator is a
> regime marker (ADX, ATR expansion) whose direction does not map to trade
> direction.

---

### 6.6 `.confirm(other, lookbacks)` — Lookback Confirmation

**What it means:** The primary signal is accepted only if `other` fired in the
**same direction** within the last `N` bars. This introduces a tolerance window:
the confirmation does not have to happen on the exact same bar — it just needs
to have happened recently.

`lookbacks` is a tuple of values to sweep as part of the backtest grid.

**How it works bar by bar:**

```
Bar 10:  MACD = +1   (MACD fires long)
Bar 11:  MACD =  0
Bar 12:  MACD =  0
Bar 13:  EMA  = +1   (EMA fires long)
         lookback=5 → checks bars 8-12 for MACD long → bar 10 qualifies ✓ → EMA accepted
Bar 14:  EMA  = +1
         lookback=3 → checks bars 11-13 → MACD was 0, 0, 0 → EMA rejected ✗
```

```python
from siton.sdk import ema_cross, macd_signal

# EMA crossover is accepted only if MACD was bullish at some point in the
# last 3, 5, or 8 bars (each lookback value is a separate backtest combo)
confirmed = ema_cross(fast=[8, 12], slow=[26, 50]).confirm(
    macd_signal(fast=[12], slow=[26], signal_period=[9]),
    lookbacks=(3, 5, 8),
)
# Grid: 4 EMA combos × 1 MACD combo × 3 lookbacks = 12 backtests
```

**When to use it:** When two indicators rarely fire on exactly the same bar but
you still want them to be "recently in sync". A crossover system and a momentum
oscillator often lead/lag each other by a few bars.

> **Tip:** Smaller lookback values (3–5) enforce stricter recency. Larger values
> (8–13) allow more lag between the two indicators but increase the risk of
> stale confirmations. Sweeping lookbacks lets the optimizer find the right
> tolerance.

---

### 6.7 `Signal.majority(*signals)` — Majority Vote

**What it means:** Supply three or more signals and take a trade only when
**more than half** vote in the same direction.

This is a robust alternative to chaining multiple `&` operators. With `&`, a
single neutral indicator blocks every trade. With `majority()`, one neutral
indicator is outvoted by the others.

**Voting rules (example with 3 signals):**

| A  | B  | C  | `&` chain | `majority()` |
|----|----|----|-----------|--------------|
| +1 | +1 | +1 | +1 | +1 |
| +1 | +1 |  0 |  0 | +1 ← majority wins |
| +1 |  0 |  0 |  0 |  0 (no majority) |
| +1 | -1 |  0 |  0 |  0 (tie) |
| +1 | +1 | -1 |  0 | +1 ← 2 vs 1 |

```python
from siton.sdk import ema_cross, rsi, macd_signal, Signal, Strategy, backtest

# At least 2 of 3 indicators must agree
signal = Signal.majority(
    ema_cross(fast=[12], slow=[50]),
    rsi(periods=[14], oversold=[30], overbought=[70]),
    macd_signal(fast=[12], slow=[26], signal_period=[9]),
)

STRATEGY = Strategy("MajorityVote", signal=signal, top=10)
```

**With 5 signals (3 out of 5 required):**

```python
signal = Signal.majority(
    ema_cross(fast=[12], slow=[50]),
    macd_signal(fast=[12], slow=[26], signal_period=[9]),
    rsi(periods=[14], oversold=[30], overbought=[70]),
    obv(ma_periods=[20]),
    adx(periods=[14], thresholds=[25]),
)
# 3 out of 5 must agree on direction
```

> **`&` vs `majority()` — when to choose which:**
>
> - Use `&` when every indicator in the chain is genuinely required as a
>   necessary condition (e.g., trend + regime filter + volume).
> - Use `majority()` when you have several loosely correlated indicators and
>   want a democratic consensus rather than a strict requirement — it tolerates
>   one dissenter.

---

### 6.8 `.negate()` — Algebraic Inversion (Long ↔ Short Swap)

**What it means:** Flips the direction of every bar: long becomes short, short
becomes long, flat stays flat.

| Original | `.negate()` |
|----------|-------------|
| +1 (long)  | -1 (short) |
| -1 (short) | +1 (long) |
|  0 (flat)  |  0 (flat) |

**When to use it:** Building contrarian strategies, or reversing an exit signal
for use as an entry.

```python
from siton.sdk import ema_cross, rsi

# Standard trend follower
trend = ema_cross(fast=[12], slow=[50])

# Contrarian: short when trend says long, long when trend says short
contrarian = trend.negate()

# RSI mean-reversion signal: +1 when oversold (buy dip)
# Negate it to get: +1 when overbought (momentum continuation) — rare use case
overbought_momentum = rsi(periods=[14], oversold=[30], overbought=[70]).negate()
```

> **`.negate()` vs `~`:** These are completely different.
>
> - `~signal` — "where is this indicator NOT active?" → binary gate, always outputs 0 or +1
> - `signal.negate()` — "flip the trade direction" → always outputs 0, +1, or -1

---

### 6.9 Chaining Operators — Building Complex Logic

All operators return a `Signal` object, so they chain freely. Reading left to
right, each operator applies to the result of everything before it.

**Example — full strategy built step by step:**

```python
from siton.sdk import ema_cross, macd_signal, adx, rsi, obv

# Step 1: Base trend signal
trend = ema_cross(fast=[8, 12, 21], slow=[55, 89, 144])

# Step 2: Require MACD to agree on direction
trend_confirmed = trend & macd_signal(fast=[12], slow=[26], signal_period=[9])

# Step 3: Only trade when ADX says the market is in trend mode
trend_in_regime = trend_confirmed.filter_by(adx(periods=[14], thresholds=[25]))

# Step 4: Block entries when RSI is at an extreme (overbought/oversold)
trend_safe = trend_in_regime.filter_by(~rsi(periods=[14], oversold=[30], overbought=[70]))

# Step 5: Require volume to confirm
final_signal = trend_safe.filter_by(obv(ma_periods=[20]))
```

This reads naturally: "Take a trend trade when EMA and MACD agree, the market is
trending (ADX), RSI is not at an extreme, and volume is confirming."

**Equivalent one-liner (less readable but identical result):**

```python
signal = (
    ema_cross(fast=[8, 12, 21], slow=[55, 89, 144])
    & macd_signal(fast=[12], slow=[26], signal_period=[9])
).filter_by(adx(periods=[14], thresholds=[25]))
 .filter_by(~rsi(periods=[14], oversold=[30], overbought=[70]))
 .filter_by(obv(ma_periods=[20]))
```

---

### 6.10 Quick-Reference: When to Use Which Operator

| You want to... | Use |
|---------------|-----|
| Require two indicators to agree on direction | `a & b` |
| Switch between two strategy arms | `a \| b` |
| Block trades when an oscillator is at extreme | `.filter_by(~oscillator(...))` |
| Allow trades only during a specific regime | `.filter_by(regime_indicator)` |
| Require a second indicator to also agree directionally | `.agree(other)` |
| Accept a signal if another fired recently (within N bars) | `.confirm(other, lookbacks)` |
| Take a trade when most of several indicators agree | `Signal.majority(a, b, c, ...)` |
| Build a contrarian / opposite strategy | `.negate()` |

### 6.11 Full Operator Truth Table

```
a    b    a & b    a | b    ~a    a.filter_by(b)    a.agree(b)
+1   +1     +1      +1       0         +1                +1
+1   -1      0       0       0         +1                 0
+1    0      0      +1       0          0                 0
-1   +1      0       0       1         -1                 0
-1   -1     -1      -1       1         -1                -1
-1    0      0      -1       1          0                 0
 0   +1      0      +1       1          0                 0
 0   -1      0      -1       1          0                 0
 0    0      0       0       1          0                 0
```

---

## 7. The Strategy Class

`Strategy` packages a signal (or entry/exit pair) with execution parameters.

```python
Strategy(
    name,               # string identifier shown in output
    *,
    signal=None,        # Signal object — position directly tracks signal
    entry=None,         # Signal object — entry state machine
    exit=None,          #   (must be paired with entry=)
    # Position management (see Section 8)
    stop_loss=None,
    take_profit=None,
    trailing_stop=None,
    atr_period=14,
    atr_stop_mult=None,
    atr_tp_mult=None,
    atr_trail_mult=None,
    # Sizing
    fraction=1.0,       # position size as fraction of equity (1.0 = 100%)
    risk_per_trade=None,# fixed dollar risk per stop-out (e.g. 0.01 = 1%)
    capital=10000.0,    # starting capital
    # Execution costs
    fee=None,           # taker fee fraction (e.g. 0.001 = 0.1%)
    slippage=0.05,      # slippage fraction (0.05 = 0.05%)
    # Output control
    top=10,             # number of results to display
    sort="sharpe_ratio",# metric to rank by
    long_only=False,    # ignore short signals (useful for spot crypto)
    risk_free_rate=0.0, # annualized risk-free rate for Sharpe/Sortino
    # Purged K-fold cross-validation (see Section 12)
    cv=False,
    cv_folds=5,
    purge_bars=50,
    n_splits=5,         # stability check windows within last fold
)
```

### 7.1 signal= vs entry=/exit=

```python
# SIGNAL MODE: position = signal value each bar
# Closes automatically when signal returns to 0
STRATEGY = Strategy("TrendFollower", signal=ema_cross(fast=[12], slow=[50]))

# ENTRY/EXIT MODE: state machine — holds until exit fires
STRATEGY = Strategy(
    "TrendWithSAR",
    entry=ema_cross(fast=[12], slow=[50]),
    exit=sar(accelerations=[0.02], maximums=[0.2]),
)
```

### 7.2 Sort Metrics

The `sort=` parameter controls how results are ranked:

| Value | Metric |
|-------|--------|
| `"sharpe_ratio"` | Annualized Sharpe ratio (default) |
| `"total_return_pct"` | Total percentage return |
| `"max_drawdown_pct"` | Maximum drawdown (ascending = less drawdown first) |
| `"win_rate_pct"` | Percentage of winning trades |
| `"profit_factor"` | Gross profit / gross loss |
| `"num_trades"` | Number of trades executed |

### 7.3 long_only Mode

For spot cryptocurrency trading where shorting is not available:

```python
STRATEGY = Strategy("SpotBTC", signal=my_signal, long_only=True)
```

Short signals (`-1`) are suppressed; only long positions are taken.

**What happens to each signal value:**

| Signal | `long_only=False` (default) | `long_only=True` (spot) |
|--------|-----------------------------|--------------------------|
| `+1` (long) | Open long position | Open long position |
| `-1` (short) | Open short position | **Treated as flat (0)** |
| `0` (flat) | No position | No position |

**Designing for spot markets:** when you know your strategy will run on a spot
account, bias your indicators toward capturing bullish moves rather than
symmetric long/short setups:

- **Prefer oscillators in oversold mode** — e.g. `rsi()` fires `+1` when
  oversold (buy the dip); that long signal is useful on spot.
- **Use `~rsi()` as a filter** — avoids entering longs when overbought rather
  than entering shorts.
- **Mean-reversion strategies suit spot well** — they go long on dips and exit
  on recovery; no shorts needed.
- **Trend strategies work too** — just accept that the short leg will be flat,
  meaning you sit in cash during downtrends rather than profiting from them.

**Comparing spot vs. futures performance for the same strategy:**

```python
from siton.sdk import ema_cross, adx, Strategy, backtest

signal = ema_cross(fast=[8, 12], slow=[26, 50]).filter_by(
    adx(periods=[14], thresholds=[25])
)

# Run both in one command to see the difference
STRATEGY = [
    Strategy("Spot_LongOnly",    signal=signal, long_only=True,  top=5),
    Strategy("Futures_LongShort", signal=signal, long_only=False, top=5),
]

if __name__ == "__main__":
    backtest(STRATEGY)
```

The output shows both variants side by side so you can see how much value the
short leg adds (or subtracts) on your chosen asset and timeframe.

---

## 8. Position Management

### 8.1 Fixed Percentage Stops

```python
STRATEGY = Strategy(
    "ManagedTrend",
    signal=my_signal,
    stop_loss=[1.0, 2.0, 3.0],    # % below entry (long) / above entry (short)
    take_profit=[3.0, 5.0, 10.0], # % above entry (long) / below entry (short)
    trailing_stop=[2.0, 3.0],     # trailing stop %, moves with favorable price
)
```

Each stop value list adds a dimension to the backtest grid.

**How fixed stops work (long position example):**

```
Entry price = $100
stop_loss=2.0   → stop at $98   (cuts loss if price falls 2%)
take_profit=5.0 → exit at $105  (locks gain if price rises 5%)
trailing_stop=3.0 → initially at $97, moves up as price rises
```

### 8.2 ATR-Based Stops (Volatility-Adaptive)

ATR-based stops adapt automatically to asset volatility — essential when
backtesting across different assets or volatility regimes.

```python
STRATEGY = Strategy(
    "ATRManagedTrend",
    signal=my_signal,
    atr_period=14,                  # ATR lookback period
    atr_stop_mult=[1.0, 1.5, 2.0, 3.0],   # stop = entry ± mult × ATR
    atr_tp_mult=[2.0, 3.0, 5.0, 8.0],     # TP = entry ± mult × ATR
    atr_trail_mult=[1.5, 2.0],             # trailing = mult × ATR
)
```

**Comparison: fixed vs ATR stops for a 2× multiplier:**

| Asset | 4h ATR/price | ATR stop at 2× | Fixed 2% stop |
|-------|-------------|----------------|---------------|
| BTC   | ~1-2% | 2-4% below entry | 2% below entry |
| ETH   | ~1.5-3% | 3-6% below entry | 2% below entry |
| SOL   | ~3-7% | 6-14% below entry | 2% below entry (too tight!) |

Use ATR stops when: running on multiple assets, testing multiple timeframes,
or when fixed stops are hit by normal noise before any trend develops.

### 8.3 Risk-Per-Trade Sizing

By default every trade deploys `fraction` of equity regardless of stop width.
This means wider stops risk more equity.

With `risk_per_trade`, position size is calculated from the stop distance to keep
dollar risk **constant** per stop-out:

```
trade_value = equity × risk_per_trade / stop_distance
```

```python
STRATEGY = Strategy(
    "FixedRisk",
    signal=my_signal,
    stop_loss=[1.0, 2.0, 4.0],
    take_profit=[3.0, 6.0, 12.0],
    fraction=1.0,         # leverage cap: never exceed 100% of equity
    risk_per_trade=0.01,  # risk exactly 1% of equity per stop-out
)
```

**Worked example:**

```
capital = $10,000   risk_per_trade = 1%   stop_loss = 2%
trade_value = $10,000 × 1% / 2% = $5,000
loss if stopped = $5,000 × 2% = $100 = 1% of capital ✓

capital = $10,000   risk_per_trade = 1%   stop_loss = 4%
trade_value = $10,000 × 1% / 4% = $2,500
loss if stopped = $2,500 × 4% = $100 = 1% of capital ✓
```

### 8.4 Position Sizing Decision Tree

Use this flowchart to pick the right sizing approach for your strategy:

```
Do you use stop_loss or atr_stop_mult?
│
├── NO  → Use fraction= only.
│         Default fraction=1.0 deploys 100% of equity each trade.
│         Lower it (e.g. fraction=0.5) to reduce per-trade exposure.
│
└── YES → Do you want constant dollar risk per stop-out?
          │
          ├── NO  → fraction= only (wider stops risk more equity per trade)
          │
          └── YES → Add risk_per_trade=0.01 (1%) or similar.
                    Position size is calculated as:
                      trade_value = equity × risk_per_trade / stop_distance
                    fraction= acts as a leverage cap:
                      trade_value is capped at equity × fraction
```

**Practical sizing examples (equity = $10,000):**

| `fraction` | `risk_per_trade` | `stop_loss` | Trade size | Max loss |
|-----------|-----------------|------------|-----------|---------|
| 1.0 | — | — | $10,000 | unlimited (no stop) |
| 1.0 | — | 2.0% | $10,000 | $200 (2% of equity) |
| 1.0 | 1.0% | 2.0% | $5,000 | $100 (1% of equity) |
| 1.0 | 1.0% | 0.5% | $10,000 (capped) | $50 (0.5% of equity) |
| 0.5 | — | — | $5,000 | unlimited |

**Rule of thumb:**
- `fraction=1.0` (full equity) with `risk_per_trade=0.01` and `atr_stop_mult` is the
  most disciplined setup: risk is fixed in dollar terms, size adapts to volatility.
- Never set `risk_per_trade` without a stop — the formula divides by stop distance,
  which is undefined without a stop.

### 8.5 Capital and Compounding

Siton tracks a realistic equity curve: each trade's P&L updates the cash balance,
and the next trade sizes off that updated balance. This is **full compounding**.

```python
capital=10000.0  # starting equity; all sizing is relative to current equity
fraction=1.0     # fraction of *current* equity deployed — grows with wins, shrinks with losses
```

**Implications of compounding:**

- A 50% loss requires a 100% gain to recover — drawdown is asymmetric.
- With `fraction=1.0` and no stops, a single catastrophic bar can wipe equity to zero
  (margin call on a short, or gap through stop on a long).
- `risk_per_trade` + stop is the strongest protection: losses are bounded in
  fractional equity terms, so recovery is feasible even after a string of losses.

**Setting `capital=` for realistic cost simulation:**

The `capital=` value affects the absolute dollar P&L shown in logs but **not**
percentage-based metrics (return%, Sharpe, etc.), which are capital-neutral.
Set it to your intended real-money amount to see realistic fee drag:

```python
# Binance spot with realistic capital
STRATEGY = Strategy(
    "MyStrat",
    signal=signal,
    capital=50000.0,   # $50k starting capital
    fee=0.075,         # 0.075% taker fee (Binance standard)
    slippage=0.05,     # 0.05% slippage (liquid pair)
)
```

---

## 9. Data Sources

### 9.1 Synthetic Data (Demo Mode)

No internet required. Generates 10,000 realistic BTC-like 1h candles using a
GARCH(1,1) volatility model with fat-tailed Student-t returns.

```bash
siton my_strategy.py --demo
```

### 9.2 Live Exchange Data (ccxt)

Fetches from any ccxt-supported exchange with automatic pagination.

```bash
# Latest 5000 1h candles (default)
siton my_strategy.py -s BTC/USDT -t 1h

# Date range
siton my_strategy.py -s ETH/USDT -t 4h --start 2023-01-01 --end 2023-12-31

# Different exchange
siton my_strategy.py -s BTC/USDT -t 1d -e kraken
```

**Supported timeframes:** `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`,
`6h`, `8h`, `12h`, `1d`, `3d`, `1w`

### 9.3 CSV File

```bash
siton my_strategy.py --csv my_data.csv
```

The CSV must have columns: `timestamp` (epoch milliseconds), `open`, `high`,
`low`, `close`, `volume`.

### 9.4 Programmatic Data Loading

```python
from siton.data import fetch_ohlcv, load_csv, generate_sample
from siton.sdk import run

df = fetch_ohlcv("BTC/USDT", "1h", "binance", start="2024-01-01")
# or
df = load_csv("my_data.csv", start="2024-01-01", end="2024-12-31")
# or
df = generate_sample(n=5000)

import numpy as np
data = {k: df[k].to_numpy().astype(np.float64) for k in ["open","high","low","close","volume"]}

results = run([STRATEGY], data, timeframe="1h")
```

---

## 10. Running a Backtest

### 10.1 CLI Flags

```
siton <strategy_file> [options]

Positional:
  strategy_file         Path to .py file defining STRATEGY variable

Data source (pick one):
  --demo                Synthetic data (no API)
  --csv FILE            Load from CSV file
  -s, --symbol PAIR     Trading pair (default: BTC/USDT)
  -e, --exchange NAME   Exchange (default: binance)
  -n, --limit N         Number of candles (default: 5000)
  --start YYYY-MM-DD    Start date (inclusive)
  --end   YYYY-MM-DD    End date (inclusive)
  -t, --timeframe TF    Candle size (default: 1h)

Cross-validation:
  --cv                  Enable purged K-fold cross-validation
  --cv-folds N          Number of folds (default: 5)
  --purge-bars N        Bars to remove at train/test boundary (default: 50)
```

### 10.2 The STRATEGY Variable

Your strategy file must expose a `STRATEGY` variable at module level. It can be
a single `Strategy` or a list of strategies (compared side by side):

```python
# Single strategy
STRATEGY = Strategy("MyStrat", signal=signal)

# Multiple variants compared in one run
STRATEGY = [STRATEGY_FIXED, STRATEGY_RPT, STRATEGY_ATR]
```

### 10.3 Programmatic API

```python
from siton.sdk import run
from siton.engine import rank_results

results = run([STRATEGY], data, timeframe="1h")
# results is list[Result] when cv=False (default)
ranked = rank_results(results, sort_by="sharpe_ratio")

for r in ranked[:5]:
    print(f"{r.strategy}: return={r.total_return_pct:+.2f}%, sharpe={r.sharpe_ratio:.3f}")
    print(f"  params: {r.params}")

# results is dict{"folds", "aggregated", "cv_folds", "purge_bars", "stability"} when cv=True
```

The `Result` dataclass fields:

| Field | Type | Description |
|-------|------|-------------|
| `strategy` | str | Strategy name |
| `params` | dict | Parameter combination |
| `total_return_pct` | float | Total return in % |
| `sharpe_ratio` | float | Annualized Sharpe ratio |
| `max_drawdown_pct` | float | Maximum drawdown in % |
| `win_rate_pct` | float | % of winning trades |
| `num_trades` | int | Number of completed trades |
| `profit_factor` | float | Gross profit / gross loss |
| `sortino_ratio` | float | Downside-deviation Sharpe |
| `calmar_ratio` | float | Return / max drawdown |
| `psr` | float | Probabilistic Sharpe Ratio |
| `cv_consistency` | float | Fraction of CV folds with positive OOS Sharpe (0.0 when not using CV) |
| `equity_curve` | ndarray | Per-bar equity (if attached) |

---

## 11. Reading the Results

A typical output looks like:

```
======================================================================
  TOP 5 STRATEGIES (sorted by sharpe_ratio)
======================================================================
  # Strategy              Return%   Sharpe   MaxDD%  WinRate%  Trades      PF  Params
---------------------------------------------------------------------
  1 EMA_RSI_Filter         +42.31%    1.847    12.44    58.20%      87   1.821  ema_fast=12, ema_slow=50, rsi_period=14, rsi_oversold=30, rsi_overbought=70
  2 EMA_RSI_Filter         +38.77%    1.712    14.21    56.10%      93   1.743  ema_fast=8, ema_slow=50, ...
  ...

======================================================================
  BEST STRATEGY: EMA_RSI_Filter
  Params: {'ema_fast': 12, 'ema_slow': 50, ...}
  Return: +42.31% | Sharpe: 1.847 | MaxDD: 12.44%
  Win Rate: 58.20% | Trades: 87 | Profit Factor: 1.821

  Buy & Hold: +28.15%
  Alpha (vs B&H): +14.16%
======================================================================
```

### Metric interpretation guide

| Metric | Good | Caution | Poor |
|--------|------|---------|------|
| Sharpe Ratio | > 1.0 | 0.5 – 1.0 | < 0.5 |
| Max Drawdown | < 10% | 10 – 25% | > 25% |
| Win Rate | Depends on R:R | — | < 35% with PF < 1.5 |
| Profit Factor | > 1.5 | 1.0 – 1.5 | < 1.0 |
| Num Trades | > 30 | 10 – 30 | < 10 (not statistically significant) |

**Always compare to Buy & Hold.** A high return that still lags buy-and-hold
on a trending asset does not justify the complexity or risk.

---

## 12. Purged K-Fold Cross-Validation

Walk-forward validation gives one OOS estimate, which is noisy. Purged K-fold CV
(López de Prado, *Advances in Financial Machine Learning*) repeats the evaluation
across **K non-overlapping OOS folds**, using an anchored expanding training window
for each, and aggregates the results for a more statistically robust ranking.

#### What "purged" means

Indicator warm-up periods cause the last few bars of any training window to be
contaminated by data from the upcoming test window. A **purge gap** removes
`purge_bars` bars from the end of each training window before fitting, eliminating
this look-ahead bias at the boundary.

```
Full dataset (n bars)
├── Fold 1:  train=[0 : fold_size - purge_bars]   test=[fold_size : 2×fold_size]
├── Fold 2:  train=[0 : 2×fold_size - purge_bars]  test=[2×fold_size : 3×fold_size]
└── Fold 3:  train=[0 : 3×fold_size - purge_bars]  test=[3×fold_size : end]
             └── purge gap ──┘  ← contaminated bars removed
```

Training always starts from bar 0 (anchored expanding) to respect temporal order
and maximize the training set available at each fold.

#### Enabling CV

```python
STRATEGY = Strategy(
    "TrendFollowing",
    signal=ema_cross(fast=[8, 12], slow=[26, 50]) & adx(periods=[14], thresholds=[25]),
    cv=True,         # enable purged K-fold CV
    cv_folds=5,      # number of folds (fold 0 always skipped — no prior training data)
    purge_bars=50,   # bars removed at each train/test boundary
    top=5,
)
```

```bash
# CLI flags override Strategy defaults
siton examples/07_purged_kfold_cv.py --demo --cv --cv-folds 5 --purge-bars 50
```

#### Output format

```
======================================================================
  PURGED K-FOLD CV — 5 FOLDS, PURGE=50 BARS
======================================================================
  AGGREGATED OOS (sorted by sharpe_ratio)
======================================================================
  #  Strategy              Return%   Sharpe   MaxDD%  WinRate%  Trades     PF    PSR  Cons  Params
  1  TrendFollowing        +12.3%    1.234    -8.45%   63.2%      89    1.567  0.872   4/3  ema_fast=8, ...

  FOLD DETAILS
  Fold 1: train=[0:950]  test=[1000:2000]
    IS top-1: ema_fast=8, ema_slow=26 | Sharpe=1.452
    OOS best: ema_fast=8, ema_slow=26 | Return=+8.34%  Sharpe=1.120
  ...
```

The **Cons** (consistency) column shows `n_positive_folds / n_folds_run` — the
fraction of OOS windows in which the combo produced a positive Sharpe ratio.

#### The `cv_consistency` metric

```python
results = run(STRATEGY, df)
for r in results["aggregated"]:
    n = len(results["folds"])
    print(f"Consistency: {int(r.cv_consistency * n)}/{n}  Sharpe: {r.sharpe_ratio:.3f}")
```

| Consistency | Interpretation |
|-------------|----------------|
| K/K (e.g. 3/3) | Profitable in every fold — strongest robustness signal |
| (K−1)/K | One bad fold — acceptable; check if it was an anomalous period |
| < 50% | More losing folds than winning — likely overfitted; discard |

#### Programmatic API

```python
from siton.sdk import ema_cross, adx, Strategy, run
from siton.data import generate_sample

df = generate_sample(n=5000)
strat = Strategy(
    "Test",
    signal=ema_cross(fast=[8, 12], slow=[26, 50]),
    cv=True, cv_folds=5, purge_bars=50,
)
results = run(strat, df)

print(results.keys())
# dict_keys(['folds', 'aggregated', 'cv_folds', 'purge_bars'])

# Aggregated OOS ranking — mean metrics across all folds
for r in results["aggregated"]:
    n = len(results["folds"])
    print(f"{r.params}  sharpe={r.sharpe_ratio:.3f}  consistency={int(r.cv_consistency*n)}/{n}")

# Per-fold detail (IS top-K and OOS results for each fold)
for fd in results["folds"]:
    print(f"Fold {fd['fold']}: train={fd['train_range']}  test={fd['test_range']}")
    print(f"  IS top-1: {fd['train_top'][0].params}")
```

#### Choosing `cv_folds` and `purge_bars`

| Dataset size | Recommended `cv_folds` | Recommended `purge_bars` |
|---|---|---|
| < 2,000 bars | 3 | 20 |
| 2,000 – 10,000 bars | 5 | 50 |
| > 10,000 bars | 5 – 10 | 100 |

- **`purge_bars`** should be at least as large as the longest indicator lookback
  in your signal. For a 200-period SMA use `purge_bars=200`.
- **`cv_folds`** controls the trade-off between fold count and fold length.
  With 5,000 bars and 5 folds, each OOS window is 1,000 bars. Fewer folds give
  longer (more reliable) OOS windows; more folds give more diverse estimates.
- Fold 0 is always skipped — no prior data to train on — so `cv_folds=5`
  produces at most 4 OOS estimates (fewer if early folds don't have enough
  training data after the purge).

---

## 13. Strategy Patterns Cookbook

### Pattern 1: Simple Trend Following

```python
from siton.sdk import ema_cross, Strategy, backtest

signal = ema_cross(fast=[8, 12, 21], slow=[50, 100, 200])

STRATEGY = Strategy("SimpleTrend", signal=signal, top=5, sort="sharpe_ratio")

if __name__ == "__main__":
    backtest(STRATEGY)
```

Grid: up to 9 EMA pairs (invalid combos where fast >= slow are auto-excluded).

---

### Pattern 2: Trend with Oscillator Filter

Prevent entering during overbought/oversold conditions.

```python
from siton.sdk import ema_cross, rsi, Strategy, backtest

trend = ema_cross(fast=[8, 12], slow=[26, 50])

# ~rsi = neutral zone gate (fires when RSI is NOT at extremes)
rsi_gate = ~rsi(periods=[14], oversold=[25, 30], overbought=[70, 75])

signal = trend.filter_by(rsi_gate)

STRATEGY = Strategy(
    "TrendRSIFilter",
    signal=signal,
    long_only=True,     # spot trading only
    top=10,
)
```

Grid: 2 × 2 × 2 × 2 × 2 = 32 combinations.

---

### Pattern 3: Multi-Indicator Confluence

All indicators must agree before entering.

```python
from siton.sdk import ema_cross, macd_signal, adx, Strategy, backtest

trend     = ema_cross(fast=[8, 12], slow=[26, 50])
momentum  = macd_signal(fast=[12], slow=[26], signal_period=[9])
confirmed = trend & momentum   # both must point same way

strength  = adx(periods=[14, 20], thresholds=[20, 25])
signal    = confirmed.filter_by(strength)   # only in trending markets

STRATEGY = Strategy("Confluence", signal=signal, stop_loss=[2.0, 3.0], top=10)
```

Grid: 2×2 EMA × 1×1×1 MACD × 2×2 ADX × 2 SL = 64 combinations.

---

### Pattern 4: Regime-Adaptive (Trend + Mean Reversion)

Automatically switch strategy based on market regime.

```python
from siton.sdk import (
    ema_cross, macd_signal, obv,
    bollinger_bands, rsi, cci,
    adx, Strategy, backtest,
)

# Regime detection
trending = adx(periods=[14], thresholds=[25])
ranging  = ~adx(periods=[14], thresholds=[25])   # complementary

# Trend arm: all three must agree, fires only when trending
trend_arm = (
    ema_cross(fast=[8, 13, 21], slow=[55, 89, 144])
    & macd_signal(fast=[8, 12], slow=[21, 26], signal_period=[9])
    & obv(ma_periods=[20])
).filter_by(trending)

# Mean reversion arm: all three must agree, fires only when ranging
mr_arm = (
    bollinger_bands(windows=[14, 20, 26], num_std=[1.5, 2.0])
    & rsi(periods=[9, 14], oversold=[28, 32], overbought=[68, 72])
    & cci(periods=[14, 20], oversold=[-100], overbought=[100])
).filter_by(ranging)

# Mutually exclusive: only one arm can fire at a time
entry = trend_arm | mr_arm

STRATEGY = Strategy(
    "RegimeAdaptive",
    signal=entry,
    stop_loss=[1.0, 1.5, 2.0, 3.0],
    take_profit=[2.0, 3.0, 5.0, 8.0],
    top=20,
    sort="sharpe_ratio",
)
```

Grid: 36 trend × 96 MR × 4 SL × 4 TP = 55,296 backtests.

---

### Pattern 5: Entry/Exit State Machine

Use a separate, asymmetric exit indicator.

```python
from siton.sdk import ema_cross, sar, Strategy, backtest

entry_signal = ema_cross(fast=[8, 12], slow=[26, 50])
exit_signal  = sar(accelerations=[0.02, 0.03], maximums=[0.2, 0.3])

STRATEGY = Strategy(
    "EMAEntryWithSAR",
    entry=entry_signal,
    exit=exit_signal,
    top=10,
)
```

Position opens when EMA crossover fires. Position is held until SAR flips —
even if the EMA cross reverts. Use this pattern when you want the exit to "ride"
a trend past the entry signal's natural close.

---

### Pattern 6: Candlestick Pattern Entry with MA Exit

```python
from siton.sdk import candlestick_pattern, ema_cross, Strategy, backtest

# Bullish engulfing as entry, EMA cross as exit
entry  = candlestick_pattern("CDLENGULFING")
exit_  = ema_cross(fast=[12], slow=[50])

STRATEGY = Strategy("CandleEntry", entry=entry, exit=exit_, top=5)
```

---

### Pattern 7: Majority Vote

Enter only when most indicators agree.

```python
from siton.sdk import ema_cross, rsi, macd_signal, Signal, Strategy, backtest

signal = Signal.majority(
    ema_cross(fast=[12], slow=[50]),
    rsi(periods=[14], oversold=[30], overbought=[70]),
    macd_signal(fast=[12], slow=[26], signal_period=[9]),
)

STRATEGY = Strategy("MajorityVote", signal=signal, top=10)
```

Fires long only when at least 2 of 3 indicators are bullish.

---

### Pattern 8: Spot Market Long-Only Strategy

Optimized for buying and holding crypto on a spot exchange — no shorting.
The strategy holds cash during downtrends rather than going short.

```python
from siton.sdk import ema_cross, rsi, obv, adx, Strategy, backtest

# Trend signal: bullish when fast EMA > slow EMA
trend = ema_cross(fast=[8, 12, 21], slow=[50, 100, 200])

# Only enter longs when RSI is not yet overbought (avoid buying tops)
rsi_not_extreme = ~rsi(periods=[14], oversold=[30], overbought=[65, 70])

# Volume confirming accumulation
vol_confirm = obv(ma_periods=[20, 50])

# Trending regime (avoid ranging markets)
in_trend = adx(periods=[14], thresholds=[20, 25])

signal = (
    trend
    .filter_by(rsi_not_extreme)   # don't chase overbought moves
    .filter_by(vol_confirm)        # require volume backing
    .filter_by(in_trend)           # only in trend mode
)

STRATEGY = Strategy(
    "SpotBTC_LongOnly",
    signal=signal,
    long_only=True,         # spot account — no shorting
    atr_period=14,
    atr_stop_mult=[1.5, 2.0, 2.5],
    atr_tp_mult=[3.0, 5.0, 8.0],
    risk_per_trade=0.01,    # risk 1% of equity per stop-out
    top=10,
    sort="sharpe_ratio",
)

if __name__ == "__main__":
    backtest(STRATEGY)
```

Key properties of this design:
- Sits in **cash** during downtrends and ranging markets (no exposure, no loss).
- Enters longs only when three independent conditions agree.
- ATR stops adapt to asset volatility automatically.
- `risk_per_trade=0.01` keeps each stop-out at exactly 1% of equity.

---

### Pattern 9: Derivatives Long + Short (Perpetual Futures)

On a perpetual futures exchange you can profit from both rising and falling prices.
This strategy is designed to capture both directions.

```python
from siton.sdk import ema_cross, macd_signal, adx, rsi, Strategy, backtest

# Base trend: long when fast > slow, short when fast < slow
trend = ema_cross(fast=[8, 12], slow=[50, 100])

# Require MACD to agree directionally for both longs and shorts
momentum = macd_signal(fast=[12], slow=[26], signal_period=[9])

# Gate: only trade when ADX confirms a trend is present
trending = adx(periods=[14], thresholds=[20, 25])

# Block entries at RSI extremes (avoid chasing overbought longs / oversold shorts)
rsi_neutral = ~rsi(periods=[14], oversold=[25, 30], overbought=[70, 75])

signal = (
    (trend & momentum)              # trend + momentum must agree on direction
    .filter_by(trending)            # only in trending regimes
    .filter_by(rsi_neutral)         # block RSI extremes
)

STRATEGY = Strategy(
    "Futures_LongShort",
    signal=signal,
    # long_only=False is the default — both +1 and -1 signals fire
    atr_period=14,
    atr_stop_mult=[1.5, 2.0, 3.0],
    atr_tp_mult=[3.0, 5.0, 8.0],
    risk_per_trade=0.01,
    top=10,
    sort="sharpe_ratio",
)

if __name__ == "__main__":
    backtest(STRATEGY)
```

This fires short positions during confirmed downtrends, which can significantly
improve performance during bear markets compared to a long-only equivalent.

> **Deployment note:** Make sure your live account type matches your backtest.
> Running this strategy on a spot account will skip all short signals at runtime,
> producing results that diverge from the backtest.

---

### Pattern 10: ATR-Based Stops for Cross-Asset Robustness

```python
from siton.sdk import ema_cross, adx, Strategy, backtest

signal = ema_cross(fast=[8, 12], slow=[26, 50]).filter_by(
    adx(periods=[14], thresholds=[25])
)

STRATEGY = Strategy(
    "ATRStops",
    signal=signal,
    atr_period=14,
    atr_stop_mult=[1.0, 1.5, 2.0, 3.0],
    atr_tp_mult=[2.0, 3.0, 5.0, 8.0],
    fraction=1.0,
    risk_per_trade=0.01,    # 1% equity risk per trade
    top=10,
)
```

This combination (ATR stops + risk-per-trade sizing) is the most
volatility-adaptive setup: stop distance and position size both adapt to
current market conditions.

---

## 14. Grid Sizing Guide

The backtest grid is the **cartesian product** of all parameter lists. Large grids
are fast (the engine parallelizes across cores) but increase the risk of
overfitting. A warning is printed for grids > 500 combinations per signal.

### Rough guidelines

| Scenario | Suggested grid size |
|----------|-------------------|
| Quick exploration | < 100 |
| Standard optimization | 100 – 5,000 |
| Deep search | 5,000 – 100,000 |
| Very deep (regime-adaptive) | 50,000 – 200,000 |

### How to count your grid

```python
# Each indicator adds a multiplier:
ema_cross(fast=[8,12], slow=[26,50])               # 2 × 2 = 4
macd_signal(fast=[12], slow=[26], signal_period=[9]) # 1 × 1 × 1 = 1
adx(periods=[14,20], thresholds=[20,25])            # 2 × 2 = 4

# Composition multiplies:
(ema & macd).filter_by(adx)  # 4 × 1 × 4 = 16 signal combos

# Strategy adds stop/tp sweep:
Strategy(..., stop_loss=[1,2,3], take_profit=[3,5,10])  # × 3 × 3 = 9
# Total: 16 × 9 = 144 backtests
```

### Tips to manage grid size

1. **Fix structural parameters**: ADX threshold at 25 (Wilder's standard), MACD
   signal period at 9. These are well-established values, not variables to
   optimize.
2. **Use fewer but spread-out values**: `[8, 21, 55]` tests the range better
   than `[8, 9, 10, 11, 12]`.
3. **Limit stop/TP sweep**: 3-4 values per dimension is usually enough.
4. **Use `long_only=True`**: halves signal diversity and focuses optimization.

---

## 15. Common Pitfalls

### 15.1 fast >= slow in crossover indicators

Any combo where `fast >= slow` is automatically detected and returns all zeros
(no signal). You will still see these in the grid count but they contribute no
trades and will rank last.

```python
ema_cross(fast=[8, 50], slow=[26, 50])
# Combo (fast=50, slow=50) → skipped automatically
# Combo (fast=50, slow=26) → skipped automatically
```

### 15.2 `~signal` is NOT long/short inversion

`~signal` produces a **binary filter** (1 where flat, 0 where non-flat). Use it
with `.filter_by()` to mean "when this indicator is calm/neutral".

```python
# CORRECT: trade trend only when RSI is not at extremes
safe = trend.filter_by(~rsi(...))

# WRONG intention: this does NOT make trend go short when RSI is at extremes
# It gates trend through RSI-neutral zones
wrong = trend & ~rsi(...)   # same result as filter_by but semantics are clear
```

To flip long to short, use `.negate()`.

### 15.3 `signal=` vs `entry=/exit=` overfitting

The `entry=/exit=` state machine holds positions until the exit fires. With a
slow SAR (`acceleration=0.01`), losing mean-reversion positions can be held for
days, accumulating large losses. In `signal=` mode, positions close as soon as
conditions disappear.

**Recommendation**: default to `signal=` mode. Use `entry=/exit=` only when you
explicitly want the state machine's hold-until-exit behavior.

### 15.4 Timeframe mismatch with fixed stops

On short timeframes (1h), a 2% stop-loss is often smaller than one bar's natural
range. The stop fires on noise before a trend can develop.

- For **1h** data: stops below ~0.5% are very tight (many false exits).
- For **4h** data: 2-3% stops are appropriate.
- For **1d** data: 5-10% stops are typical.
- **Best solution**: use ATR-based stops — they adapt automatically to timeframe
  and asset volatility.

### 15.5 Too few trades

A Sharpe ratio computed on 5 trades is meaningless. Check `num_trades` and aim
for at least 30 trades for any metric to be statistically informative.

If your top results all have very few trades, your signal is too selective:
- Loosen your filters
- Use a longer date range
- Use a higher-frequency timeframe

### 15.6 Overfitting / in-sample bias

Every parameter combination you search is a potential overfitting degree of
freedom. To guard against it:

1. **Run `--cv`** on every promising strategy before trusting results — K
   independent OOS estimates are far more reliable than a single in-sample run
   (see Section 12).
2. **Check `cv_consistency`**: a combo that is profitable in < 50% of folds
   should be discarded regardless of average Sharpe.
3. **Check the parameter stability table**: stable params across IS windows
   are a strong robustness signal.
4. **Limit grid diversity**: fewer parameters, wider spacing.
5. **Fix known-good values**: use literature defaults (ADX=25, RSI 30/70)
   before widening the search.
6. **Check OOS Sharpe > 0**: a strategy that profits in-sample but loses
   out-of-sample is pure curve-fitting.

### 15.7 Not enough data

TA-Lib indicators need a warm-up period equal to their longest parameter before
they produce valid values. A 200-period SMA needs 200+ candles just to
initialize.

Rule of thumb: `data_length > 3 × max_indicator_period + desired_trade_count × avg_trade_duration`

Fetch at least 3,000–5,000 candles for meaningful results.

### 15.8 Spot vs. Derivatives Mismatch

A backtest that runs in full long+short mode (`long_only=False`, the default)
will not match what happens live on a spot account, because a spot account
cannot execute short positions.

**Symptom:** Backtest shows strong performance, but live spot account only
captures roughly half the trades and diverges from the equity curve.

**Root cause:** Short signals (`-1`) fire in the backtest but are silently
skipped on spot, creating a structural mismatch.

**Fix:**

```python
# WRONG for a spot account: backtests short legs that will never execute live
STRATEGY = Strategy("WrongForSpot", signal=my_signal)

# CORRECT for a spot account: backtest matches live behavior
STRATEGY = Strategy("CorrectForSpot", signal=my_signal, long_only=True)
```

**Checklist before live deployment:**

| Question | Spot account answer | Futures account answer |
|----------|--------------------|-----------------------|
| Can you short? | No | Yes |
| Set `long_only=`? | `True` | `False` (default) |
| Short signals in backtest? | Must be absent | Expected |
| Right benchmark? | Buy & Hold | Optional (B&H is one-sided) |

### 15.9 Grid collision renaming

When two composed signals have the same parameter name (e.g., both use `adx`),
the second signal's keys are automatically renamed with a numeric suffix
(`adx_period_2`, `adx_threshold_2`). This is expected behavior — the output
table will show both.

---

## 16. Execution Costs and Realistic Backtesting

Every live trade incurs two categories of friction: fees charged by the exchange
and market impact (slippage). Siton models both in every backtest.

### 16.1 Trading Fees

```python
Strategy("MyStrat", signal=signal, fee=0.075)   # 0.075% taker fee
```

`fee` is expressed as a **percentage** (not a fraction). It is deducted on both
the entry fill and the exit fill of every trade.

**Fee is applied to both legs:**

```
Entry:  you spend exactly trade_value cash; shares received = trade_value / (fill × (1 + fee/100))
Exit:   you receive shares × fill × (1 - fee/100)
```

This matches exchange behavior: the exchange keeps a percentage of every transaction.

**Default:** `fee=0.075` (0.075% = Binance standard taker rate, BNB discount not applied).

**Reference fees by context:**

| Exchange / Context | Typical taker fee | Siton `fee=` |
|-------------------|------------------|-------------|
| Binance Spot (standard) | 0.10% | `0.10` |
| Binance Spot (BNB discount) | 0.075% | `0.075` (default) |
| Binance Futures (standard) | 0.05% | `0.05` |
| Coinbase Advanced | 0.06–0.08% | `0.06` to `0.08` |
| Kraken Pro | 0.10–0.26% | `0.10` to `0.26` |
| OKX Futures | 0.02–0.05% | `0.02` to `0.05` |
| Hypothetical zero-fee | — | `fee=0.0` |

**Fee drag on frequent trading:**

A round-trip (entry + exit) costs `2 × fee`. At 0.075%:

```
Round-trip cost = 2 × 0.075% = 0.15% per trade
100 trades/year → 15% return consumed by fees alone
```

This is why strategies with many small trades (`num_trades > 500` per year) rarely
survive realistic fee modeling. Always set `fee=` to match your actual exchange.

### 16.2 Slippage Model

```python
Strategy("MyStrat", signal=signal, slippage=0.05)   # 0.05% slippage
```

`slippage` models market impact: the fill price is **worse than the bar open** due
to order book depth, bid-ask spread, and latency.

**How slippage is applied:**

```
Buy fill  = open[i] × (1 + slippage/100)   ← you pay more than the open
Sell fill = open[i] × (1 - slippage/100)   ← you receive less than the open
```

This is a simple but conservative model. A `slippage=0.05` (0.05%) means your fills
are 0.05% worse than the theoretical open price in each direction.

**Choosing a slippage value:**

| Pair / Liquidity | Realistic slippage | `slippage=` |
|-----------------|-------------------|-------------|
| BTC/USDT, ETH/USDT (top exchanges) | 0.01–0.05% | `0.02` to `0.05` |
| Mid-cap coins (SOL, BNB) | 0.05–0.10% | `0.05` to `0.10` |
| Small-cap / low liquidity | 0.2–1.0%+ | `0.20` to `1.0` |
| Conservative worst-case | — | `0.10` |

**Default:** `slippage=0.05`. This is reasonable for major pairs but may be too
optimistic for small-cap assets — adjust accordingly.

### 16.3 The Full Round-Trip Cost Model

Every completed trade consumes:

```
round_trip_cost = 2 × fee + 2 × slippage   (both in %)
```

With defaults (`fee=0.075, slippage=0.05`):

```
round_trip = 2 × 0.075% + 2 × 0.05% = 0.25% per round trip
```

A strategy must generate more than 0.25% per trade just to break even.

**Stress-testing with realistic costs:**

```python
# Conservative — assumes higher fees and worse fills
STRATEGY_CONSERVATIVE = Strategy(
    "ConservativeTest",
    signal=signal,
    fee=0.10,        # higher fee tier
    slippage=0.10,   # 0.10% slippage (wider spread / less liquid)
    top=5,
)

# Optimistic — best-case fee + ultra-liquid pair
STRATEGY_OPTIMISTIC = Strategy(
    "OptimisticTest",
    signal=signal,
    fee=0.02,        # VIP tier or maker rebate scenario
    slippage=0.02,
    top=5,
)

# Compare both to see how robust the strategy is to cost assumptions
STRATEGY = [STRATEGY_CONSERVATIVE, STRATEGY_OPTIMISTIC]
```

A strategy that survives the conservative scenario is likely to hold up in live trading.

### 16.4 Stop Exits and Re-entry Costs

When a stop fires intrabar, the engine closes the position at the stop price and
immediately re-enters (at the close price) if the signal is still active.
**Each stop-out counts as two trades** (close + reopen), so excessive stops
generate extra fee drag.

```
Stop fires → exit at stop price (with slippage + fee)
           → signal still active → re-enter at close[i] (with slippage + fee)
Total cost  = 2 more fills on top of the normal round-trip
```

This is one reason aggressive trailing stops can hurt performance on noisy assets:
they generate many stop-outs and re-entries, each consuming fees.

### 16.5 Short Selling Costs (Borrow Rate)

Shorting on derivatives (perpetual futures) incurs a **funding rate** or borrow cost
paid periodically to the long side. The engine models this as `borrow_rate_per_bar`,
which is currently set to `0.0` (disabled) in all backtests.

This means **short-side backtests are optimistic** for funding-rate-heavy periods
(e.g., strong bull markets where funding rates can reach 0.1%/8h = ~110% annualized).

For conservative futures backtesting, account for this manually when interpreting
short-side returns by subtracting an estimated annual borrow cost from the OOS result.

---

## 17. Performance Metrics Deep Dive

Siton computes eight metrics for every parameter combination. This section explains
what each one measures, how it is computed, and when to use it for ranking.

### 17.1 Sharpe Ratio

**Formula:**

```
Sharpe = mean(excess_returns) / std(excess_returns) × √(bars_per_year)
excess_return[i] = bar_return[i] - risk_free_rate_per_bar
```

**Key properties:**
- Only active bars (position open) contribute to the mean and variance — flat periods
  do not penalize the ratio for being in cash.
- Uses **sample standard deviation** (divides by N−1), which is the statistically
  correct estimator for finite series.
- Annualized via `√(bars_per_year)` where the annualization factor is derived from
  the timeframe (e.g., `√8760` for 1h, `√365` for 1d).

**Interpretation:**

| Sharpe | Meaning |
|--------|---------|
| > 2.0 | Exceptional — verify it is not overfitted |
| 1.0 – 2.0 | Good — worth investigating further |
| 0.5 – 1.0 | Mediocre — may improve with better risk management |
| < 0.5 | Poor — likely not profitable after costs in live trading |

**Setting `risk_free_rate`:**

```python
# US Treasury rate (~5% in 2024), annualized
STRATEGY = Strategy("MyStrat", signal=signal, risk_free_rate=0.05)
```

Default is `0.0`. Raising it lowers every Sharpe ratio but makes comparisons
more rigorous: a strategy should beat the risk-free rate to justify the risk.

### 17.2 Sortino Ratio

Like Sharpe, but uses **downside deviation** (standard deviation of negative
bar returns only) instead of total deviation. Strategies that have smooth upswings
but sharp occasional losses are penalized more by Sortino than by Sharpe.

```
Sortino = mean(excess_returns) / downside_std(excess_returns) × √(bars_per_year)
```

Sortino is always ≥ Sharpe (because downside_std ≤ total_std). A strategy with
Sharpe 1.5 but Sortino 1.2 has significant left-tail risk. A strategy with
Sharpe 1.5 and Sortino 2.0 has mostly upside variance — a positive trait.

**When to prefer Sortino:** when evaluating strategies with frequent small wins and
occasional large losses (e.g., mean-reversion strategies that can catch flash crashes).

### 17.3 Calmar Ratio

```
Calmar = annualized_CAGR / |max_drawdown_pct|
```

Measures return per unit of worst historical drawdown. A Calmar of 1.0 means
you earned your max drawdown back in one year.

| Calmar | Meaning |
|--------|---------|
| > 3.0 | Excellent capital efficiency |
| 1.0 – 3.0 | Good |
| < 1.0 | Drawdown is large relative to return |

**When to use Calmar over Sharpe:** when your primary concern is drawdown control
(e.g., investor capital, risk-of-ruin scenarios).

### 17.4 Max Drawdown

```
max_drawdown = max((equity_peak - equity_trough) / equity_peak)  over all time
```

The intrabar worst point is used (low for longs, high for shorts), making this
a conservative / realistic estimate of what you would have experienced live.

**Drawdown is asymmetric:**

```
Drawdown  Recovery needed
  10%  →     11.1%
  20%  →     25.0%
  50%  →    100.0%
  80%  →    400.0%
```

A 50% drawdown requires doubling your remaining capital just to get back to flat.
This asymmetry makes drawdown management as important as return maximization.

**Rules of thumb:**

| Strategy type | Acceptable max DD |
|--------------|------------------|
| Conservative (capital preservation) | < 10% |
| Moderate (retail systematic) | 10 – 20% |
| Aggressive (high return seeking) | 20 – 40% |
| Speculative | > 40% (usually not sustainable) |

### 17.5 Profit Factor

```
Profit Factor = gross_profit / gross_loss
```

Computed at the trade level (not bar level). A PF > 1.0 means wins exceed losses in
aggregate. Note that PF does not account for the frequency of wins vs losses — a
strategy with PF = 2.0 could be winning 30% of trades at 3:1 R or winning 67% at 1:1 R.

**Combined interpretation (PF + win rate):**

| Win Rate | PF needed to be profitable |
|----------|---------------------------|
| 30% | > 2.33 |
| 40% | > 1.50 |
| 50% | > 1.00 |
| 60% | > 0.67 |
| 70% | > 0.43 |

Low win-rate strategies (trend followers) need high PF. High win-rate strategies
(mean reversion) can survive with lower PF.

### 17.6 Probabilistic Sharpe Ratio (PSR)

```
PSR = P(true_Sharpe > 0 | observed_Sharpe, num_trades)
```

The PSR adjusts the observed Sharpe for the number of trades (sample size). A Sharpe
of 1.5 from 10 trades is far less reliable than 1.5 from 300 trades.

| PSR | Interpretation |
|-----|---------------|
| > 0.99 | Very high confidence the strategy has positive true Sharpe |
| 0.95 – 0.99 | High confidence |
| 0.80 – 0.95 | Moderate — increase trade count before trusting |
| < 0.80 | Low — likely noise; do not trust observed Sharpe |

PSR is shown in purged K-fold CV mode (`--cv`), where it is computed on the
full dataset length.

### 17.7 Choosing a Sort Metric

```python
Strategy("MyStrat", signal=signal, sort="sharpe_ratio")   # default
```

| Priority | Use `sort=` |
|----------|-------------|
| Risk-adjusted return (balanced) | `"sharpe_ratio"` |
| Drawdown control above all | `"calmar_ratio"` |
| Downside risk sensitivity | `"sortino_ratio"` |
| Raw return maximization | `"total_return_pct"` |
| Consistent winning trades | `"win_rate_pct"` |
| Risk:reward quality | `"profit_factor"` |
| Avoiding illiquid strategies | `"num_trades"` |

> **Warning:** sorting by `total_return_pct` or `win_rate_pct` alone strongly
> encourages overfitting. Sharpe and Calmar are much harder to game because they
> explicitly penalize variance and drawdown.

---

## 18. Multi-Strategy Runs and Capital Allocation

### 18.1 Running Multiple Strategies Side by Side

Pass a list of `Strategy` objects to compare them in one command:

```python
from siton.sdk import ema_cross, bollinger_bands, rsi, adx, Strategy, backtest

signal_trend = ema_cross(fast=[8, 12], slow=[50, 100]).filter_by(
    adx(periods=[14], thresholds=[25])
)
signal_mr = bollinger_bands(windows=[20], num_std=[2.0]) & rsi(
    periods=[14], oversold=[30], overbought=[70]
)

STRATEGY = [
    Strategy("TrendFollower", signal=signal_trend, top=5, sort="sharpe_ratio"),
    Strategy("MeanReversion",  signal=signal_mr,    top=5, sort="sharpe_ratio"),
]

if __name__ == "__main__":
    backtest(STRATEGY)
```

Both strategies are evaluated on the same data in one pass. The output shows
two ranked tables, one per strategy. This lets you compare regime suitability
directly — which one performs better on BTC/4h vs ETH/1h, for example.

### 18.2 Multi-Asset Workflow

Siton backtests one asset at a time. To evaluate a strategy across multiple assets,
run it in a loop and collect results:

```python
from siton.sdk import ema_cross, adx, Strategy, run
from siton.data import fetch_ohlcv
import numpy as np

signal = ema_cross(fast=[8, 12], slow=[50, 100]).filter_by(
    adx(periods=[14], thresholds=[25])
)
strategy = Strategy("TrendCross", signal=signal, top=3)

assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

for symbol in assets:
    df = fetch_ohlcv(symbol, "4h", start="2023-01-01")
    data = {k: df[k].to_numpy().astype(np.float64) for k in ["open","high","low","close","volume"]}
    results = run([strategy], data, timeframe="4h")
    best = results[0]
    print(f"{symbol:15s}  return={best.total_return_pct:+7.2f}%  "
          f"sharpe={best.sharpe_ratio:.3f}  params={best.params}")
```

Look for strategies whose **best parameter set is stable across assets** — that is a
strong sign of robustness rather than asset-specific curve fitting.

### 18.3 Interpreting Multi-Asset Results

| Pattern | Interpretation |
|---------|---------------|
| Same params win on all assets | Strategy is robust; parameter likely reflects market structure |
| Different params win on each asset | Likely overfit to each asset's historical noise |
| Positive on 3/4 assets | Consider asset-specific filtering in live deployment |
| Negative on trending assets | Strategy is mean-reversion — only use in non-trending regimes |

### 18.4 Capital Allocation Between Strategies

Siton does not model a portfolio of concurrent positions — each strategy is
evaluated as if it has full use of capital. For live multi-strategy deployment,
apply a capital allocation fraction manually:

```python
# Allocate 60% of capital to trend, 40% to mean reversion
STRATEGY = [
    Strategy("Trend", signal=signal_trend, capital=60000.0, top=5),
    Strategy("MeanRev", signal=signal_mr,  capital=40000.0, top=5),
]
```

This only affects the absolute dollar P&L display — percentage-based metrics
(return%, Sharpe) are unaffected by capital size.

**Simple capital allocation guidelines:**

| Strategy type | Typical allocation |
|--------------|-------------------|
| Primary (highest Sharpe) | 40 – 60% |
| Secondary (complementary regime) | 20 – 40% |
| Satellite (high-risk/high-reward) | 10 – 20% |
| Cash / risk-free reserve | 10 – 20% |

Strategies that are **negatively correlated** (trend-following + mean-reversion)
complement each other and reduce portfolio-level drawdown when allocated together.

### 18.5 Strategy Correlation Check

Before allocating capital to two strategies, check whether their equity curves move
together using the programmatic API:

```python
from siton.sdk import run
import numpy as np

results_trend = run([strategy_trend], data, timeframe="4h")
results_mr    = run([strategy_mr],    data, timeframe="4h")

curve_trend = results_trend[0].equity_curve
curve_mr    = results_mr[0].equity_curve

# Bar returns
ret_trend = np.diff(curve_trend) / curve_trend[:-1]
ret_mr    = np.diff(curve_mr)    / curve_mr[:-1]

corr = np.corrcoef(ret_trend, ret_mr)[0, 1]
print(f"Strategy correlation: {corr:.3f}")
# corr ≈ 0     → uncorrelated — good diversification
# corr > +0.7  → strategies move together — limited diversification benefit
# corr < -0.3  → negatively correlated — excellent portfolio pair
```

> **Note:** `equity_curve` is only attached to results when using the programmatic
> `run()` API. It is not printed by the CLI.

---

## Appendix A — Complete Import Reference

```python
# All public names available via wildcard import
from siton.sdk import (
    # MA crossovers
    ema_cross, sma_cross, dema_cross, tema_cross, wma_cross, kama_cross, trima_cross,
    # Oscillators
    rsi, cmo, cci, willr, mfi, stoch_rsi, ultimate_osc,
    # Momentum
    macd_signal, apo, ppo, mom, roc, trix,
    # Trend strength
    adx, adxr, dx, di_cross, dm_cross,
    # Volume
    obv, ad, adosc,
    # Volatility
    bollinger_bands, atr_breakout, natr_breakout, channel_breakout,
    # Exit signals
    sar, sar_ext,
    # Candlestick patterns
    candlestick_pattern,
    # Custom signals
    custom, indicator,
    # Core classes
    Signal, Strategy,
    # Runners
    backtest, run,
    # Utilities
    expand_grid, clear_cache, clear_sdk_cache,
)
```

---

## Appendix B — Minimal Strategy Template

```python
"""
My Strategy Name
================
Brief description of the strategy logic.

Run:
    python my_strategy.py --demo
    siton my_strategy.py -s BTC/USDT -t 4h --start 2023-01-01
"""

from siton.sdk import (
    # import your indicators here
    ema_cross, adx,
    Strategy, backtest,
)

# ── Signals ───────────────────────────────────────────────────────────────────

signal = ema_cross(fast=[8, 12], slow=[26, 50]).filter_by(
    adx(periods=[14], thresholds=[25])
)

# ── Strategy ──────────────────────────────────────────────────────────────────

STRATEGY = Strategy(
    "MyStrategy",
    signal=signal,
    stop_loss=[2.0, 3.0],
    take_profit=[5.0, 8.0],
    top=10,
    sort="sharpe_ratio",
)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    backtest(STRATEGY)
```
