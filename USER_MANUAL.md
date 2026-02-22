# Siton User Manual
### Building, Backtesting and Evaluating Trading Strategies

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
5. [Signal Constructors Reference](#5-signal-constructors-reference)
6. [Signal Composition Operators](#6-signal-composition-operators)
7. [The Strategy Class](#7-the-strategy-class)
8. [Position Management](#8-position-management)
9. [Data Sources](#9-data-sources)
10. [Running a Backtest](#10-running-a-backtest)
11. [Reading the Results](#11-reading-the-results)
12. [Walk-Forward Validation](#12-walk-forward-validation)
13. [Strategy Patterns Cookbook](#13-strategy-patterns-cookbook)
14. [Grid Sizing Guide](#14-grid-sizing-guide)
15. [Common Pitfalls](#15-common-pitfalls)

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

Signals can be combined using Python operators and method calls. All compositions
merge the parameter grids flat, producing the full cartesian product.

### 6.1 `&` — AND (Both Must Agree)

Both signals must point in the **same direction**. Any disagreement or neutral
signal produces `0`.

```
long + long  = long
short + short = short
long + short  = flat
long + flat   = flat
flat + flat   = flat
```

```python
# EMA trend AND MACD momentum — both must be bullish to go long
confluence = ema_cross(fast=[8, 12], slow=[26, 50]) & macd_signal(fast=[12], slow=[26], signal_period=[9])
```

### 6.2 `|` — OR (First Non-Flat Wins)

Takes the first non-zero signal. If both are non-zero and **conflict** (one long,
one short), the result is flat.

```
long + flat   = long   (trend arm active, mr arm silent)
flat + long   = long   (mr arm active, trend arm silent)
long + short  = flat   (conflict)
flat + flat   = flat
```

This is the key operator for **mutually exclusive regime arms**:

```python
# When ADX gates are complementary, only one arm can fire
combined = trend_arm | mean_reversion_arm
```

### 6.3 `~` — Invert to Binary Filter

`~signal` converts a signal to a **binary gate**: outputs `1` wherever the
original signal is `0` (flat), and `0` elsewhere.

This is **not** long-to-short inversion. It means "NOT in an extreme zone":

```python
# rsi() fires +1 when oversold, -1 when overbought
# ~rsi()  fires +1 when NEUTRAL (not at extremes), 0 when overbought/oversold
rsi_neutral = ~rsi(periods=[14], oversold=[30], overbought=[70])

# Gate trend through neutral RSI zone
safe_trend = trend.filter_by(rsi_neutral)
```

To algebraically invert long ↔ short, use `.negate()`.

### 6.4 `.filter_by(other)` — Gate a Signal

The primary signal passes through only where `other` is **non-zero** (active).
The direction of `other` does not need to match.

```python
# trend fires only when ADX > 25 (strong trend confirmed)
gated = trend.filter_by(adx(periods=[14], thresholds=[25]))
```

### 6.5 `.agree(other)` — Directional Gate

Stricter than `.filter_by()`: the primary signal passes only where `other` agrees
on **direction** (both long, or both short).

```python
# Signal passes only where volume confirms the same direction
volume_confirmed = signal.agree(obv(ma_periods=[20]))
```

### 6.6 `.confirm(other, lookbacks)` — Lookback Confirmation

Signal is valid only if `other` fired in the **same direction** within the last
`N` bars. Adds a `lookback` dimension to the grid.

```python
# EMA trend valid only if MACD agreed within 3, 5, or 8 bars
confirmed = ema_cross(fast=[12], slow=[50]).confirm(
    macd_signal(fast=[12], slow=[26], signal_period=[9]),
    lookbacks=(3, 5, 8),
)
```

### 6.7 `Signal.majority(*signals)` — Majority Vote

More than half of the supplied signals must agree on direction.

```python
# At least 2 out of 3 indicators must agree
signal = Signal.majority(
    ema_cross(fast=[12], slow=[50]),
    rsi(periods=[14], oversold=[30], overbought=[70]),
    macd_signal(fast=[12], slow=[26], signal_period=[9]),
)
```

### 6.8 `.negate()` — Algebraic Inversion

Flips long ↔ short: `+1 → -1`, `-1 → +1`, `0 → 0`. Rarely needed directly;
usually `~` (where-flat) is what you want.

### Operator summary table

| Expression | Semantics |
|-----------|-----------|
| `a & b` | Both agree |
| `a \| b` | First non-flat wins; conflict → flat |
| `~a` | Binary gate: 1 where a is flat |
| `a.filter_by(b)` | a passes where b is non-zero |
| `a.agree(b)` | a passes where b agrees on direction |
| `a.confirm(b, lookbacks)` | a passes where b fired same direction within N bars |
| `Signal.majority(a, b, c)` | Direction majority vote |
| `a.negate()` | Long ↔ short swap |

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
    # Walk-forward validation
    validate=False,
    train_ratio=0.7,
    n_splits=5,
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

Validation:
  --validate            Enable walk-forward validation
  --train-ratio FLOAT   IS/OOS split (default: 0.7 = 70% train)
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
ranked = rank_results(results, sort_by="sharpe_ratio")

for r in ranked[:5]:
    print(f"{r.strategy}: return={r.total_return_pct:+.2f}%, sharpe={r.sharpe_ratio:.3f}")
    print(f"  params: {r.params}")
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

## 12. Walk-Forward Validation

Walk-forward validation splits the data into an in-sample (IS) training period
and an out-of-sample (OOS) test period, helping detect overfitting.

```bash
siton my_strategy.py -s BTC/USDT -t 4h --start 2022-01-01 --end 2024-12-31 --validate --train-ratio 0.7
```

This uses 70% of bars for optimization and 30% for out-of-sample testing.

The output adds two extra columns:

- **PSR** (Probabilistic Sharpe Ratio): probability that the true Sharpe is
  positive. Values > 0.95 are strong.
- **WFE** (Walk-Forward Efficiency): OOS return / IS return. Values > 0.5 suggest
  the strategy is generalizing well.

The **parameter stability table** shows the top-1 parameter set across expanding
IS windows. Consistent parameters across windows are a strong sign of robustness.

### Interpreting IS/OOS results

| Scenario | Interpretation |
|----------|---------------|
| Strong IS, strong OOS | Robust strategy |
| Strong IS, flat/negative OOS | Overfitted to training period |
| Moderate IS, moderate OOS | Possibly generalizable |
| Params change each window | Unstable, likely random pattern mining |

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

### Pattern 8: ATR-Based Stops for Cross-Asset Robustness

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

1. **Run `--validate`** on every promising strategy before trusting results.
2. **Check the parameter stability table**: stable params across IS windows
   are a strong robustness signal.
3. **Limit grid diversity**: fewer parameters, wider spacing.
4. **Fix known-good values**: use literature defaults (ADX=25, RSI 30/70)
   before widening the search.
5. **Check OOS Sharpe > 0**: a strategy that profits in-sample but loses
   out-of-sample is pure curve-fitting.

### 15.7 Not enough data

TA-Lib indicators need a warm-up period equal to their longest parameter before
they produce valid values. A 200-period SMA needs 200+ candles just to
initialize.

Rule of thumb: `data_length > 3 × max_indicator_period + desired_trade_count × avg_trade_duration`

Fetch at least 3,000–5,000 candles for meaningful results.

### 15.8 Grid collision renaming

When two composed signals have the same parameter name (e.g., both use `adx`),
the second signal's keys are automatically renamed with a numeric suffix
(`adx_period_2`, `adx_threshold_2`). This is expected behavior — the output
table will show both.

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
