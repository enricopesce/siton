<div align="center">

<img src="siton_logo.webp" alt="Siton logo" width="220"/>

# Siton

**A fast, fluent Python backtesting framework for cryptocurrency trading strategies.**

Compose strategies from 25+ technical indicators with a chainable API.
Evaluate tens of thousands of parameter combinations in seconds — powered by Numba JIT on all CPU cores.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/siton?color=orange)](https://pypi.org/project/siton/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

</div>

---

## Table of Contents

- [Why Siton?](#why-siton)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [The SDK](#the-sdk)
  - [Signals](#signals)
  - [Composition Operators](#composition-operators)
  - [Strategy Class](#strategy-class)
  - [Indicator Reference](#indicator-reference)
- [Example Strategies](#example-strategies)
- [CLI Reference](#cli-reference)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why Siton?

Most backtesting frameworks force you to write loops, manage state, and wrestle with vectorization. Siton takes a different approach:

```python
# Describe what you want, not how to compute it
trend   = ema_cross(fast=[8, 13], slow=[55, 89])
confirm = macd_signal(fast=[12], slow=[26], signal_period=[9])
gate    = adx(periods=[14], thresholds=[20, 25])

STRATEGY = Strategy(
    name="Trend+Momentum",
    entry=trend & confirm & gate,
    stop_loss=[2.0, 3.0],
    take_profit=[6.0, 10.0],
)
```

Run it, and Siton exhaustively backtests every parameter combination and ranks results:

```
$ siton my_strategy.py --demo

Strategy: Trend+Momentum  |  Combos: 16  |  Bars: 10 000  |  Data: synthetic

 Rank  ema_fast  ema_slow  adx_threshold  stop_loss  take_profit  Return%   Sharpe  Drawdown%  WinRate%  Trades  PF
    1        13        55             20        2.0         10.0    +84.3    1.82      -18.4      54.1      91  1.71
    2         8        55             20        2.0         10.0    +79.1    1.74      -19.2      52.8      88  1.64
    3        13        89             25        3.0         10.0    +71.4    1.61      -22.1      51.9      73  1.58
   ...

Buy & Hold: +42.3%  |  Best Alpha: +42.0%
```

**Key benefits:**

| Feature | Detail |
|---|---|
| Fluent API | Chain indicators with `&`, `\|`, `~`, `.filter_by()`, `.confirm()` |
| Parallel engine | Numba JIT, `prange` across all CPU cores |
| Walk-forward validation | Expanding IS windows, PSR/DSR overfitting detection |
| 25+ indicators | Trend, momentum, oscillators, volume, volatility, candlestick patterns |
| ATR-adaptive stops | Position sizing that adapts to market volatility |
| Flexible data | CCXT (100+ exchanges), CSV, or built-in synthetic OHLCV generator |

---

## Quick Start

```bash
# 1. Install TA-Lib C library (required)
sudo apt-get install libta-lib-dev   # Ubuntu/Debian
brew install ta-lib                  # macOS

# 2. Install siton
pip install siton

# 3. Run a built-in example on synthetic data
siton examples/01_ema_crossover.py --demo
```

That's it. No API keys, no data files needed to try it out.

---

## Installation

### Prerequisites

Siton requires the **TA-Lib C library** to be installed system-wide before `pip install`.

| Platform | Command |
|---|---|
| Ubuntu / Debian | `sudo apt-get install libta-lib-dev` |
| macOS | `brew install ta-lib` |
| Windows | See [TA-Lib Windows install guide](https://github.com/ta-lib/ta-lib-python#windows) |
| Other | See [TA-Lib dependencies](https://github.com/ta-lib/ta-lib-python#dependencies) |

### Install from PyPI

```bash
pip install siton
```

### Install from Source (for development)

```bash
git clone https://github.com/enricopesce/siton.git
cd siton
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Dependencies

- Python ≥ 3.10
- [polars](https://pola.rs/) ≥ 1.0 — fast DataFrame operations
- [numpy](https://numpy.org/) ≥ 2.0 — array math
- [numba](https://numba.readthedocs.io/) ≥ 0.60 — JIT compilation
- [ta-lib](https://github.com/ta-lib/ta-lib-python) ≥ 0.6 — technical indicators
- [ccxt](https://docs.ccxt.com/) ≥ 4.0 — exchange data

---

## The SDK

### Signals

Every indicator function returns a `Signal` object. Signals are lazy — they carry a parameter grid and only compute when the engine runs a backtest.

```python
from siton.sdk import *

# Each argument is a list → defines the parameter grid
ema = ema_cross(fast=[8, 13, 21], slow=[55, 89, 144])  # 3×3 = 9 combos
rsi_sig = rsi(periods=[14], oversold=[25, 30], overbought=[70, 75])  # 1×2×2 = 4 combos
```

### Composition Operators

| Operator | Syntax | Behavior |
|---|---|---|
| AND | `a & b` | Both must agree (long+long=long, else flat) |
| OR | `a \| b` | First non-zero signal wins; conflict → flat |
| NOT-filter | `~a` | Active (1) where `a` is flat; use as gate |
| Filter by | `a.filter_by(b)` | `a` passes only where `b` is non-zero |
| Agree | `a.agree(b)` | `a` passes where `b` agrees on direction |
| Confirm | `a.confirm(b, n)` | `a` valid if `b` fired same direction in last `n` bars |
| Majority | `Signal.majority(a, b, c)` | Takes the majority direction across signals |
| Negate | `a.negate()` | Flips long ↔ short |

```python
# Gate EMA trend by ADX strength, require RSI not extreme
entry = ema_cross(fast=[8, 13], slow=[55, 89]) \
    .filter_by(adx(periods=[14], thresholds=[20, 25])) \
    .agree(rsi(periods=[14], oversold=[30], overbought=[70]))

# Majority vote across three independent signals
vote = Signal.majority(
    ema_cross(fast=[8], slow=[21]),
    macd_signal(fast=[12], slow=[26], signal_period=[9]),
    obv(ma_periods=[20]),
)
```

### Strategy Class

```python
STRATEGY = Strategy(
    name="MyStrategy",

    # --- Mode 1: single signal (position tracks signal) ---
    signal=entry,

    # --- Mode 2: separate entry + exit signals ---
    entry=entry_signal,
    exit=exit_signal,

    # --- Fixed % stops ---
    stop_loss=[1.5, 2.0, 3.0],    # % of entry price
    take_profit=[5.0, 10.0, 15.0],

    # --- ATR-normalized stops (alternative to fixed %) ---
    atr_period=14,
    atr_stop_mult=[1.0, 1.5, 2.0],
    atr_tp_mult=[2.0, 3.0, 5.0],

    # --- Execution ---
    fee=0.075,        # taker fee %
    slippage=0.05,    # slippage %
    capital=10_000,   # initial capital

    # --- Result filtering ---
    top=10,           # show top N combos
    sort="sharpe",    # rank by: sharpe | return | sortino | calmar | profit_factor

    # --- Walk-forward validation ---
    validate=True,
    train_ratio=0.7,
    n_splits=5,
)
```

### Indicator Reference

#### Trend / Moving Average Crossovers

| Function | Parameters | Description |
|---|---|---|
| `ema_cross()` | `fast`, `slow` | Exponential MA crossover |
| `sma_cross()` | `fast`, `slow` | Simple MA crossover |
| `dema_cross()` | `fast`, `slow` | Double EMA crossover |
| `tema_cross()` | `fast`, `slow` | Triple EMA crossover |
| `wma_cross()` | `fast`, `slow` | Weighted MA crossover |
| `kama_cross()` | `fast`, `slow` | Kaufman Adaptive MA crossover |
| `trima_cross()` | `fast`, `slow` | Triangular MA crossover |

#### Oscillators

| Function | Parameters | Description |
|---|---|---|
| `rsi()` | `periods`, `oversold`, `overbought` | Relative Strength Index |
| `cmo()` | `periods`, `oversold`, `overbought` | Chande Momentum Oscillator |
| `stoch_rsi()` | `periods`, `fastk`, `fastd`, `oversold`, `overbought` | Stochastic RSI |
| `cci()` | `periods`, `oversold`, `overbought` | Commodity Channel Index |
| `willr()` | `periods`, `oversold`, `overbought` | Williams %R |
| `mfi()` | `periods`, `oversold`, `overbought` | Money Flow Index |
| `ultimate_osc()` | `p1`, `p2`, `p3`, `oversold`, `overbought` | Ultimate Oscillator |

#### Momentum

| Function | Parameters | Description |
|---|---|---|
| `macd_signal()` | `fast`, `slow`, `signal_period` | MACD line vs signal line |
| `apo()` | `fast`, `slow` | Absolute Price Oscillator |
| `ppo()` | `fast`, `slow` | Percentage Price Oscillator |
| `mom()` | `periods` | Momentum (zero-line cross) |
| `roc()` | `periods` | Rate of Change |
| `trix()` | `periods` | 1-day ROC of triple-smooth EMA |

#### Trend Strength

| Function | Parameters | Description |
|---|---|---|
| `adx()` | `periods`, `thresholds` | Average Directional Index (gate) |
| `adxr()` | `periods`, `thresholds` | Smoothed ADX |
| `dx()` | `periods`, `thresholds` | Directional Index |
| `di_cross()` | `periods` | +DI / -DI crossover |
| `dm_cross()` | `periods` | +DM / -DM crossover |

#### Volume

| Function | Parameters | Description |
|---|---|---|
| `obv()` | `ma_periods` | On-Balance Volume vs MA |
| `ad()` | `ma_periods` | Accumulation/Distribution vs MA |
| `adosc()` | `fast`, `slow` | A/D Oscillator |

#### Volatility

| Function | Parameters | Description |
|---|---|---|
| `bollinger_bands()` | `windows`, `num_std` | Mean-reversion on band touches |
| `atr_breakout()` | `periods`, `multipliers` | Breakout on expanding ATR |
| `natr_breakout()` | `periods`, `multipliers` | Normalized ATR breakout |
| `channel_breakout()` | `periods` | Highest high / lowest low breakout |

#### Exit Signals

| Function | Parameters | Description |
|---|---|---|
| `sar()` | `acceleration`, `maximum` | Parabolic SAR crossover |
| `sar_ext()` | `start_long`, `inc_long`, `max_long`, `start_short`, `inc_short`, `max_short` | Asymmetric SAR |

#### Candlestick Patterns

```python
# Any TA-Lib CDL* pattern by name
eng  = candlestick_pattern("CDLENGULFING")
doji = candlestick_pattern("CDLDOJI")
```

#### Escape Hatches

```python
# Wrap any external factory
my_signal = indicator(my_factory_fn)

# Define a raw signal from scratch
my_signal = custom(
    name="my_indicator",
    grid={"period": [10, 20], "threshold": [0.5, 1.0]},
    fn=lambda close, high, low, volume, period, threshold: ...
)
```

---

## Example Strategies

The `examples/` directory contains six progressively complex strategies:

| File | Level | Combos | Concept |
|---|---|---|---|
| [`01_ema_crossover.py`](examples/01_ema_crossover.py) | Beginner | 4 | Simple EMA crossover |
| [`02_rsi_filtered_trend.py`](examples/02_rsi_filtered_trend.py) | Easy | 32 | EMA trend filtered by RSI |
| [`03_trend_momentum_confluence.py`](examples/03_trend_momentum_confluence.py) | Intermediate | 16 | EMA + MACD + ADX confluence |
| [`04_entry_exit_with_stops.py`](examples/04_entry_exit_with_stops.py) | Advanced | 55 296 | Regime-adaptive: trend vs mean-reversion arms |
| [`05_entry_exit_with_atr_stops.py`](examples/05_entry_exit_with_atr_stops.py) | Advanced | 55 296 | Same as above, ATR-normalized stops |
| [`05_multi_strategy_walkforward.py`](examples/05_multi_strategy_walkforward.py) | Expert | varies | Multi-strategy walk-forward comparison |

Run any example with:

```bash
siton examples/01_ema_crossover.py --demo           # synthetic data
siton examples/04_entry_exit_with_stops.py -s BTC/USDT -t 4h --start 2023-01-01
```

---

## CLI Reference

```
siton <strategy_file> [options]

Data source (pick one):
  --demo              Use built-in synthetic OHLCV data (10 000 bars)
  --csv PATH          Load from CSV file
  -s, --symbol        Trading pair  (default: BTC/USDT)
  -e, --exchange      CCXT exchange (default: binance)
  -t, --timeframe     Candle size   (default: 1h)
  --start DATE        ISO date, e.g. 2023-01-01
  --end DATE          ISO date (optional)
  -n, --limit         Max candles to fetch

Output:
  --top N             Show top N results (default: 10)
  --sort METRIC       Sort by: sharpe | return | sortino | calmar | profit_factor

Validation:
  --validate          Enable walk-forward validation
  --train-ratio       Fraction of data for training (default: 0.7)
```

---

## Architecture

```
siton/
├── sdk.py       ← Strategy composition (Signal, Strategy, ~25 indicators)
├── engine.py    ← Numba-JIT backtesting engine (parallel prange)
├── cli.py       ← CLI entry point (argparse, display, walk-forward)
└── data.py      ← Data loading (CCXT, CSV, synthetic GARCH generator)
```

**Data flow:**

```
CLI (cli.py)
 └─ loads strategy file (importlib)
 └─ fetches OHLCV data (data.py)
     └─ SDK expands parameter grid (sdk.py)
         └─ generates signal arrays for every combo
             └─ engine backtests all combos in parallel (engine.py)
                 └─ ranks & displays results (cli.py)
```

**Signal internals:**

Every indicator is a **factory closure** returning `(name, grid, signal_fn)`. The `Signal` class wraps factories with Python operators (`&`, `|`, `~`). When the engine runs, it calls the factory chain once per combo — TA-Lib calls are **memoized** so shared sub-signals are computed only once.

**Engine internals:**

Two Numba-JIT kernels, both parallelised with `prange`:

- `backtest_batch` — no stops, signal-following only
- `backtest_batch_managed` — stop-loss / take-profit / trailing stop
- `backtest_batch_atr_managed` — ATR-normalized stops

Returns 8 metrics per combo (return, Sharpe, drawdown, win-rate, trades, profit factor, Sortino, Calmar). Equity curves generated on demand for the top result.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

**Quick contribution guide:**

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes
3. Verify with `siton examples/01_ema_crossover.py --demo` (no test framework yet — see roadmap)
4. Open a PR describing what you changed and why

**Good first issues:**

- Add a new indicator (see any existing one in `sdk.py` as a template)
- Improve CLI output formatting
- Write unit tests (test framework setup is on the roadmap)
- Add support for a new data format

**Areas that need help:**

- [ ] Pytest test suite
- [ ] Equity curve plotting (`matplotlib` / `plotly`)
- [ ] Portfolio-level multi-asset backtesting
- [ ] Live trading integration (paper trading mode)
- [ ] Windows native TA-Lib install helper

---

## Roadmap

- [ ] **v0.2** — Pytest suite, equity curve plots, pip release
- [ ] **v0.3** — Portfolio mode (multiple assets simultaneously)
- [ ] **v0.4** — Paper trading / live signal generation
- [ ] **v1.0** — Stable API, full docs site

---

## License

[MIT](LICENSE) © 2024 Siton Contributors
