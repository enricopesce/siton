# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Siton is a Python framework for backtesting trading strategies on historical cryptocurrency data. It provides a fluent, chainable SDK for composing strategies from technical indicators, with a Numba-JIT parallel backtesting engine that evaluates thousands of parameter combinations in seconds.

## Prerequisites

The TA-Lib C library must be installed before `pip install`:
- Ubuntu/Debian: `sudo apt-get install libta-lib-dev`
- macOS: `brew install ta-lib`

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running

```bash
# CLI entry point (installed via pyproject.toml [project.scripts])
siton examples/01_ema_crossover.py --demo
siton examples/01_ema_crossover.py -s BTC/USDT -t 1h --start 2024-01-01

# Or run strategy files directly
python examples/01_ema_crossover.py --demo

# Or as module
python -m siton examples/01_ema_crossover.py --demo
```

## Architecture

The data flow is: **CLI loads strategy file → loads OHLCV data → SDK expands parameter grid → generates signal arrays → engine backtests all combos in parallel → ranks results**.

### Core Modules

- **`siton/sdk.py`** — The entire SDK in one file. Contains:
  - `Signal` class: wraps a factory callable `() -> (name, grid, signal_fn)` with chainable operators (`&`, `|`, `~`, `.filter_by()`, `.agree()`, `.confirm()`, `.where_flat()`, `Signal.majority()`)
  - ~25 indicator constructors (e.g. `ema_cross`, `rsi`, `adx`, `sar`) that return `Signal` objects
  - `Strategy` class: binds signals (either `signal=` or `entry=`/`exit=` pair) with position management params (stop_loss, take_profit, trailing_stop)
  - `backtest()`: CLI-oriented runner that parses args and prints results
  - `run()`: programmatic API that returns `Result` objects
  - Grid merge system (`_merge_flat`) handles parameter name collisions when composing signals

- **`siton/engine.py`** — Numba-JIT backtesting engine. Two paths:
  - `backtest_batch`: simple signal-following (no stops), parallelized with `nb.prange`
  - `backtest_batch_managed`: adds stop-loss, take-profit, trailing stop; also parallelized
  - Both warm up JIT at import time with dummy data
  - Returns 6 metrics per combo: total_return, sharpe, max_drawdown, win_rate, num_trades, profit_factor

- **`siton/cli.py`** — CLI entry point (`main()`). Dynamically imports strategy files via `importlib`, expects them to define a `STRATEGY` variable.

- **`siton/data.py`** — Data loading: `fetch_ohlcv` (ccxt with pagination), `load_csv`, `generate_sample` (synthetic). All return Polars DataFrames with columns: timestamp, open, high, low, close, volume.

### Key Design Patterns

- **Factory pattern**: Every indicator is a closure returning `(name, grid, signal_fn)`. The `Signal` class wraps these with composition operators. `_labeled_factory` prefixes grid keys with the indicator name (e.g. `ema_fast`, `adx_period`) so composed grids remain flat and human-readable.
- **Memoized caching**: `_cached()` memoizes TA-Lib calls during signal generation. `_sdk_caches` registry tracks per-composition caches. Both are cleared between runs via `clear_cache()` / `clear_sdk_cache()`.
- **Signals are ternary**: All signal arrays use `{-1.0, 0.0, 1.0}` (short, flat, long).
- **Grid expansion**: `expand_grid()` produces the cartesian product of all parameter lists. Composition merges grids flat — the total combo count is the product of all leaf indicator grid sizes.
- **Sharpe annualization**: `_ann_factor(timeframe)` computes `sqrt(periods_per_year)` for proper Sharpe ratio scaling across timeframes (1m through 1w). Uses sample standard deviation (Welford's algorithm with N-1 denominator).

## Testing

No test framework is configured yet. Run example strategies with `--demo` to verify behavior.

## Dependencies

Python >=3.10, polars, numpy (>=2.0), ccxt, ta-lib, numba. Defined in `pyproject.toml`.
