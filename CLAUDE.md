# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Siton is a fast crypto strategy backtester CLI that tests thousands of trading strategy parameter combinations against OHLCV candle data and ranks them by performance metrics (Sharpe ratio, return, drawdown, etc.).

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run with synthetic data (no API keys needed)
siton examples/complex_strategy.py --demo

# Or run without installing
python -m siton examples/complex_strategy.py --demo

# Run against live exchange data
siton my_strategy.py -s BTC/USDT -t 1h -n 5000

# Run from CSV
siton my_strategy.py --csv data.csv
```

Execution parameters (fee, slippage, capital, fraction, top, sort) are defined
in the Strategy file, not on the CLI.

No test suite or linter is configured.

## Architecture

The codebase is a Python package (`siton/`) with five modules and a clear data pipeline:

1. **`siton/data.py`** — Data ingestion. Three sources: live exchange via ccxt (`fetch_ohlcv` with automatic pagination), CSV file (`load_csv`), or synthetic random-walk generator (`generate_sample`). All return a Polars DataFrame with columns: `timestamp, open, high, low, close, volume`.

2. **`siton/indicators.py`** — Memoized TA-Lib wrapper and signal pattern factories. Caches indicator computations via `_cached()` so each unique `(function, tag, params)` tuple is computed once. Provides ~35 factory functions covering close-only, HL, HLC, HLCV, OHLC, and volume indicator patterns. The `_tag` parameter disambiguates calls feeding different inputs to the same TA-Lib function (e.g. `SMA(close)` vs `SMA(OBV)`).

3. **`siton/strategies.py`** — Strategy registry. One line per strategy using the factory functions from `indicators.py`. All strategies are collected in `ALL_STRATEGIES`. Currently 120 strategies (~1,800 parameter combinations) covering MA crossovers, oscillators, momentum, MACD variants, Hilbert Transform cycles, directional movement, Stochastic, SAR, volatility breakouts, candlestick patterns, volume indicators, and price transforms.

4. **`siton/engine.py`** — Numba JIT backtesting engine with realistic portfolio model. Tracks `cash + shares` (signed: positive=long, negative=short) with per-trade slippage, fee, and fixed-fraction position sizing. `_backtest_one` is a `@njit(cache=True)` single-pass backtest using Welford's online variance (zero array allocations). `_backtest_one_managed` adds stop-loss, take-profit, and trailing stop. `backtest_batch` / `backtest_batch_managed` are `@njit(parallel=True, cache=True)` distributing combos across CPU cores via `nb.prange`. JIT is warmed up at import time.

5. **`siton/cli.py`** — CLI entry point (argparse). Orchestrates: load data → extract full OHLCV dict to numpy once → generate all signals → batch backtest (Numba parallel) → rank and display top N results.

## Key Conventions

- **Signal contract**: Strategy signal functions receive a `data` dict with keys `open, high, low, close, volume` (each a float64 `np.ndarray`) and return a `np.ndarray` of float64 values in `{-1, 0, 1}` with the same length.
- **Strategy registration**: Add new strategies using the factory functions in `indicators.py` and append to `ALL_STRATEGIES` in `strategies.py`.
- **Performance stack**: TA-Lib (C) for indicators, Numba `@njit(parallel=True)` for backtest engine, Polars for data loading only. OHLCV arrays extracted to numpy once and shared across all strategy/backtest calls. Indicator results are memoized.
- **Portfolio model**: Engine tracks `cash` + `shares` (signed) per backtest. Equity = `cash + shares * price`. Supports initial capital, fixed-fraction sizing, slippage, and fee — all configured on the `Strategy` object. Blown-account protection exits the loop if equity <= 0.
- **Sharpe ratio assumes hourly candles** (annualized with `sqrt(8760)`).
