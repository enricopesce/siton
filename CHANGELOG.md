# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Six progressive example strategies (`01_ema_crossover` → `05_multi_strategy_walkforward`)
- ATR-normalized stop-loss / take-profit / trailing stop (`backtest_batch_atr_managed`)
- Walk-forward validation with expanding IS windows and PSR/DSR overfitting detection
- `Signal.majority()` — majority-vote composition across N signals
- `signal.confirm(b, lookbacks)` — require confirming signal fired in last N bars
- `signal.agree(b)` — directional agreement filter
- `sar_ext()` — asymmetric Parabolic SAR with separate long/short parameters
- `candlestick_pattern(name)` — generic wrapper for all TA-Lib CDL* functions
- `custom()` — escape hatch for fully user-defined signal functions
- Sortino ratio and Calmar ratio in result output
- Equity curve generation for top result
- `--validate` / `--train-ratio` CLI flags for walk-forward mode
- GitHub Actions CI workflow (lint + smoke tests on Python 3.10–3.12)
- `CONTRIBUTING.md`, issue templates, PR template

### Changed
- Consolidated `indicators.py`, `composites.py`, `strategies.py` → single `sdk.py`
- Renamed example files to numbered convention (`01_`, `02_`, ...)
- `pyproject.toml`: added PyPI classifiers, keywords, optional dev dependencies

### Fixed
- Welford online variance uses N-1 denominator for sample Sharpe ratio
- Memoized TA-Lib cache disambiguates by function name + args id to prevent cross-indicator collisions

---

## [0.1.0] - 2024-01-01

### Added
- Initial release — fluent SDK, Numba-JIT parallel engine, CCXT data loading, synthetic OHLCV generator
