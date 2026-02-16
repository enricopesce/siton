# Siton

Fast crypto strategy backtester. Test thousands of trading strategy parameter combinations against OHLCV data and rank them by Sharpe ratio, return, drawdown, and more.

**120 strategies | 1,800+ parameter combinations | 10,000 candles in < 1s**

## Performance Stack

| Layer | Tool | Role |
|-------|------|------|
| Indicators | TA-Lib (C library) | SMA, EMA, RSI, MACD, 61 candlestick patterns, etc. |
| Backtesting | Numba `@njit(parallel=True)` | Single-pass engine across all CPU cores |
| Caching | Memoized `_cached()` | Each unique indicator computation runs exactly once |
| Data loading | Polars | Fast DataFrame I/O, converted to numpy once |

## Prerequisites

TA-Lib C library must be installed before `pip install`:

```bash
# Ubuntu / Debian
sudo apt install libta-lib-dev

# macOS
brew install ta-lib

# RHEL / Oracle Linux / Fedora
sudo yum install ta-lib-devel
# or
sudo dnf install ta-lib-devel

# From source (any Linux)
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
tar -xzf ta-lib-0.6.4-src.tar.gz && cd ta-lib-0.6.4
./configure --prefix=/usr && make && sudo make install
```

## Setup

```bash
git clone <repo-url> && cd siton
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Python >= 3.10** required. Dependencies: `polars`, `numpy`, `numba`, `ta-lib`, `ccxt`.

## Usage

```bash
# Synthetic data (no API keys needed)
siton --demo

# Live exchange data
siton -s BTC/USDT -t 1h -n 5000

# From CSV file (expects columns: timestamp,open,high,low,close,volume)
siton --csv data.csv --fee 0.1

# Custom output
siton --demo --top 20 --sort total_return_pct

# Run without installing
python -m siton --demo
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --symbol` | `BTC/USDT` | Trading pair |
| `-t, --timeframe` | `1h` | Candle timeframe |
| `-e, --exchange` | `binance` | Exchange (any ccxt-supported) |
| `-n, --limit` | `5000` | Number of candles to fetch |
| `--csv` | — | Load from CSV instead of exchange |
| `--demo` | — | Use synthetic data (no API needed) |
| `--fee` | `0.075` | Fee per trade in % |
| `--top` | `10` | Show top N results |
| `--sort` | `sharpe_ratio` | Sort by: `sharpe_ratio`, `total_return_pct`, `profit_factor`, `win_rate_pct` |

## Architecture

```
siton/
├── __init__.py       Package marker + version
├── cli.py            CLI entry point (argparse) — orchestrates the pipeline
├── data.py           Data ingestion (exchange via ccxt, CSV, synthetic generator)
├── engine.py         Numba JIT backtesting engine (zero changes needed to add strategies)
├── indicators.py     Memoized TA-Lib wrapper + ~35 signal pattern factories
└── strategies.py     Strategy registry — one line per strategy
```

### Data Pipeline

```
Load data (Polars DataFrame)
    │
    ▼
Extract OHLCV dict ──► { open, high, low, close, volume } as float64 numpy arrays
    │
    ▼
Generate signals ──► Each strategy factory produces a signal_fn(data, **params) → {-1, 0, 1}
    │                  TA-Lib indicator results are memoized via _cached()
    ▼
Batch backtest ──► Numba @njit(parallel=True) across all CPU cores
    │               Single-pass per combo: Welford's online variance, zero array allocations
    ▼
Rank & display ──► Sort by chosen metric, print top N
```

### Signal Contract

Every signal function follows this contract:

```python
def signal_fn(data: dict[str, np.ndarray], **params) -> np.ndarray:
    """
    Args:
        data: {"open": arr, "high": arr, "low": arr, "close": arr, "volume": arr}
              Each value is a float64 numpy array of length N.
        **params: Strategy-specific parameters from the grid.

    Returns:
        float64 array of length N with values in {-1.0, 0.0, 1.0}
        where 1 = long, -1 = short, 0 = flat.
    """
```

### Backtest Engine

The engine (`engine.py`) knows nothing about indicators or strategies. It only consumes:
- `close`: 1D float64 array of close prices
- `signals_2d`: 2D float64 array (N_combos x N_bars) of pre-computed signals
- `fee`: decimal fee rate

This means **you never need to modify the engine** to add new strategies.

## Strategy Coverage

### 120 Strategies Across 8 Categories

| Category | Strategies | Combos | Examples |
|----------|-----------|--------|----------|
| MA Crossovers | 7 | 314 | SMA, EMA, DEMA, TEMA, WMA, KAMA, TRIMA |
| Oscillators | 10 | 813 | RSI, CMO, StochRSI, CCI, WILLR, MFI, ULTOSC, STOCH, STOCHF |
| MACD Family | 5 | 205 | MACD, MACDEXT, MACDFIX, APO, PPO |
| Momentum | 9 | 53 | MOM, ROC, ROCP, ROCR, ROCR100, TRIX, LR_Slope, LR_Angle |
| Trend / Direction | 10 | 106 | ADX, ADXR, DX, DM_Cross, DI_Cross, SAR, SAREXT, AROON, AROONOSC |
| Volatility / Breakout | 4 | 59 | Bollinger, Channel Breakout, ATR Breakout, NATR Breakout |
| Hilbert Transform + Overlays | 8 | 21 | HT_Sine, HT_Phasor, HT_TrendMode, MIDPOINT, LINEARREG, TSF, MAMA |
| Candlestick Patterns | 61 | 82 | All 61 CDL functions (7 with penetration parameter grids) |
| Volume | 3 | 26 | OBV, AD, ADOSC |
| Price Transforms | 5 | 5 | BOP, AVGPRICE, TYPPRICE, WCLPRICE, MEDPRICE |
| **Total** | **120** | **1,802** | |

## Contributing

### Adding a New Strategy

1. **Pick a factory** from `indicators.py` that matches your indicator type (see table below).
2. **Add one line** to `ALL_STRATEGIES` in `strategies.py`.
3. **Run `siton --demo`** to verify it works.

```python
# Example: add a T3 MA crossover
crossover("T3_Cross", talib.T3, fast=[5, 10, 20], slow=[50, 100])
```

### Factory Reference

| Factory | Input | Use Case | Example |
|---------|-------|----------|---------|
| `crossover` | close | Two MAs compared | SMA, EMA, DEMA, TEMA, WMA, KAMA, TRIMA, T3 |
| `threshold` | close | Oscillator vs oversold/overbought | RSI, CMO |
| `zero_cross` | close | Single-param indicator vs zero | MOM, ROC, TRIX, ROCP |
| `dual_zero_cross` | close | Two-param oscillator vs zero | APO, PPO |
| `macd` | close | MACD line vs signal line | MACD |
| `macdext` | close | MACD with MA type param | MACDEXT |
| `macdfix` | close | MACD fixed 12/26 | MACDFIX |
| `stochrsi` | close | Stochastic RSI thresholds | StochRSI |
| `bollinger` | close | Price vs Bollinger Bands | Bollinger |
| `price_crossover` | close | Close vs overlay indicator | MIDPOINT, LINEARREG, TSF, HT_TRENDLINE |
| `mama_crossover` | close | MAMA vs FAMA | MAMA |
| `ratio_cross` | close | Indicator vs center value | ROCR (1.0), ROCR100 (100.0) |
| `cycle_crossover` | close | HT_SINE sine vs leadsine | HT_Sine |
| `phasor_crossover` | close | HT_PHASOR inphase vs quadrature | HT_Phasor |
| `ht_trendmode` | close | Trend mode + trendline filter | HT_TrendMode |
| `breakout` | close | Channel breakout (MAX/MIN) | Channel Breakout |
| `aroon_crossover` | high, low | AroonUp vs AroonDown | AROON |
| `zero_cross_hl` | high, low | HL indicator vs zero | AROONOSC |
| `price_crossover_hl` | high, low | Close vs HL overlay | MIDPRICE |
| `dm_crossover` | high, low | PLUS_DM vs MINUS_DM | DM_Cross |
| `sar_crossover` | high, low | Close vs SAR | SAR |
| `sarext_crossover` | high, low | Close vs SAREXT | SAREXT |
| `threshold_hlc` | high, low, close | HLC oscillator thresholds | CCI, WILLR |
| `threshold_hlcv` | high, low, close, vol | HLCV oscillator thresholds | MFI |
| `trend_strength` | high, low, close | ADX/DX + SMA direction | ADX, ADXR, DX |
| `ultosc_threshold` | high, low, close | 3-period Ultimate Oscillator | ULTOSC |
| `stoch_hlc` | high, low, close | Full Stochastic | STOCH |
| `stochf_hlc` | high, low, close | Fast Stochastic | STOCHF |
| `di_crossover` | high, low, close | PLUS_DI vs MINUS_DI | DI_Cross |
| `volatility_breakout` | high, low, close | ATR expansion + direction | ATR, NATR |
| `candlestick` | open, high, low, close | CDL pattern to -1/0/1 | Any CDL function |
| `bop_signal` | open, high, low, close | BOP zero cross | BOP |
| `price_transform_crossover` | configurable | Close vs price transform | AVGPRICE, TYPPRICE, WCLPRICE, MEDPRICE |
| `obv_crossover` | close, volume | OBV vs SMA(OBV) | OBV |
| `ad_crossover` | high, low, close, vol | AD vs SMA(AD) | AD |
| `adosc_zero_cross` | high, low, close, vol | ADOSC zero cross | ADOSC |

### Creating a New Factory

If no existing factory fits your indicator, add one to `indicators.py`:

```python
def my_factory(name, some_params):
    """Describe the signal logic."""
    def factory():
        grid = {"param": some_params}

        def signal_fn(data, param):
            close = data["close"]
            ind = _cached(talib.SOME_FUNC, close, timeperiod=param)
            sig = np.where(ind > 0, 1.0, np.where(ind < 0, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory
```

Key rules:
- **Return shape**: signal array must be same length as input, values in `{-1.0, 0.0, 1.0}`
- **NaN handling**: always zero out NaN regions (warmup period of indicators)
- **Use `_cached()`**: wrap all TA-Lib calls for memoization
- **Use `_tag`**: if passing non-default inputs to a function (e.g. `SMA(obv)` vs `SMA(close)`), add `_tag="obv_ma"` to disambiguate in cache
- **Guard invalid combos**: return `np.zeros()` early for nonsensical params (e.g. `fast >= slow`)

### Project Structure Rules

- **`engine.py`** should never be modified to add strategies. It only consumes signals.
- **`indicators.py`** contains factory functions and the cache. No strategy definitions.
- **`strategies.py`** contains only strategy registrations. No logic.
- **`data.py`** handles all data ingestion. Returns Polars DataFrames.
- **`cli.py`** orchestrates the pipeline. Minimal logic.

### Indicator Cache

The `_cached()` function memoizes TA-Lib calls by `(function_name, _tag, **kwargs)`:

```python
# These are cached separately:
_cached(talib.SMA, close, timeperiod=20)              # key: ('SMA', None, ('timeperiod', 20))
_cached(talib.SMA, obv, _tag="obv_ma", timeperiod=20) # key: ('SMA', 'obv_ma', ('timeperiod', 20))
```

The cache is cleared between runs via `clear_cache()` in `cli.py`.

## TA-Lib Indicators Not Mapped (and Why)

| Category | Indicators | Reason |
|----------|-----------|--------|
| Math transforms | ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH | Pure math, no trading signal |
| Index outputs | MAXINDEX, MININDEX, MINMAXINDEX | Returns bar indices, not values |
| Two-series arithmetic | ADD, SUB, MULT, DIV | Requires two input series |
| Statistical | BETA, CORREL, STDDEV, VAR, LINEARREG_INTERCEPT | Not directly tradeable |
| Variable period | MAVP | Requires per-bar period array |
| Cycle analytics | HT_DCPERIOD, HT_DCPHASE | Output is cycle length/phase, no direction |
| Redundant | MA (use SMA/EMA via matype), SUM, TRANGE (use ATR) | Covered by other strategies |

## License

MIT
