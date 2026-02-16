"""Indicator wrapper — memoized TA-Lib calls + signal pattern factories.

Provides factory functions that eliminate boilerplate when defining strategies.
Each factory returns a callable that produces (name, grid, signal_fn) tuples.

Signal contract: every signal_fn receives a ``data`` dict with keys
``open, high, low, close, volume`` (each a float64 np.ndarray) and returns
a float64 array in {-1, 0, 1} with the same length.

Supported patterns:
    crossover           — Two MAs compared (SMA, EMA, DEMA, TEMA, WMA, KAMA, TRIMA, T3)
    threshold           — Indicator vs oversold/overbought levels (RSI, CMO)
    zero_cross          — Single-param indicator vs zero line (MOM, ROC, TRIX, ROCP, etc.)
    dual_zero_cross     — Two-param oscillator vs zero line (APO, PPO)
    macd                — MACD line vs signal line
    stochrsi            — Stochastic RSI with threshold levels
    bollinger           — Price vs Bollinger Bands
    price_crossover     — Close vs overlay indicator (MIDPOINT, LINEARREG, TSF, HT_TRENDLINE)
    mama_crossover      — MAMA vs FAMA crossover
    ratio_cross         — Indicator crosses a center value (ROCR, ROCR100)
    macdext             — MACD with configurable MA type
    macdfix             — MACD with fixed 12/26 periods
    cycle_crossover     — HT_SINE sine vs leadsine crossover
    phasor_crossover    — HT_PHASOR inphase vs quadrature
    ht_trendmode        — HT_TRENDMODE + HT_TRENDLINE combined
    breakout            — Channel breakout using MAX/MIN with 1-bar shift
    zero_cross_hl       — Zero-line cross for HL indicators (AROONOSC)
    aroon_crossover     — AroonUp vs AroonDown
    sar_crossover       — Close vs SAR
    sarext_crossover    — Close vs SAREXT
    dm_crossover        — PLUS_DM vs MINUS_DM
    threshold_hlc       — HLC oscillator threshold (CCI, WILLR)
    threshold_hlcv      — HLC+V oscillator threshold (MFI)
    trend_strength      — ADX/DX + SMA direction filter
    ultosc_threshold    — 3-period ultimate oscillator
    stoch_hlc           — Full Stochastic
    stochf_hlc          — Fast Stochastic
    di_crossover        — PLUS_DI vs MINUS_DI
    volatility_breakout — High ATR + price direction
    price_crossover_hl  — Close vs HL overlay (MIDPRICE)
    candlestick         — CDL patterns mapped to -1/0/1
    bop_signal          — BOP zero cross
    price_transform_crossover — Close vs AVGPRICE/TYPPRICE/etc
    obv_crossover       — OBV vs SMA(OBV)
    ad_crossover        — AD vs SMA(AD)
    adosc_zero_cross    — ADOSC zero cross
"""

import numpy as np
import talib
from itertools import product


# ---------------------------------------------------------------------------
# Memoized indicator cache
# ---------------------------------------------------------------------------

_cache = {}


def clear_cache():
    """Clear indicator cache. Call between runs."""
    _cache.clear()


def _cached(func, *args, _tag=None, **kwargs):
    """Memoized TA-Lib call (single or multi-output).

    ``_tag`` disambiguates calls that feed different inputs to the same
    function (e.g. ``SMA(close)`` vs ``SMA(OBV)``).
    """
    key = (func.__name__, _tag) + tuple(sorted(kwargs.items()))
    if key not in _cache:
        _cache[key] = func(*args, **kwargs)
    return _cache[key]


def expand_grid(grid: dict) -> list[dict]:
    """Expand parameter grid into list of param dicts."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ---------------------------------------------------------------------------
# Phase 1 — Existing factories (updated to accept ``data`` dict)
# ---------------------------------------------------------------------------

def crossover(name, ta_fn, fast, slow):
    """MA crossover: long when fast > slow, short when fast < slow."""
    def factory():
        grid = {"fast": fast, "slow": slow}

        def signal_fn(data, fast, slow):
            close = data["close"]
            if fast >= slow:
                return np.zeros(len(close))
            f = _cached(ta_fn, close, timeperiod=fast)
            s = _cached(ta_fn, close, timeperiod=slow)
            sig = np.where(f > s, 1.0, np.where(f < s, -1.0, 0.0))
            sig[np.isnan(s)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def threshold(name, ta_fn, periods, oversold, overbought):
    """Mean reversion: long when indicator < oversold, short when > overbought."""
    def factory():
        grid = {"period": periods, "oversold": oversold, "overbought": overbought}

        def signal_fn(data, period, oversold, overbought):
            close = data["close"]
            ind = _cached(ta_fn, close, timeperiod=period)
            sig = np.where(ind < oversold, 1.0, np.where(ind > overbought, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def zero_cross(name, ta_fn, periods):
    """Momentum: long when indicator > 0, short when < 0."""
    def factory():
        grid = {"period": periods}

        def signal_fn(data, period):
            close = data["close"]
            ind = _cached(ta_fn, close, timeperiod=period)
            sig = np.where(ind > 0, 1.0, np.where(ind < 0, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def dual_zero_cross(name, ta_fn, fast, slow):
    """Two-param oscillator zero-line cross: long when > 0, short when < 0."""
    def factory():
        grid = {"fast": fast, "slow": slow}

        def signal_fn(data, fast, slow):
            close = data["close"]
            if fast >= slow:
                return np.zeros(len(close))
            ind = _cached(ta_fn, close, fastperiod=fast, slowperiod=slow)
            sig = np.where(ind > 0, 1.0, np.where(ind < 0, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def macd(name, fast, slow, signal_period):
    """MACD: long when MACD line > signal line, short when below."""
    def factory():
        grid = {"fast": fast, "slow": slow, "signal_period": signal_period}

        def signal_fn(data, fast, slow, signal_period):
            close = data["close"]
            if fast >= slow:
                return np.zeros(len(close))
            macd_line, sig_line, _ = _cached(
                talib.MACD, close, fastperiod=fast, slowperiod=slow, signalperiod=signal_period)
            sig = np.where(macd_line > sig_line, 1.0, np.where(macd_line < sig_line, -1.0, 0.0))
            sig[np.isnan(sig_line)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def stochrsi(name, periods, fastk, fastd, oversold, overbought):
    """Stochastic RSI: long when %D < oversold, short when > overbought."""
    def factory():
        grid = {"period": periods, "fastk": fastk, "fastd": fastd,
                "oversold": oversold, "overbought": overbought}

        def signal_fn(data, period, fastk, fastd, oversold, overbought):
            close = data["close"]
            _, d = _cached(
                talib.STOCHRSI, close, timeperiod=period, fastk_period=fastk, fastd_period=fastd)
            sig = np.where(d < oversold, 1.0, np.where(d > overbought, -1.0, 0.0))
            sig[np.isnan(d)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def bollinger(name, windows, num_std):
    """Bollinger Bands mean reversion: long below lower band, short above upper."""
    def factory():
        grid = {"window": windows, "num_std": num_std}

        def signal_fn(data, window, num_std):
            close = data["close"]
            upper, _, lower = _cached(
                talib.BBANDS, close, timeperiod=window, nbdevup=num_std, nbdevdn=num_std)
            sig = np.where(close < lower, 1.0, np.where(close > upper, -1.0, 0.0))
            sig[np.isnan(upper)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


# ---------------------------------------------------------------------------
# Phase 2 — New close-only factories
# ---------------------------------------------------------------------------

def price_crossover(name, ta_fn, periods):
    """Close vs overlay indicator: long when close > indicator, short when below."""
    def factory():
        grid = {"period": periods} if periods else {}

        def signal_fn(data, period=None):
            close = data["close"]
            if period is not None:
                ind = _cached(ta_fn, close, timeperiod=period)
            else:
                ind = _cached(ta_fn, close)
            sig = np.where(close > ind, 1.0, np.where(close < ind, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def mama_crossover(name, fast_limits, slow_limits):
    """MAMA vs FAMA: long when MAMA > FAMA, short when below."""
    def factory():
        grid = {"fast_limit": fast_limits, "slow_limit": slow_limits}

        def signal_fn(data, fast_limit, slow_limit):
            close = data["close"]
            mama_val, fama_val = _cached(
                talib.MAMA, close, fastlimit=fast_limit, slowlimit=slow_limit)
            sig = np.where(mama_val > fama_val, 1.0, np.where(mama_val < fama_val, -1.0, 0.0))
            sig[np.isnan(fama_val)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def ratio_cross(name, ta_fn, periods, center):
    """Indicator crosses a center value: long when > center, short when < center."""
    def factory():
        grid = {"period": periods}

        def signal_fn(data, period):
            close = data["close"]
            ind = _cached(ta_fn, close, timeperiod=period)
            sig = np.where(ind > center, 1.0, np.where(ind < center, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def macdext(name, fast, slow, signal_period, ma_types):
    """MACD with configurable MA type."""
    def factory():
        grid = {"fast": fast, "slow": slow, "signal_period": signal_period, "ma_type": ma_types}

        def signal_fn(data, fast, slow, signal_period, ma_type):
            close = data["close"]
            if fast >= slow:
                return np.zeros(len(close))
            macd_line, sig_line, _ = _cached(
                talib.MACDEXT, close,
                fastperiod=fast, fastmatype=ma_type,
                slowperiod=slow, slowmatype=ma_type,
                signalperiod=signal_period, signalmatype=ma_type)
            sig = np.where(macd_line > sig_line, 1.0, np.where(macd_line < sig_line, -1.0, 0.0))
            sig[np.isnan(sig_line)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def macdfix(name, signal_periods):
    """MACD with fixed 12/26 periods, variable signal period."""
    def factory():
        grid = {"signal_period": signal_periods}

        def signal_fn(data, signal_period):
            close = data["close"]
            macd_line, sig_line, _ = _cached(
                talib.MACDFIX, close, signalperiod=signal_period)
            sig = np.where(macd_line > sig_line, 1.0, np.where(macd_line < sig_line, -1.0, 0.0))
            sig[np.isnan(sig_line)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def cycle_crossover(name):
    """HT_SINE: long when sine > leadsine, short when below."""
    def factory():
        grid = {}

        def signal_fn(data):
            close = data["close"]
            sine, leadsine = _cached(talib.HT_SINE, close)
            sig = np.where(sine > leadsine, 1.0, np.where(sine < leadsine, -1.0, 0.0))
            sig[np.isnan(leadsine)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def phasor_crossover(name):
    """HT_PHASOR: long when inphase > quadrature, short when below."""
    def factory():
        grid = {}

        def signal_fn(data):
            close = data["close"]
            inphase, quadrature = _cached(talib.HT_PHASOR, close)
            sig = np.where(inphase > quadrature, 1.0, np.where(inphase < quadrature, -1.0, 0.0))
            sig[np.isnan(quadrature)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def ht_trendmode(name):
    """HT_TRENDMODE + HT_TRENDLINE: trend mode direction with trendline filter."""
    def factory():
        grid = {}

        def signal_fn(data):
            close = data["close"]
            mode = _cached(talib.HT_TRENDMODE, close)
            trendline = _cached(talib.HT_TRENDLINE, close)
            # mode=1 means trending: use trendline direction
            sig = np.where(
                mode == 1,
                np.where(close > trendline, 1.0, -1.0),
                0.0,
            )
            sig[np.isnan(trendline)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def breakout(name, periods):
    """Channel breakout: long when close > prev MAX, short when < prev MIN."""
    def factory():
        grid = {"period": periods}

        def signal_fn(data, period):
            close = data["close"]
            upper = _cached(talib.MAX, close, timeperiod=period)
            lower = _cached(talib.MIN, close, timeperiod=period)
            # Shift by 1 to avoid lookahead
            upper_shifted = np.empty_like(upper)
            lower_shifted = np.empty_like(lower)
            upper_shifted[0] = np.nan
            lower_shifted[0] = np.nan
            upper_shifted[1:] = upper[:-1]
            lower_shifted[1:] = lower[:-1]
            sig = np.where(close > upper_shifted, 1.0, np.where(close < lower_shifted, -1.0, 0.0))
            sig[np.isnan(upper_shifted)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


# ---------------------------------------------------------------------------
# Phase 3 — HL / HLC / HLCV factories
# ---------------------------------------------------------------------------

def zero_cross_hl(name, ta_fn, periods):
    """Zero-line cross for HL indicators (AROONOSC)."""
    def factory():
        grid = {"period": periods}

        def signal_fn(data, period):
            high, low = data["high"], data["low"]
            ind = _cached(ta_fn, high, low, timeperiod=period)
            sig = np.where(ind > 0, 1.0, np.where(ind < 0, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def aroon_crossover(name, periods):
    """AroonUp vs AroonDown: long when up > down, short when below."""
    def factory():
        grid = {"period": periods}

        def signal_fn(data, period):
            high, low = data["high"], data["low"]
            down, up = _cached(talib.AROON, high, low, timeperiod=period)
            sig = np.where(up > down, 1.0, np.where(up < down, -1.0, 0.0))
            sig[np.isnan(up)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def price_crossover_hl(name, ta_fn, periods):
    """Close vs HL overlay indicator (MIDPRICE): long when close > ind."""
    def factory():
        grid = {"period": periods}

        def signal_fn(data, period):
            high, low, close = data["high"], data["low"], data["close"]
            ind = _cached(ta_fn, high, low, timeperiod=period)
            sig = np.where(close > ind, 1.0, np.where(close < ind, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def sar_crossover(name, accelerations, maximums):
    """Close vs SAR: long when close > SAR, short when below."""
    def factory():
        grid = {"acceleration": accelerations, "maximum": maximums}

        def signal_fn(data, acceleration, maximum):
            high, low, close = data["high"], data["low"], data["close"]
            sar = _cached(talib.SAR, high, low, acceleration=acceleration, maximum=maximum)
            sig = np.where(close > sar, 1.0, np.where(close < sar, -1.0, 0.0))
            sig[np.isnan(sar)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def sarext_crossover(name, accel_inits, accelerations, maximums):
    """Close vs SAREXT: symmetric long/short params."""
    def factory():
        grid = {"accel_init": accel_inits, "acceleration": accelerations, "maximum": maximums}

        def signal_fn(data, accel_init, acceleration, maximum):
            high, low, close = data["high"], data["low"], data["close"]
            sar = _cached(
                talib.SAREXT, high, low,
                startvalue=0.0,
                offsetonreverse=0.0,
                accelerationinitlong=accel_init,
                accelerationlong=acceleration,
                accelerationmaxlong=maximum,
                accelerationinitshort=accel_init,
                accelerationshort=acceleration,
                accelerationmaxshort=maximum,
            )
            sig = np.where(close > sar, 1.0, np.where(close < sar, -1.0, 0.0))
            sig[np.isnan(sar)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def dm_crossover(name, periods):
    """PLUS_DM vs MINUS_DM: long when plus > minus, short when below."""
    def factory():
        grid = {"period": periods}

        def signal_fn(data, period):
            high, low = data["high"], data["low"]
            plus = _cached(talib.PLUS_DM, high, low, timeperiod=period)
            minus = _cached(talib.MINUS_DM, high, low, timeperiod=period)
            sig = np.where(plus > minus, 1.0, np.where(plus < minus, -1.0, 0.0))
            sig[np.isnan(plus)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def threshold_hlc(name, ta_fn, periods, oversold, overbought):
    """HLC oscillator with threshold: long < oversold, short > overbought."""
    def factory():
        grid = {"period": periods, "oversold": oversold, "overbought": overbought}

        def signal_fn(data, period, oversold, overbought):
            high, low, close = data["high"], data["low"], data["close"]
            ind = _cached(ta_fn, high, low, close, timeperiod=period)
            sig = np.where(ind < oversold, 1.0, np.where(ind > overbought, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def threshold_hlcv(name, ta_fn, periods, oversold, overbought):
    """HLCV oscillator with threshold (MFI): long < oversold, short > overbought."""
    def factory():
        grid = {"period": periods, "oversold": oversold, "overbought": overbought}

        def signal_fn(data, period, oversold, overbought):
            high, low, close, volume = data["high"], data["low"], data["close"], data["volume"]
            ind = _cached(ta_fn, high, low, close, volume, timeperiod=period)
            sig = np.where(ind < oversold, 1.0, np.where(ind > overbought, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def trend_strength(name, ta_fn, periods, thresholds):
    """ADX/DX trend strength: trade in trend direction when indicator > threshold."""
    def factory():
        grid = {"period": periods, "threshold": thresholds}

        def signal_fn(data, period, threshold):
            high, low, close = data["high"], data["low"], data["close"]
            ind = _cached(ta_fn, high, low, close, timeperiod=period)
            # SMA of close for direction
            sma = _cached(talib.SMA, close, _tag="trend_dir", timeperiod=period)
            direction = np.where(close > sma, 1.0, np.where(close < sma, -1.0, 0.0))
            sig = np.where(ind > threshold, direction, 0.0)
            sig[np.isnan(ind) | np.isnan(sma)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def ultosc_threshold(name, p1_list, p2_list, p3_list, oversold, overbought):
    """Ultimate Oscillator with 3 periods and threshold levels."""
    def factory():
        grid = {"p1": p1_list, "p2": p2_list, "p3": p3_list,
                "oversold": oversold, "overbought": overbought}

        def signal_fn(data, p1, p2, p3, oversold, overbought):
            high, low, close = data["high"], data["low"], data["close"]
            ind = _cached(talib.ULTOSC, high, low, close,
                          timeperiod1=p1, timeperiod2=p2, timeperiod3=p3)
            sig = np.where(ind < oversold, 1.0, np.where(ind > overbought, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def stoch_hlc(name, fastk_periods, slowk_periods, slowd_periods, oversold, overbought):
    """Full Stochastic: long when %D < oversold, short when > overbought."""
    def factory():
        grid = {"fastk": fastk_periods, "slowk": slowk_periods, "slowd": slowd_periods,
                "oversold": oversold, "overbought": overbought}

        def signal_fn(data, fastk, slowk, slowd, oversold, overbought):
            high, low, close = data["high"], data["low"], data["close"]
            _, d = _cached(
                talib.STOCH, high, low, close,
                fastk_period=fastk, slowk_period=slowk, slowd_period=slowd)
            sig = np.where(d < oversold, 1.0, np.where(d > overbought, -1.0, 0.0))
            sig[np.isnan(d)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def stochf_hlc(name, fastk_periods, fastd_periods, oversold, overbought):
    """Fast Stochastic: long when %D < oversold, short when > overbought."""
    def factory():
        grid = {"fastk": fastk_periods, "fastd": fastd_periods,
                "oversold": oversold, "overbought": overbought}

        def signal_fn(data, fastk, fastd, oversold, overbought):
            high, low, close = data["high"], data["low"], data["close"]
            _, d = _cached(
                talib.STOCHF, high, low, close,
                fastk_period=fastk, fastd_period=fastd)
            sig = np.where(d < oversold, 1.0, np.where(d > overbought, -1.0, 0.0))
            sig[np.isnan(d)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def di_crossover(name, periods):
    """PLUS_DI vs MINUS_DI: long when plus > minus, short when below."""
    def factory():
        grid = {"period": periods}

        def signal_fn(data, period):
            high, low, close = data["high"], data["low"], data["close"]
            plus = _cached(talib.PLUS_DI, high, low, close, timeperiod=period)
            minus = _cached(talib.MINUS_DI, high, low, close, timeperiod=period)
            sig = np.where(plus > minus, 1.0, np.where(plus < minus, -1.0, 0.0))
            sig[np.isnan(plus)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def volatility_breakout(name, ta_fn, periods, lookbacks):
    """High volatility + price direction: trade when ATR/NATR is expanding."""
    def factory():
        grid = {"period": periods, "lookback": lookbacks}

        def signal_fn(data, period, lookback):
            high, low, close = data["high"], data["low"], data["close"]
            atr = _cached(ta_fn, high, low, close, timeperiod=period)
            atr_ma = _cached(talib.SMA, atr, _tag=f"atr_ma_{ta_fn.__name__}", timeperiod=lookback)
            direction = np.where(close > np.roll(close, lookback), 1.0,
                                 np.where(close < np.roll(close, lookback), -1.0, 0.0))
            sig = np.where(atr > atr_ma, direction, 0.0)
            sig[np.isnan(atr_ma)] = 0.0
            # Zero out initial lookback bars where np.roll wraps around
            sig[:lookback] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


# ---------------------------------------------------------------------------
# Phase 4 — OHLC / OHLCV / Volume factories
# ---------------------------------------------------------------------------

def candlestick(name, ta_fn, penetration_values=None):
    """CDL pattern: maps -100/0/100 output to -1/0/1."""
    def factory():
        if penetration_values:
            grid = {"penetration": penetration_values}
        else:
            grid = {}

        def signal_fn(data, penetration=None):
            o, h, l, c = data["open"], data["high"], data["low"], data["close"]
            if penetration is not None:
                raw = ta_fn(o, h, l, c, penetration=penetration)
            else:
                raw = ta_fn(o, h, l, c)
            sig = np.where(raw > 0, 1.0, np.where(raw < 0, -1.0, 0.0))
            return sig

        return name, grid, signal_fn
    return factory


def bop_signal(name):
    """BOP (Balance of Power): long when > 0, short when < 0."""
    def factory():
        grid = {}

        def signal_fn(data):
            o, h, l, c = data["open"], data["high"], data["low"], data["close"]
            ind = _cached(talib.BOP, o, h, l, c)
            sig = np.where(ind > 0, 1.0, np.where(ind < 0, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def price_transform_crossover(name, ta_fn, input_keys):
    """Close vs price transform (AVGPRICE, TYPPRICE, etc.): long when close > transform."""
    def factory():
        grid = {}

        def signal_fn(data):
            args = [data[k] for k in input_keys]
            ind = _cached(ta_fn, *args)
            close = data["close"]
            sig = np.where(close > ind, 1.0, np.where(close < ind, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def obv_crossover(name, ma_periods):
    """OBV vs SMA(OBV): long when OBV > SMA(OBV), short when below."""
    def factory():
        grid = {"ma_period": ma_periods}

        def signal_fn(data, ma_period):
            close, volume = data["close"], data["volume"]
            obv = _cached(talib.OBV, close, volume)
            obv_ma = _cached(talib.SMA, obv, _tag="obv_ma", timeperiod=ma_period)
            sig = np.where(obv > obv_ma, 1.0, np.where(obv < obv_ma, -1.0, 0.0))
            sig[np.isnan(obv_ma)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def ad_crossover(name, ma_periods):
    """AD vs SMA(AD): long when AD > SMA(AD), short when below."""
    def factory():
        grid = {"ma_period": ma_periods}

        def signal_fn(data, ma_period):
            high, low, close, volume = data["high"], data["low"], data["close"], data["volume"]
            ad_val = _cached(talib.AD, high, low, close, volume)
            ad_ma = _cached(talib.SMA, ad_val, _tag="ad_ma", timeperiod=ma_period)
            sig = np.where(ad_val > ad_ma, 1.0, np.where(ad_val < ad_ma, -1.0, 0.0))
            sig[np.isnan(ad_ma)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory


def adosc_zero_cross(name, fast_periods, slow_periods):
    """ADOSC: long when > 0, short when < 0."""
    def factory():
        grid = {"fast": fast_periods, "slow": slow_periods}

        def signal_fn(data, fast, slow):
            high, low, close, volume = data["high"], data["low"], data["close"], data["volume"]
            if fast >= slow:
                return np.zeros(len(close))
            ind = _cached(talib.ADOSC, high, low, close, volume,
                          fastperiod=fast, slowperiod=slow)
            sig = np.where(ind > 0, 1.0, np.where(ind < 0, -1.0, 0.0))
            sig[np.isnan(ind)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory
