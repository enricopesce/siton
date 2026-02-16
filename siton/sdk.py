"""Strategy SDK — fluent API for building composite strategies in ~20 lines.

Usage::

    from siton.sdk import *

    trend = ema_cross(fast=[8, 12], slow=[26, 50]).filter_by(
        adx(periods=[14], thresholds=[20, 25, 30])
    )
    confirmed = trend & obv(ma_periods=[20, 50])
    safe = confirmed.filter_by(~rsi(periods=[14], oversold=[25], overbought=[75]))

    STRATEGY = Strategy("TrendRegimeSniper",
        entry=safe,
        exit=sar(accelerations=[0.02, 0.03], maximums=[0.2])
    )

    if __name__ == "__main__":
        backtest(STRATEGY)  # python my_strat.py --demo

Parameter names in the output are human-readable::

    ema_fast=12, ema_slow=50, adx_period=14, adx_threshold=30, ...
"""

import argparse
import itertools
import time
import warnings

import numpy as np
import talib

from siton.indicators import (
    crossover as _ind_crossover,
    threshold as _ind_threshold,
    zero_cross as _ind_zero_cross,
    dual_zero_cross as _ind_dual_zero_cross,
    macd as _ind_macd,
    stochrsi as _ind_stochrsi,
    bollinger as _ind_bollinger,
    breakout as _ind_breakout,
    sar_crossover as _ind_sar,
    sarext_crossover as _ind_sarext,
    obv_crossover as _ind_obv,
    ad_crossover as _ind_ad,
    adosc_zero_cross as _ind_adosc,
    trend_strength as _ind_trend_strength,
    di_crossover as _ind_di_crossover,
    dm_crossover as _ind_dm_crossover,
    threshold_hlc as _ind_threshold_hlc,
    threshold_hlcv as _ind_threshold_hlcv,
    ultosc_threshold as _ind_ultosc,
    volatility_breakout as _ind_volatility_breakout,
    candlestick as _ind_candlestick,
    expand_grid, clear_cache,
)
from siton.composites import (
    sig_and, sig_or, sig_not, sig_filter, sig_agree, sig_confirm,
    sig_majority, sig_entry_exit,
    _warn_grid_size,
)
from siton.engine import backtest_batch, backtest_batch_managed, Result, rank_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_list(v):
    """Normalize a scalar or list to a list."""
    if v is None:
        return v
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


_signal_counter = 0

def _next_id():
    global _signal_counter
    _signal_counter += 1
    return _signal_counter


# Sub-signal memoization registry — caches are created inside factory
# closures and registered here so they can be bulk-cleared after use.
_sdk_caches = []

def _register_cache(*caches):
    _sdk_caches.extend(caches)

def clear_sdk_cache():
    """Free all memoized sub-signal arrays."""
    for c in _sdk_caches:
        c.clear()
    _sdk_caches.clear()


def _labeled_factory(inner_factory, label):
    """Wrap a factory so its grid keys become ``{label}_{key}``.

    This is the core trick: each leaf indicator gets human-readable
    prefixed keys (e.g. ``ema_fast``, ``adx_period``), and composition
    just merges them flat — no prefix stacking.
    """
    prefix = label + "_"
    def factory():
        name, grid, fn = inner_factory()
        labeled_grid = {prefix + k: v for k, v in grid.items()}

        def labeled_fn(data, **params):
            inner_params = {k[len(prefix):]: v
                           for k, v in params.items() if k.startswith(prefix)}
            return fn(data, **inner_params)

        return name, labeled_grid, labeled_fn
    return factory


def _merge_flat(grid_l, grid_r):
    """Merge two grids. On key collision, append _2, _3, ... to right keys.

    Returns (merged_grid, rename_map) where rename_map maps
    new_key -> old_key for any renamed right keys.
    """
    collision = set(grid_l) & set(grid_r)
    if not collision:
        return {**grid_l, **grid_r}, {}

    rename = {}
    for k in grid_r:
        if k in collision:
            for i in range(2, 100):
                new_k = f"{k}_{i}"
                if new_k not in grid_l and new_k not in rename.values():
                    rename[k] = new_k
                    break
        # non-colliding keys keep their name
    grid_r_final = {rename.get(k, k): v for k, v in grid_r.items()}
    # reverse: new_key -> old_key
    reverse = {v: k for k, v in rename.items()}
    return {**grid_l, **grid_r_final}, reverse


def _compose_binary(combine_fn, left, right, label):
    """Compose two Signals with flat grid merge (no prefix stacking).

    Memoizes left/right sub-signals so shared parameter combos are computed once.
    """
    def factory():
        _, grid_l, fn_l = left._factory()
        _, grid_r, fn_r = right._factory()

        keys_l = frozenset(grid_l)
        merged, reverse_map = _merge_flat(grid_l, grid_r)
        keys_r = frozenset(merged) - keys_l

        _warn_grid_size(label, merged)

        sorted_keys_l = sorted(keys_l)
        sorted_keys_r = sorted(keys_r)
        cache_l = {}
        cache_r = {}
        _register_cache(cache_l, cache_r)

        def signal_fn(data, **params):
            key_l = tuple(params[k] for k in sorted_keys_l)
            sig_l = cache_l.get(key_l)
            if sig_l is None:
                pl = {k: params[k] for k in keys_l}
                sig_l = fn_l(data, **pl)
                cache_l[key_l] = sig_l

            key_r = tuple(params[k] for k in sorted_keys_r)
            sig_r = cache_r.get(key_r)
            if sig_r is None:
                pr = {}
                for k in keys_r:
                    orig_k = reverse_map.get(k, k)
                    pr[orig_k] = params[k]
                sig_r = fn_r(data, **pr)
                cache_r[key_r] = sig_r

            return combine_fn(sig_l, sig_r)

        return label, merged, signal_fn
    return Signal(factory, label=label)


def _compose_entry_exit(name, entry, exit_sig):
    """Entry/exit state machine with flat grid merge.

    Memoizes entry/exit sub-signals so shared parameter combos are computed once.
    """
    def factory():
        _, grid_e, fn_e = entry._factory()
        _, grid_x, fn_x = exit_sig._factory()

        keys_e = frozenset(grid_e)
        merged, reverse_map = _merge_flat(grid_e, grid_x)
        keys_x = frozenset(merged) - keys_e

        _warn_grid_size(name, merged)

        sorted_keys_e = sorted(keys_e)
        sorted_keys_x = sorted(keys_x)
        cache_e = {}
        cache_x = {}
        _register_cache(cache_e, cache_x)

        def signal_fn(data, **params):
            key_e = tuple(params[k] for k in sorted_keys_e)
            s_el = cache_e.get(key_e)
            if s_el is None:
                pe = {k: params[k] for k in keys_e}
                s_el = fn_e(data, **pe)
                cache_e[key_e] = s_el

            key_x = tuple(params[k] for k in sorted_keys_x)
            s_xl = cache_x.get(key_x)
            if s_xl is None:
                px = {}
                for k in keys_x:
                    orig_k = reverse_map.get(k, k)
                    px[orig_k] = params[k]
                s_xl = fn_x(data, **px)
                cache_x[key_x] = s_xl

            return sig_entry_exit(s_el, s_xl, s_xl, s_el)

        return name, merged, signal_fn
    return factory


# ---------------------------------------------------------------------------
# Signal class — wraps a factory callable with chainable operators
# ---------------------------------------------------------------------------

class Signal:
    """Wraps an indicator factory with chainable composition operators.

    A factory is a callable returning ``(name, grid, signal_fn)``.
    Each Signal carries a human-readable ``label`` used to prefix its
    parameter names in the output (e.g. ``ema``, ``adx``, ``rsi``).
    """

    def __init__(self, factory, label="sig"):
        self._factory = factory
        self._label = label

    def __call__(self):
        return self._factory()

    def __and__(self, other):
        """``a & b`` — both must agree."""
        return _compose_binary(sig_and, self, other,
                               label=f"{self._label}_{other._label}")

    def __or__(self, other):
        """``a | b`` — first non-zero; conflict=0."""
        return _compose_binary(sig_or, self, other,
                               label=f"{self._label}_{other._label}")

    def __invert__(self):
        """``~a`` — where_flat(): active where original is 0."""
        return self.where_flat()

    def negate(self):
        """Algebraic inversion: long becomes short, vice versa."""
        inner = self._factory
        label = self._label
        def factory():
            iname, grid, fn = inner()
            def negated_fn(data, **params):
                return sig_not(fn(data, **params))
            return iname, grid, negated_fn
        return Signal(factory, label=label)

    def filter_by(self, other):
        """Signal passes only where ``other`` is active."""
        return _compose_binary(sig_filter, self, other,
                               label=f"{self._label}_{other._label}")

    def agree(self, other):
        """Signal passes only where ``other`` agrees on direction."""
        return _compose_binary(sig_agree, self, other,
                               label=f"{self._label}_{other._label}")

    def confirm(self, other, lookbacks=(3, 5, 8)):
        """Signal valid only if ``other`` fired same direction in last N bars."""
        inner_self = self._factory
        inner_other = other._factory
        label = f"{self._label}_{other._label}"

        def factory():
            _, grid_s, fn_s = inner_self()
            _, grid_c, fn_c = inner_other()

            keys_s = frozenset(grid_s)
            merged, reverse_map = _merge_flat(grid_s, grid_c)
            keys_c = frozenset(merged) - keys_s

            merged["lookback"] = list(lookbacks)
            _warn_grid_size(label, merged)

            sorted_keys_s = sorted(keys_s)
            sorted_keys_c = sorted(keys_c)
            cache_s = {}
            cache_c = {}
            _register_cache(cache_s, cache_c)

            def signal_fn(data, lookback, **params):
                key_s = tuple(params[k] for k in sorted_keys_s)
                ss = cache_s.get(key_s)
                if ss is None:
                    ps = {k: params[k] for k in keys_s}
                    ss = fn_s(data, **ps)
                    cache_s[key_s] = ss

                key_c = tuple(params[k] for k in sorted_keys_c)
                sc = cache_c.get(key_c)
                if sc is None:
                    pc = {}
                    for k in keys_c:
                        orig_k = reverse_map.get(k, k)
                        pc[orig_k] = params[k]
                    sc = fn_c(data, **pc)
                    cache_c[key_c] = sc

                return sig_confirm(ss, sc, lookback)

            return label, merged, signal_fn
        return Signal(factory, label=label)

    def where_flat(self):
        """Binary filter: 1 where original signal is 0 (flat), 0 otherwise.

        Useful as ``~rsi(...)`` to mean 'NOT in overbought/oversold'.
        """
        inner = self._factory
        label = self._label
        def factory():
            iname, grid, fn = inner()
            def flat_fn(data, **params):
                return np.where(fn(data, **params) == 0.0, 1.0, 0.0)
            return iname, grid, flat_fn
        return Signal(factory, label=label)

    @staticmethod
    def majority(*signals):
        """Majority vote across multiple signals."""
        label = "_".join(s._label for s in signals)
        factories = [s._factory for s in signals]

        def factory():
            fns = []
            merged = {}
            key_sets = []
            reverse_maps = []

            for f in factories:
                _, grid, fn = f()
                new_merged, rev = _merge_flat(merged, grid)
                new_keys = frozenset(new_merged) - frozenset(merged)
                merged = new_merged
                key_sets.append(new_keys)
                reverse_maps.append(rev)
                fns.append(fn)

            _warn_grid_size(label, merged)

            sorted_key_sets = [sorted(ks) for ks in key_sets]
            caches = [{} for _ in fns]
            _register_cache(*caches)

            def signal_fn(data, **params):
                sigs = []
                for fn, sorted_keys, rev, cache in zip(
                        fns, sorted_key_sets, reverse_maps, caches):
                    key = tuple(params[k] for k in sorted_keys)
                    sig = cache.get(key)
                    if sig is None:
                        p = {}
                        for k in sorted_keys:
                            orig_k = rev.get(k, k)
                            p[orig_k] = params[k]
                        sig = fn(data, **p)
                        cache[key] = sig
                    sigs.append(sig)
                return sig_majority(*sigs)

            return label, merged, signal_fn
        return Signal(factory, label=label)


# ---------------------------------------------------------------------------
# Signal constructors — ~25 named builders
# ---------------------------------------------------------------------------

# ---- Trend: MA crossovers ----

def ema_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _ind_crossover(f"_ema{_next_id()}", talib.EMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "ema"), label="ema")

def sma_cross(fast=(10, 20), slow=(50, 100)):
    return Signal(_labeled_factory(
        _ind_crossover(f"_sma{_next_id()}", talib.SMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "sma"), label="sma")

def dema_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _ind_crossover(f"_dema{_next_id()}", talib.DEMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "dema"), label="dema")

def tema_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _ind_crossover(f"_tema{_next_id()}", talib.TEMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "tema"), label="tema")

def wma_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _ind_crossover(f"_wma{_next_id()}", talib.WMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "wma"), label="wma")

def kama_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _ind_crossover(f"_kama{_next_id()}", talib.KAMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "kama"), label="kama")

def trima_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _ind_crossover(f"_trima{_next_id()}", talib.TRIMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "trima"), label="trima")


# ---- Oscillators ----

def rsi(periods=(14,), oversold=(30,), overbought=(70,)):
    return Signal(_labeled_factory(
        _ind_threshold(f"_rsi{_next_id()}", talib.RSI,
                       periods=_ensure_list(periods),
                       oversold=_ensure_list(oversold),
                       overbought=_ensure_list(overbought)),
        "rsi"), label="rsi")

def cmo(periods=(14,), oversold=(-50,), overbought=(50,)):
    return Signal(_labeled_factory(
        _ind_threshold(f"_cmo{_next_id()}", talib.CMO,
                       periods=_ensure_list(periods),
                       oversold=_ensure_list(oversold),
                       overbought=_ensure_list(overbought)),
        "cmo"), label="cmo")

def stoch_rsi(periods=(14,), fastk=(3,), fastd=(3,),
              oversold=(20,), overbought=(80,)):
    return Signal(_labeled_factory(
        _ind_stochrsi(f"_stochrsi{_next_id()}",
                      periods=_ensure_list(periods),
                      fastk=_ensure_list(fastk),
                      fastd=_ensure_list(fastd),
                      oversold=_ensure_list(oversold),
                      overbought=_ensure_list(overbought)),
        "stochrsi"), label="stochrsi")

def cci(periods=(14,), oversold=(-100,), overbought=(100,)):
    return Signal(_labeled_factory(
        _ind_threshold_hlc(f"_cci{_next_id()}", talib.CCI,
                           periods=_ensure_list(periods),
                           oversold=_ensure_list(oversold),
                           overbought=_ensure_list(overbought)),
        "cci"), label="cci")

def willr(periods=(14,), oversold=(-80,), overbought=(-20,)):
    return Signal(_labeled_factory(
        _ind_threshold_hlc(f"_willr{_next_id()}", talib.WILLR,
                           periods=_ensure_list(periods),
                           oversold=_ensure_list(oversold),
                           overbought=_ensure_list(overbought)),
        "willr"), label="willr")

def mfi(periods=(14,), oversold=(20,), overbought=(80,)):
    return Signal(_labeled_factory(
        _ind_threshold_hlcv(f"_mfi{_next_id()}", talib.MFI,
                            periods=_ensure_list(periods),
                            oversold=_ensure_list(oversold),
                            overbought=_ensure_list(overbought)),
        "mfi"), label="mfi")

def ultimate_osc(p1=(7,), p2=(14,), p3=(28,), oversold=(30,), overbought=(70,)):
    return Signal(_labeled_factory(
        _ind_ultosc(f"_ultosc{_next_id()}",
                    p1_list=_ensure_list(p1),
                    p2_list=_ensure_list(p2),
                    p3_list=_ensure_list(p3),
                    oversold=_ensure_list(oversold),
                    overbought=_ensure_list(overbought)),
        "ultosc"), label="ultosc")


# ---- Momentum ----

def macd_signal(fast=(12,), slow=(26,), signal_period=(9,)):
    return Signal(_labeled_factory(
        _ind_macd(f"_macd{_next_id()}",
                  fast=_ensure_list(fast),
                  slow=_ensure_list(slow),
                  signal_period=_ensure_list(signal_period)),
        "macd"), label="macd")

def apo(fast=(12,), slow=(26,)):
    return Signal(_labeled_factory(
        _ind_dual_zero_cross(f"_apo{_next_id()}", talib.APO,
                             fast=_ensure_list(fast),
                             slow=_ensure_list(slow)),
        "apo"), label="apo")

def ppo(fast=(12,), slow=(26,)):
    return Signal(_labeled_factory(
        _ind_dual_zero_cross(f"_ppo{_next_id()}", talib.PPO,
                             fast=_ensure_list(fast),
                             slow=_ensure_list(slow)),
        "ppo"), label="ppo")

def mom(periods=(10,)):
    return Signal(_labeled_factory(
        _ind_zero_cross(f"_mom{_next_id()}", talib.MOM,
                        periods=_ensure_list(periods)),
        "mom"), label="mom")

def roc(periods=(10,)):
    return Signal(_labeled_factory(
        _ind_zero_cross(f"_roc{_next_id()}", talib.ROC,
                        periods=_ensure_list(periods)),
        "roc"), label="roc")

def trix(periods=(14,)):
    return Signal(_labeled_factory(
        _ind_zero_cross(f"_trix{_next_id()}", talib.TRIX,
                        periods=_ensure_list(periods)),
        "trix"), label="trix")


# ---- Trend strength ----

def adx(periods=(14,), thresholds=(25,)):
    return Signal(_labeled_factory(
        _ind_trend_strength(f"_adx{_next_id()}", talib.ADX,
                            periods=_ensure_list(periods),
                            thresholds=_ensure_list(thresholds)),
        "adx"), label="adx")

def adxr(periods=(14,), thresholds=(25,)):
    return Signal(_labeled_factory(
        _ind_trend_strength(f"_adxr{_next_id()}", talib.ADXR,
                            periods=_ensure_list(periods),
                            thresholds=_ensure_list(thresholds)),
        "adxr"), label="adxr")

def dx(periods=(14,), thresholds=(25,)):
    return Signal(_labeled_factory(
        _ind_trend_strength(f"_dx{_next_id()}", talib.DX,
                            periods=_ensure_list(periods),
                            thresholds=_ensure_list(thresholds)),
        "dx"), label="dx")

def di_cross(periods=(14,)):
    return Signal(_labeled_factory(
        _ind_di_crossover(f"_di{_next_id()}",
                          periods=_ensure_list(periods)),
        "di"), label="di")

def dm_cross(periods=(14,)):
    return Signal(_labeled_factory(
        _ind_dm_crossover(f"_dm{_next_id()}",
                          periods=_ensure_list(periods)),
        "dm"), label="dm")


# ---- Volume ----

def obv(ma_periods=(20,)):
    return Signal(_labeled_factory(
        _ind_obv(f"_obv{_next_id()}",
                 ma_periods=_ensure_list(ma_periods)),
        "obv"), label="obv")

def ad(ma_periods=(20,)):
    return Signal(_labeled_factory(
        _ind_ad(f"_ad{_next_id()}",
                ma_periods=_ensure_list(ma_periods)),
        "ad"), label="ad")

def adosc(fast=(3,), slow=(10,)):
    return Signal(_labeled_factory(
        _ind_adosc(f"_adosc{_next_id()}",
                   fast_periods=_ensure_list(fast),
                   slow_periods=_ensure_list(slow)),
        "adosc"), label="adosc")


# ---- Volatility ----

def bollinger_bands(windows=(20,), num_std=(2.0,)):
    return Signal(_labeled_factory(
        _ind_bollinger(f"_bb{_next_id()}",
                       windows=_ensure_list(windows),
                       num_std=_ensure_list(num_std)),
        "bb"), label="bb")

def atr_breakout(periods=(14,), lookbacks=(20,)):
    return Signal(_labeled_factory(
        _ind_volatility_breakout(f"_atrbrk{_next_id()}", talib.ATR,
                                 periods=_ensure_list(periods),
                                 lookbacks=_ensure_list(lookbacks)),
        "atr"), label="atr")

def natr_breakout(periods=(14,), lookbacks=(20,)):
    return Signal(_labeled_factory(
        _ind_volatility_breakout(f"_natrbrk{_next_id()}", talib.NATR,
                                 periods=_ensure_list(periods),
                                 lookbacks=_ensure_list(lookbacks)),
        "natr"), label="natr")

def channel_breakout(periods=(20,)):
    return Signal(_labeled_factory(
        _ind_breakout(f"_chan{_next_id()}",
                      periods=_ensure_list(periods)),
        "chan"), label="chan")


# ---- Exit signals ----

def sar(accelerations=(0.02,), maximums=(0.2,)):
    return Signal(_labeled_factory(
        _ind_sar(f"_sar{_next_id()}",
                 accelerations=_ensure_list(accelerations),
                 maximums=_ensure_list(maximums)),
        "sar"), label="sar")

def sar_ext(accel_inits=(0.02,), accelerations=(0.02,), maximums=(0.2,)):
    return Signal(_labeled_factory(
        _ind_sarext(f"_sarext{_next_id()}",
                    accel_inits=_ensure_list(accel_inits),
                    accelerations=_ensure_list(accelerations),
                    maximums=_ensure_list(maximums)),
        "sarext"), label="sarext")


# ---- Candlestick patterns ----

def candlestick_pattern(name, penetration_values=None):
    """Candlestick pattern by TA-Lib function name (e.g. 'CDLENGULFING')."""
    ta_fn = getattr(talib, name)
    pv = _ensure_list(penetration_values) if penetration_values else None
    label = name.lower().replace("cdl", "cdl_") if name.startswith("CDL") else name.lower()
    return Signal(_labeled_factory(
        _ind_candlestick(f"_cdl{_next_id()}", ta_fn, penetration_values=pv),
        label), label=label)


# ---- Escape hatches ----

def indicator(factory_fn, *args, label="ind", **kwargs):
    """Wrap any indicators.py factory call as a Signal."""
    return Signal(_labeled_factory(factory_fn(*args, **kwargs), label), label=label)


def custom(fn, label="custom", **grid):
    """Wrap a raw signal function with a parameter grid.

    ``fn`` receives ``(data, **params)`` and returns a float64 array in {-1,0,1}.
    Each kwarg is a list of values to sweep.
    """
    def factory():
        return f"_custom{_next_id()}", grid, fn
    return Signal(factory, label=label)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class Strategy:
    """Named strategy with either a direct signal or entry/exit pair.

    Usage::

        # Direct signal as position
        Strategy("MyStrat", signal=some_signal)

        # Entry/exit state machine
        Strategy("MyStrat", entry=entry_signal, exit=exit_signal)

        # With position management (swept as part of the backtest grid)
        Strategy("MyStrat", signal=some_signal,
                 stop_loss=[2, 3], take_profit=[5, 10])
    """

    def __init__(self, name, *, signal=None, entry=None, exit=None,
                 stop_loss=None, take_profit=None,
                 trailing_stop=None, fee=None, slippage=0.05,
                 capital=10000.0, fraction=1.0, top=10,
                 sort="sharpe_ratio"):
        self.name = name
        if signal is not None:
            if entry is not None or exit is not None:
                raise ValueError("Provide either 'signal' or 'entry'+'exit', not both")
            self._factory = signal._factory
        elif entry is not None and exit is not None:
            self._factory = _compose_entry_exit(name, entry, exit)
        else:
            raise ValueError("Provide either 'signal' or both 'entry' and 'exit'")

        self.stop_loss = _ensure_list(stop_loss)
        self.take_profit = _ensure_list(take_profit)
        self.trailing_stop = _ensure_list(trailing_stop)
        self.fee = fee
        self.slippage = slippage
        self.capital = capital
        self.fraction = fraction
        self.top = top
        self.sort = sort

    @property
    def has_managed(self):
        """True if any position management parameter is set."""
        return any(x is not None for x in (
            self.stop_loss, self.take_profit, self.trailing_stop))

    def as_factory(self):
        return self._factory

    def to_strategy_list(self):
        return [self._factory]


# ---------------------------------------------------------------------------
# Pipeline: backtest() and run()
# ---------------------------------------------------------------------------

def run(strategies, df_or_data, fee=None, slippage=None, capital=None,
        fraction=None, top=None, sort=None):
    """Programmatic API — backtest strategies against data.

    Args:
        strategies: Strategy or list of Strategy objects.
        df_or_data: Either a Polars DataFrame with OHLCV columns, or a dict
            with keys ``open, high, low, close, volume`` (numpy float64 arrays).
        fee: Fee per trade in percent. Overridden by ``Strategy.fee`` if set.
        slippage: Slippage per trade in percent.
        capital: Initial capital.
        fraction: Fraction of equity per trade, 0-1.
        top: Number of top results to return.
        sort: Metric to sort by.

    Strategy-level values take priority, then function params, then defaults.

    Returns:
        List of Result objects, sorted by the chosen metric.
    """
    if isinstance(strategies, Strategy):
        strategies = [strategies]

    # Strategy-level values take priority over function params
    strat = strategies[0]
    fee = strat.fee if strat.fee is not None else (fee if fee is not None else 0.075)
    slippage = slippage if slippage is not None else strat.slippage
    capital = capital if capital is not None else strat.capital
    fraction = fraction if fraction is not None else strat.fraction
    top = top if top is not None else strat.top
    sort = sort if sort is not None else strat.sort

    if isinstance(df_or_data, dict):
        data = df_or_data
    else:
        df = df_or_data
        data = {
            "open":   df["open"].to_numpy().astype(np.float64),
            "high":   df["high"].to_numpy().astype(np.float64),
            "low":    df["low"].to_numpy().astype(np.float64),
            "close":  df["close"].to_numpy().astype(np.float64),
            "volume": df["volume"].to_numpy().astype(np.float64),
        }
    close = data["close"]

    # Build combos and track which strategy each combo belongs to
    combos = []
    combo_strat = []  # index into strategies list per combo
    strat_lookup = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for si, strat in enumerate(strategies):
            factory = strat.as_factory()
            name, grid, signal_fn = factory()
            strat_lookup[name] = signal_fn
            for params in expand_grid(grid):
                combos.append((name, params))
                combo_strat.append(si)

    clear_cache()
    n_combos = len(combos)
    signals_2d = np.empty((n_combos, len(close)), dtype=np.float64)
    for i, (name, params) in enumerate(combos):
        signals_2d[i] = strat_lookup[name](data, **params)
    clear_cache()
    clear_sdk_cache()

    fee_dec = fee / 100.0
    slip_dec = slippage / 100.0
    use_managed = any(s.has_managed for s in strategies)

    if not use_managed:
        metrics_2d = backtest_batch(close, signals_2d, fee_dec, slip_dec, capital, fraction)
        results = []
        for i, (name, params) in enumerate(combos):
            m = metrics_2d[i]
            results.append(Result(
                strategy=name, params=params,
                total_return_pct=round(m[0], 2), sharpe_ratio=round(m[1], 3),
                max_drawdown_pct=round(m[2], 2), win_rate_pct=round(m[3], 2),
                num_trades=int(m[4]), profit_factor=round(m[5], 3),
            ))
    else:
        # Build per-strategy PM grids and expand
        high = data["high"]
        low = data["low"]
        expanded_indices = []
        expanded_combos = []

        for base_i, (name, params) in enumerate(combos):
            strat = strategies[combo_strat[base_i]]
            sl_vals = [v / 100.0 for v in strat.stop_loss] if strat.stop_loss else [0.0]
            tp_vals = [v / 100.0 for v in strat.take_profit] if strat.take_profit else [0.0]
            trail_vals = [v / 100.0 for v in strat.trailing_stop] if strat.trailing_stop else [0.0]

            for sl_v, tp_v, trail_v in itertools.product(sl_vals, tp_vals, trail_vals):
                expanded_indices.append((base_i, sl_v, tp_v, trail_v))
                pm_params = dict(params)
                if strat.stop_loss:
                    pm_params["stop_loss"] = round(sl_v * 100, 2)
                if strat.take_profit:
                    pm_params["take_profit"] = round(tp_v * 100, 2)
                if strat.trailing_stop:
                    pm_params["trailing_stop"] = round(trail_v * 100, 2)
                expanded_combos.append((name, pm_params))

        n_expanded = len(expanded_indices)
        sig_indices = np.empty(n_expanded, dtype=np.int64)
        sl_arr = np.empty(n_expanded, dtype=np.float64)
        tp_arr = np.empty(n_expanded, dtype=np.float64)
        trail_arr = np.empty(n_expanded, dtype=np.float64)

        for idx, (base_i, sl_v, tp_v, trail_v) in enumerate(expanded_indices):
            sig_indices[idx] = base_i
            sl_arr[idx] = sl_v
            tp_arr[idx] = tp_v
            trail_arr[idx] = trail_v

        metrics_2d = backtest_batch_managed(
            close, high, low, signals_2d, fee_dec, slip_dec,
            capital, fraction,
            sig_indices, sl_arr, tp_arr, trail_arr)

        results = []
        for i, (name, params) in enumerate(expanded_combos):
            m = metrics_2d[i]
            results.append(Result(
                strategy=name, params=params,
                total_return_pct=round(m[0], 2), sharpe_ratio=round(m[1], 3),
                max_drawdown_pct=round(m[2], 2), win_rate_pct=round(m[3], 2),
                num_trades=int(m[4]), profit_factor=round(m[5], 3),
            ))

    return rank_results(results, sort_by=sort)[:top]


def backtest(*strategies):
    """CLI entry point — parse args and run strategies.

    Supports data source flags: ``--demo``, ``--csv``, ``-s``, ``-t``, ``-e``, ``-n``.

    Execution parameters (fee, slippage, capital, fraction, top, sort) are
    defined on the Strategy object.
    """
    parser = argparse.ArgumentParser(
        description="Siton SDK — Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-s", "--symbol", default="BTC/USDT")
    parser.add_argument("-t", "--timeframe", default="1h")
    parser.add_argument("-e", "--exchange", default="binance")
    parser.add_argument("-n", "--limit", type=int, default=5000)
    parser.add_argument("--start", metavar="DATE", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", metavar="DATE", help="End date YYYY-MM-DD")
    parser.add_argument("--csv", help="Load OHLCV from CSV file")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    from siton.data import fetch_ohlcv, load_csv, generate_sample

    print("=" * 70)
    print("  SITON SDK — Strategy Backtester")
    print("=" * 70)
    t0 = time.perf_counter()

    if args.demo:
        print(f"\n[*] Generating synthetic data (10000 hourly candles)...")
        df = generate_sample(n=10000)
        print(f"    Simulated BTC-like price from ${df['close'][0]:.0f} to ${df['close'][-1]:.0f}")
    elif args.csv:
        print(f"\n[*] Loading data from {args.csv}...")
        df = load_csv(args.csv, start=args.start, end=args.end)
    else:
        date_desc = ""
        if args.start and args.end:
            date_desc = f" from {args.start} to {args.end}"
        elif args.start:
            date_desc = f" from {args.start}"
        elif args.end:
            date_desc = f" until {args.end}"
        limit_desc = f"{args.limit} " if not args.start else ""
        print(f"\n[*] Fetching {limit_desc}{args.timeframe} candles for {args.symbol} from {args.exchange}{date_desc}...")
        df = fetch_ohlcv(args.symbol, args.timeframe, args.exchange, args.limit,
                         start=args.start, end=args.end)

    print(f"    {len(df)} candles loaded in {time.perf_counter() - t0:.2f}s")

    data = {
        "open":   df["open"].to_numpy().astype(np.float64),
        "high":   df["high"].to_numpy().astype(np.float64),
        "low":    df["low"].to_numpy().astype(np.float64),
        "close":  df["close"].to_numpy().astype(np.float64),
        "volume": df["volume"].to_numpy().astype(np.float64),
    }
    close = data["close"]

    if isinstance(strategies, Strategy):
        strategies = [strategies]

    combos = []
    combo_strat = []  # index into strategies list per combo
    strat_lookup = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for si, strat in enumerate(strategies):
            factory = strat.as_factory()
            name, grid, signal_fn = factory()
            strat_lookup[name] = signal_fn
            for params in expand_grid(grid):
                combos.append((name, params))
                combo_strat.append(si)

    use_managed = any(s.has_managed for s in strategies)

    t1 = time.perf_counter()

    clear_cache()
    n_combos = len(combos)
    signals_2d = np.empty((n_combos, len(close)), dtype=np.float64)
    for i, (name, params) in enumerate(combos):
        signals_2d[i] = strat_lookup[name](data, **params)
    clear_cache()
    clear_sdk_cache()

    t_sig = time.perf_counter()
    print(f"    Signal generation: {t_sig - t1:.2f}s")

    # Read execution params from the first Strategy object
    first = strategies[0]
    fee_pct = first.fee if first.fee is not None else 0.075
    slip_pct = first.slippage
    capital = first.capital
    fraction = first.fraction
    top = first.top
    sort = first.sort
    fee = fee_pct / 100.0
    slippage = slip_pct / 100.0

    if not use_managed:
        print(f"\n[*] Testing {n_combos} strategy combinations...")
        print(f"    Capital: ${capital:,.0f} | Fraction: {fraction*100:.0f}% | Fee: {fee_pct}% | Slippage: {slip_pct}%")

        metrics_2d = backtest_batch(close, signals_2d, fee, slippage, capital, fraction)
        results = []
        for i, (name, params) in enumerate(combos):
            m = metrics_2d[i]
            results.append(Result(
                strategy=name, params=params,
                total_return_pct=round(m[0], 2), sharpe_ratio=round(m[1], 3),
                max_drawdown_pct=round(m[2], 2), win_rate_pct=round(m[3], 2),
                num_trades=int(m[4]), profit_factor=round(m[5], 3),
            ))
    else:
        high = data["high"]
        low = data["low"]
        expanded_indices = []
        expanded_combos = []

        for base_i, (name, params) in enumerate(combos):
            strat = strategies[combo_strat[base_i]]
            sl_vals = [v / 100.0 for v in strat.stop_loss] if strat.stop_loss else [0.0]
            tp_vals = [v / 100.0 for v in strat.take_profit] if strat.take_profit else [0.0]
            trail_vals = [v / 100.0 for v in strat.trailing_stop] if strat.trailing_stop else [0.0]

            for sl_v, tp_v, trail_v in itertools.product(sl_vals, tp_vals, trail_vals):
                expanded_indices.append((base_i, sl_v, tp_v, trail_v))
                pm_params = dict(params)
                if strat.stop_loss:
                    pm_params["stop_loss"] = round(sl_v * 100, 2)
                if strat.take_profit:
                    pm_params["take_profit"] = round(tp_v * 100, 2)
                if strat.trailing_stop:
                    pm_params["trailing_stop"] = round(trail_v * 100, 2)
                expanded_combos.append((name, pm_params))

        n_expanded = len(expanded_indices)

        print(f"\n[*] Testing {n_expanded} backtests ({n_combos} signals x {n_expanded // n_combos} PM combos)...")
        print(f"    Capital: ${capital:,.0f} | Fraction: {fraction*100:.0f}% | Fee: {fee_pct}% | Slippage: {slip_pct}%")

        sig_indices = np.empty(n_expanded, dtype=np.int64)
        sl_arr = np.empty(n_expanded, dtype=np.float64)
        tp_arr = np.empty(n_expanded, dtype=np.float64)
        trail_arr = np.empty(n_expanded, dtype=np.float64)

        for idx, (base_i, sl_v, tp_v, trail_v) in enumerate(expanded_indices):
            sig_indices[idx] = base_i
            sl_arr[idx] = sl_v
            tp_arr[idx] = tp_v
            trail_arr[idx] = trail_v

        metrics_2d = backtest_batch_managed(
            close, high, low, signals_2d, fee, slippage,
            capital, fraction,
            sig_indices, sl_arr, tp_arr, trail_arr)

        results = []
        for i, (name, params) in enumerate(expanded_combos):
            m = metrics_2d[i]
            results.append(Result(
                strategy=name, params=params,
                total_return_pct=round(m[0], 2), sharpe_ratio=round(m[1], 3),
                max_drawdown_pct=round(m[2], 2), win_rate_pct=round(m[3], 2),
                num_trades=int(m[4]), profit_factor=round(m[5], 3),
            ))

    t_bt = time.perf_counter()
    print(f"    Numba backtest:    {t_bt - t_sig:.2f}s")

    total_combos = len(results)
    elapsed = time.perf_counter() - t1
    print(f"    Done in {elapsed:.4f}s ({total_combos / max(elapsed, 0.0001):.0f} backtests/sec)")

    ranked = rank_results(results, sort_by=sort)

    print(f"\n{'=' * 70}")
    print(f"  TOP {top} STRATEGIES (sorted by {sort})")
    print(f"{'=' * 70}")

    header = f"{'#':>3} {'Strategy':<20} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7} {'PF':>7}  Params"
    print(header)
    print("-" * len(header) + "-" * 30)

    for i, r in enumerate(ranked[:top], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
        print(
            f"{i:>3} {r.strategy:<20} {r.total_return_pct:>+8.2f}% "
            f"{r.sharpe_ratio:>8.3f} {r.max_drawdown_pct:>8.2f} "
            f"{r.win_rate_pct:>8.2f}% {r.num_trades:>7} {r.profit_factor:>7.3f}  {params_str}"
        )

    best = ranked[0]
    print(f"\n{'=' * 70}")
    print(f"  BEST STRATEGY: {best.strategy}")
    print(f"  Params: {best.params}")
    print(f"  Return: {best.total_return_pct:+.2f}% | Sharpe: {best.sharpe_ratio:.3f} | MaxDD: {best.max_drawdown_pct:.2f}%")
    print(f"  Win Rate: {best.win_rate_pct:.2f}% | Trades: {best.num_trades} | Profit Factor: {best.profit_factor:.3f}")
    print(f"{'=' * 70}")

    total_time = time.perf_counter() - t0
    print(f"\nTotal time: {total_time:.2f}s")


# ---------------------------------------------------------------------------
# Public API for ``from siton.sdk import *``
# ---------------------------------------------------------------------------

__all__ = [
    "Signal", "Strategy",
    "backtest", "run", "clear_sdk_cache",
    "ema_cross", "sma_cross", "dema_cross", "tema_cross",
    "wma_cross", "kama_cross", "trima_cross",
    "rsi", "cmo", "stoch_rsi", "cci", "willr", "mfi", "ultimate_osc",
    "macd_signal", "apo", "ppo", "mom", "roc", "trix",
    "adx", "adxr", "dx", "di_cross", "dm_cross",
    "obv", "ad", "adosc",
    "bollinger_bands", "atr_breakout", "natr_breakout", "channel_breakout",
    "sar", "sar_ext",
    "candlestick_pattern", "indicator", "custom",
]
