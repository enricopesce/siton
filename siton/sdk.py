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
import math
import threading
import time
import warnings
from itertools import product

import numpy as np
import numba as nb
import talib

from siton.engine import (backtest_batch, backtest_batch_managed, backtest_batch_atr_managed,
                          backtest_one_equity, backtest_one_managed_equity,
                          backtest_one_atr_managed_equity,
                          Result, rank_results, _NUM_CORES)


# ---------------------------------------------------------------------------
# Timeframe → Sharpe annualization
# ---------------------------------------------------------------------------

_TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800,
}


def _ann_factor(timeframe="1h"):
    """Compute sqrt(periods_per_year) for Sharpe ratio annualization."""
    secs = _TIMEFRAME_SECONDS.get(timeframe, 3600)
    periods_per_year = 365.25 * 86400 / secs
    return np.float64(periods_per_year) ** 0.5


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
    ``id()`` of positional args is included in the key so that different
    input arrays (e.g. ATR computed with different ``timeperiod``) are
    cached separately.  This is safe because arrays don't change during
    a single run and the cache is cleared between runs.
    """
    args_id = tuple(id(a) for a in args)
    key = (func.__name__, _tag, args_id) + tuple(sorted(kwargs.items()))
    if key not in _cache:
        _cache[key] = func(*args, **kwargs)
    return _cache[key]


def expand_grid(grid: dict) -> list[dict]:
    """Expand parameter grid into list of param dicts."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ---------------------------------------------------------------------------
# Signal combination primitives
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def sig_and(a, b):
    """Both must agree: long+long=long, short+short=short, else 0."""
    n = len(a)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        ai, bi = a[i], b[i]
        if ai == 1.0 and bi == 1.0:
            out[i] = 1.0
        elif ai == -1.0 and bi == -1.0:
            out[i] = -1.0
        else:
            out[i] = 0.0
    return out


@nb.njit(cache=True)
def sig_or(a, b):
    """Take first non-zero signal; conflict (long vs short) = 0."""
    n = len(a)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        ai, bi = a[i], b[i]
        if ai != 0.0:
            if bi != 0.0 and ai != bi:
                out[i] = 0.0
            else:
                out[i] = ai
        else:
            out[i] = bi
    return out


@nb.njit(cache=True)
def sig_not(a):
    """Invert signal: -a."""
    n = len(a)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = -a[i]
    return out


@nb.njit(cache=True)
def sig_filter(sig, cond):
    """sig passes only where cond != 0."""
    n = len(sig)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = sig[i] if cond[i] != 0.0 else 0.0
    return out


@nb.njit(cache=True)
def sig_agree(sig, cond):
    """sig passes only where cond agrees on direction."""
    n = len(sig)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        si, ci = sig[i], cond[i]
        if (si == 1.0 and ci == 1.0) or (si == -1.0 and ci == -1.0):
            out[i] = si
        else:
            out[i] = 0.0
    return out


@nb.njit(cache=True)
def _sig_majority_impl(stacked, n_sigs, n_bars):
    """Inner JIT loop for majority vote."""
    threshold = n_sigs / 2.0
    out = np.empty(n_bars, dtype=np.float64)
    for i in range(n_bars):
        longs = 0
        shorts = 0
        for j in range(n_sigs):
            v = stacked[j, i]
            if v == 1.0:
                longs += 1
            elif v == -1.0:
                shorts += 1
        if longs > threshold:
            out[i] = 1.0
        elif shorts > threshold:
            out[i] = -1.0
        else:
            out[i] = 0.0
    return out


def sig_majority(*sigs):
    """Majority vote: direction with >50% of votes wins."""
    stacked = np.array(sigs, dtype=np.float64)
    return _sig_majority_impl(stacked, len(sigs), stacked.shape[1])


@nb.njit(cache=True)
def sig_confirm(sig, cond, lookback):
    """sig valid only if cond was active (same direction) in last N bars."""
    n = len(sig)
    out = np.zeros(n, dtype=np.float64)
    for i in range(lookback, n):
        if sig[i] == 0.0:
            continue
        direction = sig[i]
        for j in range(1, lookback + 1):
            if cond[i - j] == direction:
                out[i] = direction
                break
    return out


@nb.njit(cache=True)
def sig_entry_exit(entry_long, exit_long, entry_short, exit_short):
    """State machine: entry/exit signals control position asymmetrically."""
    n = len(entry_long)
    out = np.zeros(n, dtype=np.float64)
    pos = 0.0  # 0=flat, 1=long, -1=short
    for i in range(n):
        if pos == 0.0:
            if entry_long[i] == 1.0:
                pos = 1.0
            elif entry_short[i] == -1.0:
                pos = -1.0
        elif pos == 1.0:
            if exit_long[i] == -1.0:
                pos = 0.0
        elif pos == -1.0:
            if exit_short[i] == 1.0:
                pos = 0.0
        out[i] = pos
    return out


# ---------------------------------------------------------------------------
# JIT warmup — compile once at import time
# ---------------------------------------------------------------------------

# JIT warmup — compile all signal combiners once at import time
_w = np.zeros(2, dtype=np.float64)
sig_and(_w, _w)
sig_or(_w, _w)
sig_not(_w)
sig_filter(_w, _w)
sig_agree(_w, _w)
_sig_majority_impl(np.zeros((2, 2), dtype=np.float64), 2, 2)
sig_entry_exit(_w, _w, _w, _w)
sig_confirm(_w, _w, 1)
del _w

def _warn_grid_size(name, grid):
    """Warn if grid produces >500 combinations."""
    sizes = [len(v) for v in grid.values() if isinstance(v, (list, tuple))]
    total = 1
    for s in sizes:
        total *= s
    if total > 500:
        warnings.warn(f"Composite '{name}': grid produces {total} combinations (>500)")



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


# BUG-11: Thread-safe counter using itertools.count + lock
_counter_lock = threading.Lock()
_counter_iter = itertools.count(1)

def _next_id():
    with _counter_lock:
        return next(_counter_iter)


# Sub-signal memoization registry — caches are created inside factory
# closures and registered here so they can be bulk-cleared after use.
_sdk_caches = []

# Set by run() so backtest() can report the true combo count.
_last_n_combos: int = 0

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

    non_colliding_r = set(grid_r) - collision
    rename = {}
    for k in grid_r:
        if k in collision:
            for i in range(2, 100):
                new_k = f"{k}_{i}"
                if (new_k not in grid_l and
                        new_k not in rename.values() and
                        new_k not in non_colliding_r):
                    rename[k] = new_k
                    break
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

            return sig_entry_exit(s_el, s_xl, s_el, s_xl)

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
        """``~a`` — binary filter: 1 where signal is flat (0), 0 elsewhere.

        NOT algebraic negation (use ``.negate()`` for long↔short swap).
        Intended for filter_by: ``signal.filter_by(~rsi(...))`` blocks
        trades in overbought/oversold zones.
        """
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

def _crossover_factory(name, ta_fn, fast, slow):
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

# ---- Trend: MA crossovers ----

def ema_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _crossover_factory(f"_ema{_next_id()}", talib.EMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "ema"), label="ema")

def sma_cross(fast=(10, 20), slow=(50, 100)):
    return Signal(_labeled_factory(
        _crossover_factory(f"_sma{_next_id()}", talib.SMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "sma"), label="sma")

def dema_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _crossover_factory(f"_dema{_next_id()}", talib.DEMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "dema"), label="dema")

def tema_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _crossover_factory(f"_tema{_next_id()}", talib.TEMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "tema"), label="tema")

def wma_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _crossover_factory(f"_wma{_next_id()}", talib.WMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "wma"), label="wma")

def kama_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _crossover_factory(f"_kama{_next_id()}", talib.KAMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "kama"), label="kama")

def trima_cross(fast=(8, 12), slow=(26, 50)):
    return Signal(_labeled_factory(
        _crossover_factory(f"_trima{_next_id()}", talib.TRIMA,
                       fast=_ensure_list(fast), slow=_ensure_list(slow)),
        "trima"), label="trima")


# ---- Oscillators ----
def _threshold_factory(name, ta_fn, periods, oversold, overbought):
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


def rsi(periods=(14,), oversold=(30,), overbought=(70,)):
    return Signal(_labeled_factory(
        _threshold_factory(f"_rsi{_next_id()}", talib.RSI,
                       periods=_ensure_list(periods),
                       oversold=_ensure_list(oversold),
                       overbought=_ensure_list(overbought)),
        "rsi"), label="rsi")

def cmo(periods=(14,), oversold=(-50,), overbought=(50,)):
    return Signal(_labeled_factory(
        _threshold_factory(f"_cmo{_next_id()}", talib.CMO,
                       periods=_ensure_list(periods),
                       oversold=_ensure_list(oversold),
                       overbought=_ensure_list(overbought)),
        "cmo"), label="cmo")

def _stochrsi_factory(name, periods, fastk, fastd, oversold, overbought):
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

def stoch_rsi(periods=(14,), fastk=(3,), fastd=(3,),
              oversold=(20,), overbought=(80,)):
    return Signal(_labeled_factory(
        _stochrsi_factory(f"_stochrsi{_next_id()}",
                      periods=_ensure_list(periods),
                      fastk=_ensure_list(fastk),
                      fastd=_ensure_list(fastd),
                      oversold=_ensure_list(oversold),
                      overbought=_ensure_list(overbought)),
        "stochrsi"), label="stochrsi")

def _threshold_hlc_factory(name, ta_fn, periods, oversold, overbought):
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


def cci(periods=(14,), oversold=(-100,), overbought=(100,)):
    return Signal(_labeled_factory(
        _threshold_hlc_factory(f"_cci{_next_id()}", talib.CCI,
                           periods=_ensure_list(periods),
                           oversold=_ensure_list(oversold),
                           overbought=_ensure_list(overbought)),
        "cci"), label="cci")

def willr(periods=(14,), oversold=(-80,), overbought=(-20,)):
    return Signal(_labeled_factory(
        _threshold_hlc_factory(f"_willr{_next_id()}", talib.WILLR,
                           periods=_ensure_list(periods),
                           oversold=_ensure_list(oversold),
                           overbought=_ensure_list(overbought)),
        "willr"), label="willr")

def _threshold_hlcv_factory(name, ta_fn, periods, oversold, overbought):
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

def mfi(periods=(14,), oversold=(20,), overbought=(80,)):
    return Signal(_labeled_factory(
        _threshold_hlcv_factory(f"_mfi{_next_id()}", talib.MFI,
                            periods=_ensure_list(periods),
                            oversold=_ensure_list(oversold),
                            overbought=_ensure_list(overbought)),
        "mfi"), label="mfi")

def _ultosc_factory(name, p1_list, p2_list, p3_list, oversold, overbought):
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

def ultimate_osc(p1=(7,), p2=(14,), p3=(28,), oversold=(30,), overbought=(70,)):
    return Signal(_labeled_factory(
        _ultosc_factory(f"_ultosc{_next_id()}",
                    p1_list=_ensure_list(p1),
                    p2_list=_ensure_list(p2),
                    p3_list=_ensure_list(p3),
                    oversold=_ensure_list(oversold),
                    overbought=_ensure_list(overbought)),
        "ultosc"), label="ultosc")


# ---- Momentum ----

def _macd_factory(name, fast, slow, signal_period):
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

def macd_signal(fast=(12,), slow=(26,), signal_period=(9,)):
    return Signal(_labeled_factory(
        _macd_factory(f"_macd{_next_id()}",
                  fast=_ensure_list(fast),
                  slow=_ensure_list(slow),
                  signal_period=_ensure_list(signal_period)),
        "macd"), label="macd")

def _dual_zero_cross_factory(name, ta_fn, fast, slow):
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

def apo(fast=(12,), slow=(26,)):
    return Signal(_labeled_factory(
        _dual_zero_cross_factory(f"_apo{_next_id()}", talib.APO,
                             fast=_ensure_list(fast),
                             slow=_ensure_list(slow)),
        "apo"), label="apo")

def ppo(fast=(12,), slow=(26,)):
    return Signal(_labeled_factory(
        _dual_zero_cross_factory(f"_ppo{_next_id()}", talib.PPO,
                             fast=_ensure_list(fast),
                             slow=_ensure_list(slow)),
        "ppo"), label="ppo")

def _zero_cross_factory(name, ta_fn, periods):
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

def mom(periods=(10,)):
    return Signal(_labeled_factory(
        _zero_cross_factory(f"_mom{_next_id()}", talib.MOM,
                        periods=_ensure_list(periods)),
        "mom"), label="mom")

def roc(periods=(10,)):
    return Signal(_labeled_factory(
        _zero_cross_factory(f"_roc{_next_id()}", talib.ROC,
                        periods=_ensure_list(periods)),
        "roc"), label="roc")

def trix(periods=(14,)):
    return Signal(_labeled_factory(
        _zero_cross_factory(f"_trix{_next_id()}", talib.TRIX,
                        periods=_ensure_list(periods)),
        "trix"), label="trix")


# ---- Trend strength ----
def _trend_strength_factory(name, ta_fn, periods, thresholds):
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

def adx(periods=(14,), thresholds=(25,)):
    return Signal(_labeled_factory(
        _trend_strength_factory(f"_adx{_next_id()}", talib.ADX,
                            periods=_ensure_list(periods),
                            thresholds=_ensure_list(thresholds)),
        "adx"), label="adx")

def adxr(periods=(14,), thresholds=(25,)):
    return Signal(_labeled_factory(
        _trend_strength_factory(f"_adxr{_next_id()}", talib.ADXR,
                            periods=_ensure_list(periods),
                            thresholds=_ensure_list(thresholds)),
        "adxr"), label="adxr")

def dx(periods=(14,), thresholds=(25,)):
    return Signal(_labeled_factory(
        _trend_strength_factory(f"_dx{_next_id()}", talib.DX,
                            periods=_ensure_list(periods),
                            thresholds=_ensure_list(thresholds)),
        "dx"), label="dx")

def _di_crossover_factory(name, periods):
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

def di_cross(periods=(14,)):
    return Signal(_labeled_factory(
        _di_crossover_factory(f"_di{_next_id()}",
                          periods=_ensure_list(periods)),
        "di"), label="di")

def _dm_crossover_factory(name, periods):
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

def dm_cross(periods=(14,)):
    return Signal(_labeled_factory(
        _dm_crossover_factory(f"_dm{_next_id()}",
                          periods=_ensure_list(periods)),
        "dm"), label="dm")


# ---- Volume ----
def _obv_crossover_factory(name, ma_periods):
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

def obv(ma_periods=(20,)):
    return Signal(_labeled_factory(
        _obv_crossover_factory(f"_obv{_next_id()}",
                 ma_periods=_ensure_list(ma_periods)),
        "obv"), label="obv")

def _ad_crossover_factory(name, ma_periods):
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

def ad(ma_periods=(20,)):
    return Signal(_labeled_factory(
        _ad_crossover_factory(f"_ad{_next_id()}",
                ma_periods=_ensure_list(ma_periods)),
        "ad"), label="ad")

def _adosc_zero_cross_factory(name, fast_periods, slow_periods):
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

def adosc(fast=(3,), slow=(10,)):
    return Signal(_labeled_factory(
        _adosc_zero_cross_factory(f"_adosc{_next_id()}",
                   fast_periods=_ensure_list(fast),
                   slow_periods=_ensure_list(slow)),
        "adosc"), label="adosc")


# ---- Volatility ----
def _bollinger_factory(name, windows, num_std):
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

def bollinger_bands(windows=(20,), num_std=(2.0,)):
    return Signal(_labeled_factory(
        _bollinger_factory(f"_bb{_next_id()}",
                       windows=_ensure_list(windows),
                       num_std=_ensure_list(num_std)),
        "bb"), label="bb")

def _volatility_breakout_factory(name, ta_fn, periods, lookbacks):
    """High volatility + price direction: trade when ATR/NATR is expanding."""
    def factory():
        grid = {"period": periods, "lookback": lookbacks}

        def signal_fn(data, period, lookback):
            high, low, close = data["high"], data["low"], data["close"]
            if lookback <= 0:
                return np.zeros(len(close), dtype=np.float64)
            atr = _cached(ta_fn, high, low, close, timeperiod=period)
            atr_ma = _cached(talib.SMA, atr, _tag=f"atr_ma_{ta_fn.__name__}", timeperiod=lookback)
            # BUG-07: Replace np.roll (wraps around) with explicit shift to avoid lookahead
            prev_close = np.empty_like(close)
            prev_close[:lookback] = np.nan
            prev_close[lookback:] = close[:-lookback]
            direction = np.where(close > prev_close, 1.0,
                                 np.where(close < prev_close, -1.0, 0.0))
            sig = np.where(atr > atr_ma, direction, 0.0)
            sig[np.isnan(atr_ma) | np.isnan(prev_close)] = 0.0
            return sig

        return name, grid, signal_fn
    return factory

def atr_breakout(periods=(14,), lookbacks=(20,)):
    return Signal(_labeled_factory(
        _volatility_breakout_factory(f"_atrbrk{_next_id()}", talib.ATR,
                                 periods=_ensure_list(periods),
                                 lookbacks=_ensure_list(lookbacks)),
        "atr"), label="atr")

def natr_breakout(periods=(14,), lookbacks=(20,)):
    return Signal(_labeled_factory(
        _volatility_breakout_factory(f"_natrbrk{_next_id()}", talib.NATR,
                                 periods=_ensure_list(periods),
                                 lookbacks=_ensure_list(lookbacks)),
        "natr"), label="natr")

def _breakout_factory(name, periods):
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

def channel_breakout(periods=(20,)):
    return Signal(_labeled_factory(
        _breakout_factory(f"_chan{_next_id()}",
                      periods=_ensure_list(periods)),
        "chan"), label="chan")


# ---- Exit signals ----
def _sar_crossover_factory(name, accelerations, maximums):
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

def sar(accelerations=(0.02,), maximums=(0.2,)):
    return Signal(_labeled_factory(
        _sar_crossover_factory(f"_sar{_next_id()}",
                 accelerations=_ensure_list(accelerations),
                 maximums=_ensure_list(maximums)),
        "sar"), label="sar")

def _sarext_crossover_factory(name, accel_inits, accelerations, maximums):
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

def sar_ext(accel_inits=(0.02,), accelerations=(0.02,), maximums=(0.2,)):
    return Signal(_labeled_factory(
        _sarext_crossover_factory(f"_sarext{_next_id()}",
                    accel_inits=_ensure_list(accel_inits),
                    accelerations=_ensure_list(accelerations),
                    maximums=_ensure_list(maximums)),
        "sarext"), label="sarext")


# ---- Candlestick patterns ----

def _candlestick_factory(name, ta_fn, penetration_values=None):
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

def candlestick_pattern(name, penetration_values=None):
    """Candlestick pattern by TA-Lib function name (e.g. 'CDLENGULFING')."""
    ta_fn = getattr(talib, name)
    pv = _ensure_list(penetration_values) if penetration_values else None
    label = name.lower().replace("cdl", "cdl_") if name.startswith("CDL") else name.lower()
    return Signal(_labeled_factory(
        _candlestick_factory(f"_cdl{_next_id()}", ta_fn, penetration_values=pv),
        label), label=label)


# ---- Escape hatches ----

def indicator(factory_fn, *args, label="ind", **kwargs):
    """Wrap any external factory callable as a Signal."""
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
                 sort="sharpe_ratio", long_only=False,
                 risk_free_rate=0.0, validate=False,
                 train_ratio=0.7, n_splits=5,
                 atr_period=14,
                 atr_stop_mult=None,
                 atr_tp_mult=None,
                 atr_trail_mult=None):
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
        self.long_only = long_only
        self.risk_free_rate = risk_free_rate
        self.validate = validate
        self.train_ratio = train_ratio
        self.n_splits = n_splits
        self.atr_period = atr_period
        self.atr_stop_mult = _ensure_list(atr_stop_mult)
        self.atr_tp_mult = _ensure_list(atr_tp_mult)
        self.atr_trail_mult = _ensure_list(atr_trail_mult)

    @property
    def has_atr_managed(self):
        """True if ATR-based position management is configured."""
        return any(x is not None for x in (
            self.atr_stop_mult, self.atr_tp_mult, self.atr_trail_mult))

    @property
    def has_managed(self):
        """True if any position management parameter is set."""
        return any(x is not None for x in (
            self.stop_loss, self.take_profit, self.trailing_stop)) or self.has_atr_managed

    def as_factory(self):
        return self._factory

    def to_strategy_list(self):
        return [self._factory]


# ---------------------------------------------------------------------------
# Pipeline: backtest() and run()
# ---------------------------------------------------------------------------

def _attach_equity_curves(results, close, high, low, open_arr, strat_lookup, data,
                          fee_dec, slip_dec, capital, fraction, ann, rf_per_bar,
                          use_managed, strategies, name_to_strat_idx, long_only,
                          bar_offset=0, atr_arr=None):
    """Attach equity curves to ranked results; regenerates each signal on-demand."""
    pm_keys = {"stop_loss", "take_profit", "trailing_stop"}
    n_bars = len(close)
    for r in results:
        base_params = {k: v for k, v in r.params.items() if k not in pm_keys}
        raw = strat_lookup[r.strategy](data, **base_params)
        n_total = len(raw)
        shifted = np.empty(n_total, dtype=np.float64)
        shifted[0] = 0.0
        shifted[1:] = raw[:-1]
        if long_only:
            shifted = np.where(shifted < 0.0, 0.0, shifted)
        sig = shifted[bar_offset : bar_offset + n_bars]
        use_atr = atr_arr is not None and any(s.has_atr_managed for s in strategies)
        if use_atr:
            sl_m    = r.params.get("atr_sl_mult",    0.0)
            tp_m    = r.params.get("atr_tp_mult",    0.0)
            trail_m = r.params.get("atr_trail_mult", 0.0)
            r.equity_curve = backtest_one_atr_managed_equity(
                close, high, low, open_arr, sig, fee_dec, slip_dec,
                capital, fraction, ann, rf_per_bar, 0.0,
                atr_arr, sl_m, tp_m, trail_m)
        elif use_managed:
            strat = strategies[name_to_strat_idx.get(r.strategy, 0)]
            sl_v = r.params.get("stop_loss", 0.0) / 100.0 if "stop_loss" in r.params else 0.0
            tp_v = r.params.get("take_profit", 0.0) / 100.0 if "take_profit" in r.params else 0.0
            trail_v = r.params.get("trailing_stop", 0.0) / 100.0 if "trailing_stop" in r.params else 0.0
            r.equity_curve = backtest_one_managed_equity(
                close, high, low, open_arr, sig, fee_dec, slip_dec,
                capital, fraction, ann, rf_per_bar,
                0.0, sl_v, tp_v, trail_v)
        else:
            r.equity_curve = backtest_one_equity(
                close, high, low, open_arr, sig, fee_dec, slip_dec, capital, fraction, ann, rf_per_bar)


def _run_batch(close, high, low, open_arr, signals_2d, combos, combo_strat, strategies,
               fee_dec, slip_dec, capital, fraction, ann, rf_per_bar, use_managed,
               combo_indices=None, atr_arr=None):
    """Run backtests for given combos. If combo_indices is given, only run those."""
    if combo_indices is not None:
        run_combos = [combos[i] for i in combo_indices]
        run_combo_strat = [combo_strat[i] for i in combo_indices]
        run_signals = signals_2d[combo_indices]
    else:
        run_combos = combos
        run_combo_strat = combo_strat
        run_signals = signals_2d

    use_atr_managed = atr_arr is not None and any(s.has_atr_managed for s in strategies)

    if use_atr_managed:
        expanded_indices = []
        expanded_combos = []

        for base_i, (name, params) in enumerate(run_combos):
            strat = strategies[run_combo_strat[base_i]]
            sl_mults  = strat.atr_stop_mult  if strat.atr_stop_mult  else [0.0]
            tp_mults  = strat.atr_tp_mult    if strat.atr_tp_mult    else [0.0]
            trail_mults = strat.atr_trail_mult if strat.atr_trail_mult else [0.0]

            for sl_m, tp_m, trail_m in product(sl_mults, tp_mults, trail_mults):
                expanded_indices.append((base_i, sl_m, tp_m, trail_m))
                pm_params = dict(params)
                if strat.atr_stop_mult:
                    pm_params["atr_sl_mult"] = sl_m
                if strat.atr_tp_mult:
                    pm_params["atr_tp_mult"] = tp_m
                if strat.atr_trail_mult:
                    pm_params["atr_trail_mult"] = trail_m
                expanded_combos.append((name, pm_params))

        n_exp = len(expanded_indices)
        sig_indices_arr  = np.empty(n_exp, dtype=np.int64)
        sl_mult_arr_     = np.empty(n_exp, dtype=np.float64)
        tp_mult_arr_     = np.empty(n_exp, dtype=np.float64)
        trail_mult_arr_  = np.empty(n_exp, dtype=np.float64)

        for idx, (base_i, sl_m, tp_m, trail_m) in enumerate(expanded_indices):
            sig_indices_arr[idx] = base_i
            sl_mult_arr_[idx]    = sl_m
            tp_mult_arr_[idx]    = tp_m
            trail_mult_arr_[idx] = trail_m

        metrics_2d = backtest_batch_atr_managed(
            close, high, low, open_arr, run_signals, atr_arr,
            fee_dec, slip_dec, capital, fraction, ann, rf_per_bar, 0.0,
            sig_indices_arr, sl_mult_arr_, tp_mult_arr_, trail_mult_arr_)

        results = []
        for i, (name, params) in enumerate(expanded_combos):
            m = metrics_2d[i]
            results.append(Result(
                strategy=name, params=params,
                total_return_pct=round(m[0], 2), sharpe_ratio=round(m[1], 3),
                max_drawdown_pct=round(m[2], 2), win_rate_pct=round(m[3], 2),
                num_trades=int(m[4]), profit_factor=round(m[5], 3),
                sortino_ratio=round(m[6], 3), calmar_ratio=round(m[7], 3),
            ))
    elif not use_managed:
        metrics_2d = backtest_batch(
            close, high, low, open_arr, run_signals, fee_dec, slip_dec, capital, fraction, ann, rf_per_bar)
        results = []
        for i, (name, params) in enumerate(run_combos):
            m = metrics_2d[i]
            results.append(Result(
                strategy=name, params=params,
                total_return_pct=round(m[0], 2), sharpe_ratio=round(m[1], 3),
                max_drawdown_pct=round(m[2], 2), win_rate_pct=round(m[3], 2),
                num_trades=int(m[4]), profit_factor=round(m[5], 3),
                sortino_ratio=round(m[6], 3), calmar_ratio=round(m[7], 3),
            ))
    else:
        expanded_indices = []
        expanded_combos = []

        for base_i, (name, params) in enumerate(run_combos):
            strat = strategies[run_combo_strat[base_i]]
            sl_vals = [v / 100.0 for v in strat.stop_loss] if strat.stop_loss else [0.0]
            tp_vals = [v / 100.0 for v in strat.take_profit] if strat.take_profit else [0.0]
            trail_vals = [v / 100.0 for v in strat.trailing_stop] if strat.trailing_stop else [0.0]

            for sl_v, tp_v, trail_v in product(sl_vals, tp_vals, trail_vals):
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
            close, high, low, open_arr, run_signals, fee_dec, slip_dec,
            capital, fraction, ann, rf_per_bar,
            0.0,  # borrow_rate_per_bar (BUG-10 placeholder, set to 0 by default)
            sig_indices, sl_arr, tp_arr, trail_arr)

        results = []
        for i, (name, params) in enumerate(expanded_combos):
            m = metrics_2d[i]
            results.append(Result(
                strategy=name, params=params,
                total_return_pct=round(m[0], 2), sharpe_ratio=round(m[1], 3),
                max_drawdown_pct=round(m[2], 2), win_rate_pct=round(m[3], 2),
                num_trades=int(m[4]), profit_factor=round(m[5], 3),
                sortino_ratio=round(m[6], 3), calmar_ratio=round(m[7], 3),
            ))

    return results


def run(strategies, df_or_data, fee=None, slippage=None, capital=None,
        fraction=None, top=None, sort=None, timeframe="1h"):
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
        list[Result] when validate=False (default).
        dict with 'train' and 'test' lists when validate=True.
    """
    if isinstance(strategies, Strategy):
        strategies = [strategies]

    # Priority: function param > strategy > default
    strat = strategies[0]
    fee = fee if fee is not None else (strat.fee if strat.fee is not None else 0.075)
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
    # Ensure contiguous arrays for optimal Numba SIMD access
    close = np.ascontiguousarray(data["close"])
    high = np.ascontiguousarray(data["high"])
    low = np.ascontiguousarray(data["low"])
    open_arr = np.ascontiguousarray(data["open"])

    # Build combos and track which strategy each combo belongs to
    combos = []
    combo_strat = []  # index into strategies list per combo
    strat_lookup = {}
    name_to_strat_idx = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for si, strat in enumerate(strategies):
            factory = strat.as_factory()
            name, grid, signal_fn = factory()
            strat_lookup[name] = signal_fn
            name_to_strat_idx[name] = si
            for params in expand_grid(grid):
                combos.append((name, params))
                combo_strat.append(si)

    clear_cache()
    n_combos = len(combos)
    n_bars = len(close)
    long_only = any(s.long_only for s in strategies)

    fee_dec = fee / 100.0
    slip_dec = slippage / 100.0
    ann = _ann_factor(timeframe)
    use_managed = any(s.has_managed for s in strategies)

    # Compute per-bar risk-free rate from annual rate
    rf_annual = strategies[0].risk_free_rate
    secs = _TIMEFRAME_SECONDS.get(timeframe, 3600)
    periods_per_year = 365.25 * 86400 / secs
    rf_per_bar = rf_annual / periods_per_year

    # Generate all signals at once (shifted by 1 bar to avoid lookahead)
    signals_2d = np.empty((n_combos, n_bars), dtype=np.float64)
    for i, (name, params) in enumerate(combos):
        signals_2d[i] = strat_lookup[name](data, **params)
    signals_2d[:, 1:] = signals_2d[:, :-1]
    signals_2d[:, 0] = 0.0
    if long_only:
        signals_2d = np.where(signals_2d < 0, 0.0, signals_2d)
    clear_cache()

    # Compute ATR for ATR-managed strategies (full series, no look-ahead)
    atr_arr_full = None
    if any(s.has_atr_managed for s in strategies):
        atr_period = strategies[0].atr_period
        _raw_atr = talib.ATR(high, low, close, timeperiod=atr_period)
        atr_arr_full = np.ascontiguousarray(_raw_atr.astype(np.float64))

    # Walk-forward validation: single IS batch + contiguous OOS + stability
    validate = any(s.validate for s in strategies)
    if validate:
        train_ratio = strategies[0].train_ratio
        n_splits = strategies[0].n_splits
        split = int(n_bars * train_ratio)
        oos_len = n_bars - split
        _pm_keys = {"stop_loss", "take_profit", "trailing_stop"}

        # Step 1 — IS: run all combos on [0:split]
        is_atr = np.ascontiguousarray(atr_arr_full[:split]) if atr_arr_full is not None else None
        is_results = _run_batch(
            np.ascontiguousarray(close[:split]),
            np.ascontiguousarray(high[:split]),
            np.ascontiguousarray(low[:split]),
            np.ascontiguousarray(open_arr[:split]),
            np.ascontiguousarray(signals_2d[:, :split]),
            combos, combo_strat, strategies,
            fee_dec, slip_dec, capital, fraction, ann, rf_per_bar, use_managed,
            atr_arr=is_atr)
        train_top = rank_results(is_results, sort_by=sort)[:top]

        # Step 2 — OOS: regenerate signals for top-K IS winners, slice to [split:]
        # Preserves IS rank — no re-ranking by OOS performance
        top_signal_params = [
            (r.strategy, {k: v for k, v in r.params.items() if k not in _pm_keys})
            for r in train_top
        ]
        top_combo_strat_idx = [name_to_strat_idx[n] for n, _ in top_signal_params]
        oos_sigs = np.empty((len(top_signal_params), oos_len), dtype=np.float64)
        for i, (name, params) in enumerate(top_signal_params):
            full = strat_lookup[name](data, **params)
            n_total = len(full)
            shifted = np.empty(n_total, dtype=np.float64)
            shifted[0] = 0.0
            shifted[1:] = full[:-1]
            if long_only:
                shifted = np.where(shifted < 0.0, 0.0, shifted)
            oos_sigs[i] = shifted[split : split + oos_len]
        clear_cache()
        oos_atr = np.ascontiguousarray(atr_arr_full[split:]) if atr_arr_full is not None else None
        oos_results = _run_batch(
            np.ascontiguousarray(close[split:]),
            np.ascontiguousarray(high[split:]),
            np.ascontiguousarray(low[split:]),
            np.ascontiguousarray(open_arr[split:]),
            oos_sigs,
            top_signal_params, top_combo_strat_idx, strategies,
            fee_dec, slip_dec, capital, fraction, ann, rf_per_bar, use_managed,
            atr_arr=oos_atr)

        # Match OOS results back to IS winners by full params — no re-ranking
        oos_lookup = {
            (r.strategy, tuple(sorted(r.params.items()))): r
            for r in oos_results
        }
        test_top = []
        for r in train_top:
            key = (r.strategy, tuple(sorted(r.params.items())))
            oos_r = oos_lookup.get(key)
            if oos_r is not None:
                test_top.append(oos_r)

        # Step 3 — Stability: expanding IS windows to track top-1 parameter robustness
        stability = []
        if train_top:
            stability.append((split, dict(train_top[0].params), train_top[0].sharpe_ratio))
        if n_splits > 1 and oos_len > 0:
            step = max(oos_len // n_splits, 50)
            for k in range(1, n_splits):
                tr_end = split + k * step
                if tr_end >= n_bars:
                    break
                win_atr = np.ascontiguousarray(atr_arr_full[:tr_end]) if atr_arr_full is not None else None
                win_results = _run_batch(
                    np.ascontiguousarray(close[:tr_end]),
                    np.ascontiguousarray(high[:tr_end]),
                    np.ascontiguousarray(low[:tr_end]),
                    np.ascontiguousarray(open_arr[:tr_end]),
                    np.ascontiguousarray(signals_2d[:, :tr_end]),
                    combos, combo_strat, strategies,
                    fee_dec, slip_dec, capital, fraction, ann, rf_per_bar, use_managed,
                    atr_arr=win_atr)
                win_ranked = rank_results(win_results, sort_by=sort)
                if win_ranked:
                    best = win_ranked[0]
                    stability.append((tr_end, dict(best.params), best.sharpe_ratio))
        clear_sdk_cache()

        # Step 4 — Equity curves with correct bar offsets
        _attach_equity_curves(
            train_top,
            np.ascontiguousarray(close[:split]),
            np.ascontiguousarray(high[:split]),
            np.ascontiguousarray(low[:split]),
            np.ascontiguousarray(open_arr[:split]),
            strat_lookup, data,
            fee_dec, slip_dec, capital, fraction, ann, rf_per_bar,
            use_managed, strategies, name_to_strat_idx, long_only,
            bar_offset=0,
            atr_arr=np.ascontiguousarray(atr_arr_full[:split]) if atr_arr_full is not None else None)

        _attach_equity_curves(
            test_top,
            np.ascontiguousarray(close[split:]),
            np.ascontiguousarray(high[split:]),
            np.ascontiguousarray(low[split:]),
            np.ascontiguousarray(open_arr[split:]),
            strat_lookup, data,
            fee_dec, slip_dec, capital, fraction, ann, rf_per_bar,
            use_managed, strategies, name_to_strat_idx, long_only,
            bar_offset=split,
            atr_arr=np.ascontiguousarray(atr_arr_full[split:]) if atr_arr_full is not None else None)

        # Step 5 — DSR
        deflated_sharpe_ratio(train_top, split)
        if test_top:
            deflated_sharpe_ratio(test_top, oos_len)

        return {"train": train_top, "test": test_top, "stability": stability, "split": split}

    # Non-validate: run all combos at once
    all_results = _run_batch(
        close, high, low, open_arr, signals_2d,
        combos, combo_strat, strategies,
        fee_dec, slip_dec, capital, fraction, ann, rf_per_bar, use_managed,
        atr_arr=atr_arr_full)

    ranked = rank_results(all_results, sort_by=sort)[:top]
    clear_sdk_cache()
    _attach_equity_curves(ranked, close, high, low, open_arr, strat_lookup, data,
                          fee_dec, slip_dec, capital, fraction, ann, rf_per_bar,
                          use_managed, strategies, name_to_strat_idx, long_only,
                          atr_arr=atr_arr_full)
    deflated_sharpe_ratio(ranked, n_bars)
    global _last_n_combos
    _last_n_combos = len(all_results)
    return ranked


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
    parser.add_argument("--validate", action="store_true", help="Enable walk-forward validation")
    parser.add_argument("--train-ratio", type=float, default=None, help="Train/test split ratio")
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
    close = np.ascontiguousarray(data["close"])
    buy_hold_pct = (close[-1] / close[0] - 1.0) * 100.0

    first = strategies[0]
    if args.validate:
        first.validate = True
    if args.train_ratio is not None:
        first.train_ratio = args.train_ratio
    top = first.top
    sort = first.sort

    t1 = time.perf_counter()
    results = run(list(strategies), data, timeframe=args.timeframe)
    elapsed = time.perf_counter() - t1

    if isinstance(results, dict):
        print(f"    Done in {elapsed:.4f}s")
        train_top = results["train"]
        test_top = results["test"]
        stability = results.get("stability", [])
        split = results.get("split", 0)
        n_bars = len(close)

        # IS table
        print(f"\n{'=' * 70}")
        print(f"  IN-SAMPLE (TRAIN) — TOP {top} (sorted by {sort})")
        print(f"{'=' * 70}")
        header = f"{'#':>3} {'Strategy':<20} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7} {'PF':>7} {'PSR':>6}  Params"
        print(header)
        print("-" * len(header) + "-" * 30)
        for i, r in enumerate(train_top[:top], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            print(
                f"{i:>3} {r.strategy:<20} {r.total_return_pct:>+8.2f}% "
                f"{r.sharpe_ratio:>8.3f} {r.max_drawdown_pct:>8.2f} "
                f"{r.win_rate_pct:>8.2f}% {r.num_trades:>7} {r.profit_factor:>7.3f} {r.psr:>6.3f}  {params_str}"
            )

        # OOS table with WFE column — IS rank order preserved, no OOS re-ranking
        is_return_lookup = {
            (r.strategy, tuple(sorted(r.params.items()))): r.total_return_pct
            for r in train_top
        }
        print(f"\n{'=' * 70}")
        print(f"  OUT-OF-SAMPLE (TEST) — IS RANK ORDER (no OOS re-ranking)")
        print(f"{'=' * 70}")
        header_oos = f"{'#':>3} {'Strategy':<20} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7} {'PF':>7} {'PSR':>6} {'WFE':>6}  Params"
        print(header_oos)
        print("-" * len(header_oos) + "-" * 30)
        for i, r in enumerate(test_top[:top], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            is_ret = is_return_lookup.get((r.strategy, tuple(sorted(r.params.items()))))
            if is_ret is not None and is_ret > 0.0:
                wfe_str = f"{r.total_return_pct / is_ret:>6.2f}"
            else:
                wfe_str = f"{'N/A':>6}"
            print(
                f"{i:>3} {r.strategy:<20} {r.total_return_pct:>+8.2f}% "
                f"{r.sharpe_ratio:>8.3f} {r.max_drawdown_pct:>8.2f} "
                f"{r.win_rate_pct:>8.2f}% {r.num_trades:>7} {r.profit_factor:>7.3f} {r.psr:>6.3f} {wfe_str}  {params_str}"
            )

        # Parameter stability section
        if stability:
            print(f"\n  PARAMETER STABILITY (top-1 as IS window expands)")
            baseline_params = stability[0][1]
            agree_count = 0
            for j, (tr_end, params, sharpe) in enumerate(stability):
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                if j == 0:
                    marker = "<- baseline"
                    agree_count += 1
                elif params == baseline_params:
                    marker = "ok"
                    agree_count += 1
                else:
                    marker = "<- changed"
                print(f"    {tr_end} bars ({tr_end * 100 / n_bars:.0f}%): {params_str} -- Sharpe {sharpe:.3f}  {marker}")
            print(f"    Stability: {agree_count}/{len(stability)} windows agree with IS winner")

        # Best strategy summary
        if train_top:
            best_is = train_top[0]
            best_oos = None
            best_key = (best_is.strategy, tuple(sorted(best_is.params.items())))
            for r in test_top:
                if (r.strategy, tuple(sorted(r.params.items()))) == best_key:
                    best_oos = r
                    break
            print(f"\n{'=' * 70}")
            print(f"  BEST STRATEGY: {best_is.strategy}")
            print(f"  Params: {best_is.params}")
            print(f"  IS  Return: {best_is.total_return_pct:>+8.2f}% | Sharpe: {best_is.sharpe_ratio:.3f} | MaxDD: {best_is.max_drawdown_pct:.2f}%")
            if best_oos:
                wfe_str = (f"{best_oos.total_return_pct / best_is.total_return_pct:.2f}x"
                           if best_is.total_return_pct > 0.0 else "N/A")
                print(f"  OOS Return: {best_oos.total_return_pct:>+8.2f}% | Sharpe: {best_oos.sharpe_ratio:.3f} | MaxDD: {best_oos.max_drawdown_pct:.2f}%  WFE: {wfe_str}")

        print(f"\n  IS bars: {split} | OOS bars: {n_bars - split}")
        print(f"  Full-period Buy & Hold: {buy_hold_pct:+.2f}%")
    else:
        total_combos = _last_n_combos
        print(f"    Done in {elapsed:.4f}s ({total_combos / max(elapsed, 0.0001):.0f} backtests/sec)")

        ranked = rank_results(results, sort_by=sort)

        print(f"\n{'=' * 70}")
        print(f"  TOP {top} STRATEGIES (sorted by {sort})")
        print(f"{'=' * 70}")

        header = f"{'#':>3} {'Strategy':<20} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7} {'PF':>7} {'PSR':>6}  Params"
        print(header)
        print("-" * len(header) + "-" * 30)

        for i, r in enumerate(ranked[:top], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            print(
                f"{i:>3} {r.strategy:<20} {r.total_return_pct:>+8.2f}% "
                f"{r.sharpe_ratio:>8.3f} {r.max_drawdown_pct:>8.2f} "
                f"{r.win_rate_pct:>8.2f}% {r.num_trades:>7} {r.profit_factor:>7.3f} {r.psr:>6.3f}  {params_str}"
            )

        best = ranked[0]
        print(f"\n{'=' * 70}")
        print(f"  BEST STRATEGY: {best.strategy}")
        print(f"  Params: {best.params}")
        print(f"  Return: {best.total_return_pct:+.2f}% | Sharpe: {best.sharpe_ratio:.3f} | MaxDD: {best.max_drawdown_pct:.2f}%")
        print(f"  Win Rate: {best.win_rate_pct:.2f}% | Trades: {best.num_trades} | Profit Factor: {best.profit_factor:.3f}")

        alpha = best.total_return_pct - buy_hold_pct
        print(f"\n  Buy & Hold: {buy_hold_pct:+.2f}%")
        print(f"  Alpha (vs B&H): {alpha:+.2f}%")
        if first.long_only:
            print(f"  Long-only mode (short signals ignored)")
        print(f"{'=' * 70}")

    total_time = time.perf_counter() - t0
    print(f"\nTotal time: {total_time:.2f}s")


# ---------------------------------------------------------------------------
# BUG-02: Probability Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via complementary error function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_cdf_inv(p: float) -> float:
    """Inverse standard normal CDF — Abramowitz & Stegun 26.2.17 rational approximation.

    Non-recursive to avoid stack overflow at p=0.5 (n_trials=2 edge case).
    """
    if p <= 0.0:
        return -1e308
    if p >= 1.0:
        return 1e308
    if p == 0.5:
        return 0.0
    flip = p > 0.5
    q = (1.0 - p) if flip else p
    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return x if flip else -x


def compute_psr(results: list, n_bars: int) -> None:
    """Compute Probability Sharpe Ratio in-place for each Result.

    PSR(SR*=0) = P(true SR > 0 | observed SR, T bars).
    Uses Mertens (2002) standard error of the Sharpe ratio.
    """
    for r in results:
        sr = r.sharpe_ratio
        if n_bars < 2:
            r.psr = 0.5
            continue
        se = math.sqrt((1.0 + 0.5 * sr * sr) / (n_bars - 1))
        r.psr = _norm_cdf(sr / se) if se > 1e-12 else (1.0 if sr > 0 else 0.0)


def deflated_sharpe_ratio(results: list, n_bars: int) -> None:
    """Compute Deflated Sharpe Ratio (DSR) in-place — corrects for selection bias.

    Bailey & López de Prado (2014). Reduces the effective SR benchmark
    proportionally to sqrt(log(n_trials)).
    """
    n_trials = len(results)
    if n_trials < 2 or n_bars < 2:
        compute_psr(results, n_bars)
        return
    gamma = 0.5772156649  # Euler-Mascheroni constant
    # Expected maximum SR under H0 for n_trials independent strategies
    e_max = ((1.0 - gamma) * _norm_cdf_inv(1.0 - 1.0 / n_trials) +
             gamma * _norm_cdf_inv(1.0 - 1.0 / (n_trials * math.e)))
    for r in results:
        sr = r.sharpe_ratio
        se = math.sqrt((1.0 + 0.5 * sr * sr) / (n_bars - 1))
        r.psr = _norm_cdf((sr - e_max) / se) if se > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Public API for ``from siton.sdk import *``
# ---------------------------------------------------------------------------

__all__ = [
    "Signal", "Strategy",
    "backtest", "run", "clear_sdk_cache",
    "compute_psr", "deflated_sharpe_ratio",
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
