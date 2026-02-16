"""Composite strategies — combine atomic signals with logical operations.

Provides:
  1. Signal combination primitives (numpy vectorized)
  2. Composite factory functions that merge parameter grids
  3. 20 pre-built composite strategies in COMPOSITE_STRATEGIES

All primitives operate on float64 arrays in {-1, 0, 1}.
Composite factories return callables producing (name, grid, signal_fn) tuples,
identical to the contract in indicators.py.
"""

import warnings
from itertools import product

import numpy as np
import numba as nb
import talib

from siton.indicators import (
    _cached, expand_grid,
    crossover, threshold, zero_cross, macd, stochrsi, bollinger,
    breakout, sar_crossover, candlestick,
    obv_crossover, ad_crossover,
    threshold_hlc, threshold_hlcv,
    trend_strength, di_crossover, volatility_breakout,
)


# ---------------------------------------------------------------------------
# Section 1: Signal combination primitives
# ---------------------------------------------------------------------------

def sig_and(a, b):
    """Both must agree: long+long=long, short+short=short, else 0."""
    out = np.where((a == 1.0) & (b == 1.0), 1.0,
          np.where((a == -1.0) & (b == -1.0), -1.0, 0.0))
    return out


def sig_or(a, b):
    """Take first non-zero signal; conflict (long vs short) = 0."""
    out = np.where(a != 0.0, a, b)
    # Resolve conflicts: if both are non-zero and disagree, set to 0
    conflict = (a != 0.0) & (b != 0.0) & (a != b)
    out[conflict] = 0.0
    return out


def sig_not(a):
    """Invert signal: -a."""
    return -a


def sig_filter(sig, cond):
    """sig passes only where cond != 0."""
    out = np.where(cond != 0.0, sig, 0.0)
    return out


def sig_agree(sig, cond):
    """sig passes only where cond agrees on direction."""
    out = np.where(
        ((sig == 1.0) & (cond == 1.0)) | ((sig == -1.0) & (cond == -1.0)),
        sig, 0.0)
    return out


def sig_majority(*sigs):
    """Majority vote: direction with >50% of votes wins."""
    stacked = np.array(sigs, dtype=np.float64)
    n = len(sigs)
    longs = np.sum(stacked == 1.0, axis=0)
    shorts = np.sum(stacked == -1.0, axis=0)
    threshold = n / 2.0
    out = np.where(longs > threshold, 1.0,
          np.where(shorts > threshold, -1.0, 0.0))
    return out


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
            if exit_long[i] != 0.0:
                pos = 0.0
        elif pos == -1.0:
            if exit_short[i] != 0.0:
                pos = 0.0
        out[i] = pos
    return out


# ---------------------------------------------------------------------------
# JIT warmup — compile once at import time
# ---------------------------------------------------------------------------

_w = np.zeros(2, dtype=np.float64)
sig_entry_exit(_w, _w, _w, _w)
sig_confirm(_w, _w, 1)
del _w

# ---------------------------------------------------------------------------
# Section 2: Composite factory functions
# ---------------------------------------------------------------------------

def _prefix_grid(grid, prefix):
    """Add prefix to all grid keys to avoid collisions."""
    return {f"{prefix}{k}": v for k, v in grid.items()}


def _unprefix_params(params, prefix):
    """Strip prefix from param keys."""
    plen = len(prefix)
    return {k[plen:]: v for k, v in params.items() if k.startswith(prefix)}


def _merge_grids(*prefixed_grids):
    """Merge multiple prefixed grids into one."""
    merged = {}
    for g in prefixed_grids:
        merged.update(g)
    return merged


def _warn_grid_size(name, grid):
    """Warn if grid produces >500 combinations."""
    sizes = [len(v) for v in grid.values() if isinstance(v, (list, tuple))]
    total = 1
    for s in sizes:
        total *= s
    if total > 500:
        warnings.warn(f"Composite '{name}': grid produces {total} combinations (>500)")


def composite_and(name, component_a, component_b):
    """AND: both signals must agree."""
    def factory():
        _, grid_a, fn_a = component_a()
        _, grid_b, fn_b = component_b()
        pg_a = _prefix_grid(grid_a, "a_")
        pg_b = _prefix_grid(grid_b, "b_")
        merged = _merge_grids(pg_a, pg_b)
        _warn_grid_size(name, merged)

        def signal_fn(data, **params):
            pa = _unprefix_params(params, "a_")
            pb = _unprefix_params(params, "b_")
            sa = fn_a(data, **pa)
            sb = fn_b(data, **pb)
            return sig_and(sa, sb)

        return name, merged, signal_fn
    return factory


def composite_or(name, component_a, component_b):
    """OR: first non-zero signal; conflicts resolve to 0."""
    def factory():
        _, grid_a, fn_a = component_a()
        _, grid_b, fn_b = component_b()
        pg_a = _prefix_grid(grid_a, "a_")
        pg_b = _prefix_grid(grid_b, "b_")
        merged = _merge_grids(pg_a, pg_b)
        _warn_grid_size(name, merged)

        def signal_fn(data, **params):
            pa = _unprefix_params(params, "a_")
            pb = _unprefix_params(params, "b_")
            sa = fn_a(data, **pa)
            sb = fn_b(data, **pb)
            return sig_or(sa, sb)

        return name, merged, signal_fn
    return factory


def composite_filter(name, signal_src, filter_src):
    """Filter: signal passes only where filter is active."""
    def factory():
        _, grid_s, fn_s = signal_src()
        _, grid_f, fn_f = filter_src()
        pg_s = _prefix_grid(grid_s, "s_")
        pg_f = _prefix_grid(grid_f, "f_")
        merged = _merge_grids(pg_s, pg_f)
        _warn_grid_size(name, merged)

        def signal_fn(data, **params):
            ps = _unprefix_params(params, "s_")
            pf = _unprefix_params(params, "f_")
            ss = fn_s(data, **ps)
            sf = fn_f(data, **pf)
            return sig_filter(ss, sf)

        return name, merged, signal_fn
    return factory


def composite_agree(name, signal_src, filter_src):
    """Agree: signal passes only where filter agrees on direction."""
    def factory():
        _, grid_s, fn_s = signal_src()
        _, grid_f, fn_f = filter_src()
        pg_s = _prefix_grid(grid_s, "s_")
        pg_f = _prefix_grid(grid_f, "f_")
        merged = _merge_grids(pg_s, pg_f)
        _warn_grid_size(name, merged)

        def signal_fn(data, **params):
            ps = _unprefix_params(params, "s_")
            pf = _unprefix_params(params, "f_")
            ss = fn_s(data, **ps)
            sf = fn_f(data, **pf)
            return sig_agree(ss, sf)

        return name, merged, signal_fn
    return factory


def composite_majority(name, *components):
    """Majority vote: direction with >50% wins."""
    def factory():
        fns = []
        merged = {}
        for idx, comp in enumerate(components):
            _, grid_c, fn_c = comp()
            prefix = f"c{idx}_"
            pg = _prefix_grid(grid_c, prefix)
            merged.update(pg)
            fns.append((prefix, fn_c))
        _warn_grid_size(name, merged)

        def signal_fn(data, **params):
            sigs = []
            for prefix, fn in fns:
                pc = _unprefix_params(params, prefix)
                sigs.append(fn(data, **pc))
            return sig_majority(*sigs)

        return name, merged, signal_fn
    return factory


def composite_confirm(name, signal_src, condition_src, lookbacks=(3, 5, 8)):
    """Confirm: signal valid only if condition fired in last N bars."""
    def factory():
        _, grid_s, fn_s = signal_src()
        _, grid_c, fn_c = condition_src()
        pg_s = _prefix_grid(grid_s, "s_")
        pg_c = _prefix_grid(grid_c, "c_")
        merged = _merge_grids(pg_s, pg_c)
        merged["lookback"] = list(lookbacks)
        _warn_grid_size(name, merged)

        def signal_fn(data, lookback, **params):
            ps = _unprefix_params(params, "s_")
            pc = _unprefix_params(params, "c_")
            ss = fn_s(data, **ps)
            sc = fn_c(data, **pc)
            return sig_confirm(ss, sc, lookback)

        return name, merged, signal_fn
    return factory


def composite_entry_exit(name, entry_long_src, exit_long_src,
                         entry_short_src=None, exit_short_src=None):
    """Entry/exit state machine with asymmetric signals."""
    def factory():
        _, grid_el, fn_el = entry_long_src()
        _, grid_xl, fn_xl = exit_long_src()
        pg_el = _prefix_grid(grid_el, "el_")
        pg_xl = _prefix_grid(grid_xl, "xl_")
        merged = _merge_grids(pg_el, pg_xl)

        has_short = entry_short_src is not None
        fn_es = fn_xl  # default: exit long signal also used as entry short
        fn_xs = fn_el  # default: entry long signal also used as exit short
        if has_short:
            _, grid_es, fn_es = entry_short_src()
            pg_es = _prefix_grid(grid_es, "es_")
            merged.update(pg_es)
        if exit_short_src is not None:
            _, grid_xs, fn_xs = exit_short_src()
            pg_xs = _prefix_grid(grid_xs, "xs_")
            merged.update(pg_xs)

        _warn_grid_size(name, merged)

        if has_short and exit_short_src is not None:
            def signal_fn(data, **params):
                pel = _unprefix_params(params, "el_")
                pxl = _unprefix_params(params, "xl_")
                pes = _unprefix_params(params, "es_")
                pxs = _unprefix_params(params, "xs_")
                s_el = fn_el(data, **pel)
                s_xl = fn_xl(data, **pxl)
                s_es = fn_es(data, **pes)
                s_xs = fn_xs(data, **pxs)
                # entry_long: where s_el == 1
                # exit_long: where s_xl == -1 (signal flips)
                # entry_short: where s_es == -1
                # exit_short: where s_xs == 1 (signal flips)
                return sig_entry_exit(s_el, s_xl, s_es, s_xs)
        elif has_short:
            def signal_fn(data, **params):
                pel = _unprefix_params(params, "el_")
                pxl = _unprefix_params(params, "xl_")
                pes = _unprefix_params(params, "es_")
                s_el = fn_el(data, **pel)
                s_xl = fn_xl(data, **pxl)
                s_es = fn_es(data, **pes)
                # exit short mirrors entry long
                return sig_entry_exit(s_el, s_xl, s_es, s_el)
        else:
            def signal_fn(data, **params):
                pel = _unprefix_params(params, "el_")
                pxl = _unprefix_params(params, "xl_")
                s_el = fn_el(data, **pel)
                s_xl = fn_xl(data, **pxl)
                # Mirror: entry short = exit long signal, exit short = entry long signal
                return sig_entry_exit(s_el, s_xl, s_xl, s_el)

        return name, merged, signal_fn
    return factory


# ---------------------------------------------------------------------------
# Section 3: 20 pre-built composite strategies
# ---------------------------------------------------------------------------

# Helper: atomic factories used as building blocks.
# We call the factory *constructors* from indicators/strategies here —
# they return factory callables that produce (name, grid, signal_fn).

def _ema_cross(fast=(8, 12), slow=(26, 50)):
    return crossover("_ema", talib.EMA, fast=list(fast), slow=list(slow))

def _sma_cross(fast=(10, 20), slow=(50, 100)):
    return crossover("_sma", talib.SMA, fast=list(fast), slow=list(slow))

def _dema_cross(fast=(8, 12), slow=(26, 50)):
    return crossover("_dema", talib.DEMA, fast=list(fast), slow=list(slow))

def _tema_cross(fast=(8, 12), slow=(26, 50)):
    return crossover("_tema", talib.TEMA, fast=list(fast), slow=list(slow))

def _rsi(periods=(14,), oversold=(30,), overbought=(70,)):
    return threshold("_rsi", talib.RSI, periods=list(periods),
                     oversold=list(oversold), overbought=list(overbought))

def _macd_sig(fast=(12,), slow=(26,), signal_period=(9,)):
    return macd("_macd", fast=list(fast), slow=list(slow),
                signal_period=list(signal_period))

def _bollinger(windows=(20,), num_std=(2.0,)):
    return bollinger("_bb", windows=list(windows), num_std=list(num_std))

def _breakout(periods=(20,)):
    return breakout("_brk", periods=list(periods))

def _mom(periods=(10, 14)):
    return zero_cross("_mom", talib.MOM, periods=list(periods))

def _roc(periods=(10, 14)):
    return zero_cross("_roc", talib.ROC, periods=list(periods))

def _trix(periods=(14,)):
    return zero_cross("_trix", talib.TRIX, periods=list(periods))

def _stochrsi(periods=(14,), fastk=(3,), fastd=(3,),
              oversold=(20,), overbought=(80,)):
    return stochrsi("_stochrsi", periods=list(periods), fastk=list(fastk),
                    fastd=list(fastd), oversold=list(oversold),
                    overbought=list(overbought))

def _cci(periods=(14, 20), oversold=(-100,), overbought=(100,)):
    return threshold_hlc("_cci", talib.CCI, periods=list(periods),
                         oversold=list(oversold), overbought=list(overbought))

def _mfi(periods=(14,), oversold=(20, 30), overbought=(70, 80)):
    return threshold_hlcv("_mfi", talib.MFI, periods=list(periods),
                          oversold=list(oversold), overbought=list(overbought))


# --- ADX filter: trend strength > threshold ---
def _adx_filter(periods=(14,), thresholds=(25,)):
    return trend_strength("_adx", talib.ADX, periods=list(periods),
                          thresholds=list(thresholds))

# --- DI crossover ---
def _di_cross(periods=(14,)):
    return di_crossover("_di", periods=list(periods))

# --- OBV crossover ---
def _obv(ma_periods=(20,)):
    return obv_crossover("_obv", ma_periods=list(ma_periods))

# --- AD crossover ---
def _ad(ma_periods=(20,)):
    return ad_crossover("_ad", ma_periods=list(ma_periods))

# --- SAR ---
def _sar(accelerations=(0.02,), maximums=(0.2,)):
    return sar_crossover("_sar", accelerations=list(accelerations),
                         maximums=list(maximums))

# --- Candlestick engulfing ---
def _engulfing():
    return candlestick("_cdl_eng", talib.CDLENGULFING)

# --- ATR breakout ---
def _atr_breakout(periods=(14,), lookbacks=(20,)):
    return volatility_breakout("_atr_brk", talib.ATR,
                               periods=list(periods), lookbacks=list(lookbacks))


# ===================================================================
# Trend-following with filter (5)
# ===================================================================

# 1. EMA_Cross+ADX — EMA crossover filtered by ADX trend strength
#    Grid: 2 fast x 3 slow x 2 period x 2 threshold = 24
comp_ema_adx = composite_filter(
    "EMA_Cross+ADX",
    _ema_cross(fast=(8, 12), slow=(26, 50, 100)),
    _adx_filter(periods=(14, 20), thresholds=(20, 25)),
)

# 2. MACD+ADX — MACD gated by ADX
#    Grid: 2 fast x 1 slow x 1 sig x 1 period x 3 threshold = 6
comp_macd_adx = composite_filter(
    "MACD+ADX",
    _macd_sig(fast=(10, 12), slow=(26,), signal_period=(9,)),
    _adx_filter(periods=(14,), thresholds=(20, 25, 30)),
)

# 3. EMA_Cross+OBV — MA crossover AND volume OBV
#    Grid: 2 fast x 2 slow x 2 ma = 8
comp_ema_obv = composite_and(
    "EMA_Cross+OBV",
    _ema_cross(fast=(8, 12), slow=(26, 50)),
    _obv(ma_periods=(20, 50)),
)

# 4. SMA_Cross+DI — SMA crossover filtered by +DI/-DI agreement
#    Grid: 2 fast x 2 slow x 2 period = 8
comp_sma_di = composite_agree(
    "SMA_Cross+DI",
    _sma_cross(fast=(10, 20), slow=(50, 100)),
    _di_cross(periods=(14, 20)),
)

# 5. TEMA_Cross+SAR — TEMA crossover AND SAR agreement
#    Grid: 2 fast x 1 slow x 2 accel = 4
comp_tema_sar = composite_and(
    "TEMA_Cross+SAR",
    _tema_cross(fast=(8, 12), slow=(50,)),
    _sar(accelerations=(0.02, 0.03), maximums=(0.2,)),
)


# ===================================================================
# Mean-reversion with confirmation (4)
# ===================================================================

# 6. RSI+BB — RSI oversold/overbought AND Bollinger Bands
#    Grid: 2 period x 2 os x 1 window x 2 std = 8
comp_rsi_bb = composite_and(
    "RSI+BB",
    _rsi(periods=(14, 21), oversold=(30,), overbought=(70,)),
    _bollinger(windows=(20,), num_std=(2.0, 2.5)),
)

# 7. RSI+Engulfing — Candlestick engulfing confirmed by recent RSI
#    Grid: 1 x 2 period x 2 os x 3 lookback = 12
comp_rsi_engulfing = composite_confirm(
    "RSI+Engulfing",
    _engulfing(),
    _rsi(periods=(14, 21), oversold=(25, 30), overbought=(70, 75)),
    lookbacks=(3, 5, 8),
)

# 8. StochRSI+CCI — Stochastic RSI AND CCI double confirmation
#    Grid: 2 os x 2 ob x 1 period x 1 os = 4 ... adjusted for ~8
comp_stochrsi_cci = composite_and(
    "StochRSI+CCI",
    _stochrsi(periods=(14,), fastk=(3,), fastd=(3,),
              oversold=(20, 25), overbought=(75, 80)),
    _cci(periods=(14, 20), oversold=(-100,), overbought=(100,)),
)

# 9. CCI+MFI — CCI mean-reversion AND MFI volume confirmation
#    Grid: 2 period x 2 os x 2 ob x 2 period x 2 os x 2 ob = 32 ... trim
comp_cci_mfi = composite_and(
    "CCI+MFI",
    _cci(periods=(14, 20), oversold=(-100, -150), overbought=(100, 150)),
    _mfi(periods=(14,), oversold=(20, 30), overbought=(70, 80)),
)


# ===================================================================
# Multi-indicator voting (3)
# ===================================================================

# 10. Triple_MA_Vote — Majority vote: SMA, EMA, DEMA crossovers
#     Grid: all fixed params = 1
comp_triple_ma = composite_majority(
    "Triple_MA_Vote",
    _sma_cross(fast=(10,), slow=(50,)),
    _ema_cross(fast=(12,), slow=(26,)),
    _dema_cross(fast=(12,), slow=(26,)),
)

# 11. Momentum_Vote — Majority vote: ROC, MOM, TRIX
#     Grid: 2 x 2 x 2 = 8
comp_momentum_vote = composite_majority(
    "Momentum_Vote",
    _roc(periods=(10, 14)),
    _mom(periods=(10, 14)),
    _trix(periods=(10, 14)),
)

# 12. Trend_Vote — Majority vote: EMA cross, MACD, DI cross
#     Grid: all fixed = 1
comp_trend_vote = composite_majority(
    "Trend_Vote",
    _ema_cross(fast=(12,), slow=(26,)),
    _macd_sig(fast=(12,), slow=(26,), signal_period=(9,)),
    _di_cross(periods=(14,)),
)


# ===================================================================
# Breakout with volume/volatility confirmation (3)
# ===================================================================

# 13. Breakout+OBV — Channel breakout filtered by OBV trend
#     Grid: 3 period x 2 ma = 6
comp_brk_obv = composite_filter(
    "Breakout+OBV",
    _breakout(periods=(14, 20, 30)),
    _obv(ma_periods=(20, 50)),
)

# 14. Breakout+ATR — Channel breakout gated by ATR expanding
#     Grid: 3 period x 1 = 3
comp_brk_atr = composite_filter(
    "Breakout+ATR",
    _breakout(periods=(14, 20, 30)),
    _atr_breakout(periods=(14,), lookbacks=(20,)),
)

# 15. BB_Break+Vol — Bollinger breakout filtered by AD volume
#     Grid: 2 window x 1 std x 2 ma = 4
comp_bb_vol = composite_filter(
    "BB_Break+Vol",
    _bollinger(windows=(20, 30), num_std=(2.0,)),
    _ad(ma_periods=(20, 50)),
)


# ===================================================================
# Entry/exit asymmetric (3)
# ===================================================================

# 16. RSI_In+MACD_Out — Entry on RSI oversold, exit on MACD cross
#     Grid: 2 os x 2 fast = 4
comp_rsi_macd_ee = composite_entry_exit(
    "RSI_In+MACD_Out",
    entry_long_src=_rsi(periods=(14,), oversold=(25, 30), overbought=(70, 75)),
    exit_long_src=_macd_sig(fast=(10, 12), slow=(26,), signal_period=(9,)),
)

# 17. CDL_In+SAR_Out — Entry on candlestick, exit on SAR reversal
#     Grid: 1
comp_cdl_sar_ee = composite_entry_exit(
    "CDL_In+SAR_Out",
    entry_long_src=_engulfing(),
    exit_long_src=_sar(accelerations=(0.02,), maximums=(0.2,)),
)

# 18. Stoch_In+EMA_Out — Entry on stochastic oversold, exit on EMA cross
#     Grid: 2 os x 2 slow = 4
comp_stoch_ema_ee = composite_entry_exit(
    "Stoch_In+EMA_Out",
    entry_long_src=_stochrsi(periods=(14,), fastk=(3,), fastd=(3,),
                             oversold=(20, 25), overbought=(75, 80)),
    exit_long_src=_ema_cross(fast=(12,), slow=(26, 50)),
)


# ===================================================================
# Temporal/sequence (2)
# ===================================================================

# 19. RSI_Then_MACD — RSI oversold, then MACD cross within N bars
#     Grid: 2 period x 2 os x 2 fast x 2 lookback = 16
comp_rsi_then_macd = composite_confirm(
    "RSI_Then_MACD",
    _macd_sig(fast=(10, 12), slow=(26,), signal_period=(9,)),
    _rsi(periods=(14, 21), oversold=(25, 30), overbought=(70, 75)),
    lookbacks=(5, 8),
)

# 20. Breakout+MOM — Breakout then momentum confirmation within N bars
#     Grid: 3 period x 2 period x 2 lookback = 12
comp_brk_mom = composite_confirm(
    "Breakout+MOM",
    _breakout(periods=(14, 20, 30)),
    _mom(periods=(10, 14)),
    lookbacks=(5, 8),
)


# ===================================================================
# Registry
# ===================================================================

COMPOSITE_STRATEGIES = [
    # Trend-following with filter
    comp_ema_adx,           # 1
    comp_macd_adx,          # 2
    comp_ema_obv,           # 3
    comp_sma_di,            # 4
    comp_tema_sar,          # 5
    # Mean-reversion with confirmation
    comp_rsi_bb,            # 6
    comp_rsi_engulfing,     # 7
    comp_stochrsi_cci,      # 8
    comp_cci_mfi,           # 9
    # Multi-indicator voting
    comp_triple_ma,         # 10
    comp_momentum_vote,     # 11
    comp_trend_vote,        # 12
    # Breakout with confirmation
    comp_brk_obv,           # 13
    comp_brk_atr,           # 14
    comp_bb_vol,            # 15
    # Entry/exit asymmetric
    comp_rsi_macd_ee,       # 16
    comp_cdl_sar_ee,        # 17
    comp_stoch_ema_ee,      # 18
    # Temporal/sequence
    comp_rsi_then_macd,     # 19
    comp_brk_mom,           # 20
]
