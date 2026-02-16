"""Backtesting engine — Numba JIT parallel across all CPU cores.

Portfolio model: tracks cash + signed shares (positive=long, negative=short).
Equity at any point = cash + shares * current_price.
Supports fixed-fraction position sizing and per-trade slippage.
"""

import numpy as np
import numba as nb
from dataclasses import dataclass


@dataclass
class Result:
    strategy: str
    params: dict
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    profit_factor: float


_SQRT_8760 = np.float64(8760.0) ** 0.5


@nb.njit(cache=True)
def _backtest_one(close, sig, fee, slippage, n, initial_capital, fraction):
    """Single backtest with realistic portfolio tracking.

    Portfolio state: cash + shares (signed).
    On signal change: close old position, open new at close[i] with slippage+fee.
    Equity tracked bar-by-bar for Sharpe / drawdown.
    """
    cash = initial_capital
    shares = 0.0
    prev_sig = 0.0

    running_max = initial_capital
    max_dd = 0.0
    win_sum = 0.0
    loss_sum = 0.0
    win_count = nb.int64(0)
    loss_count = nb.int64(0)
    num_trades = nb.int64(0)
    mean_acc = 0.0
    m2_acc = 0.0
    bar_count = nb.int64(0)

    # OPT-2: Pre-compute fee/slippage factors
    slip_buy = 1.0 + slippage
    slip_sell = 1.0 - slippage
    cost_mul = 1.0 + fee
    recv_mul = 1.0 - fee

    # OPT-1: Carry forward equity — equity_after[i-1] == equity_before[i]
    equity = initial_capital

    for i in range(n):
        equity_before = equity
        if equity_before <= 0.0:
            break

        # --- Signal change: close old, open new ---
        if sig[i] != prev_sig:
            # Close existing position
            if shares > 0.0:
                # Long exit: SELL
                fill = close[i] * slip_sell
                proceeds = shares * fill
                cash += proceeds * recv_mul
                shares = 0.0
            elif shares < 0.0:
                # Short exit: BUY to cover
                fill = close[i] * slip_buy
                cost = (-shares) * fill
                cash -= cost * cost_mul
                shares = 0.0

            # Open new position
            equity_now = cash  # flat after closing
            if sig[i] != 0.0 and equity_now > 0.0:
                trade_value = equity_now * fraction
                if sig[i] > 0.0:
                    # Long entry: BUY  (OPT-3: skip qty*fill)
                    fill = close[i] * slip_buy
                    qty = trade_value / fill
                    cash -= trade_value * cost_mul
                    shares = qty
                else:
                    # Short entry: SELL  (OPT-3: skip qty*fill)
                    fill = close[i] * slip_sell
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                num_trades += 1

        prev_sig = sig[i]

        # --- End-of-bar equity (OPT-1: carried to next bar) ---
        equity = cash + shares * close[i + 1]

        # Bar return (OPT-4: unconditional — blown-account break guarantees > 0)
        sr = (equity - equity_before) / equity_before

        # Drawdown (OPT-4: unconditional — running_max always positive)
        if equity > running_max:
            running_max = equity
        dd = (equity - running_max) / running_max
        if dd < max_dd:
            max_dd = dd

        # Win/loss
        if sr > 0.0:
            win_sum += sr
            win_count += 1
        elif sr < 0.0:
            loss_sum += sr
            loss_count += 1

        # Welford's online variance
        bar_count += 1
        k = nb.float64(bar_count)
        delta = sr - mean_acc
        mean_acc += delta / k
        m2_acc += delta * (sr - mean_acc)

    # Final metrics
    final_equity = cash + shares * close[min(n, len(close) - 1)]
    total_return = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital > 0.0 else 0.0

    std_r = (m2_acc / bar_count) ** 0.5 if bar_count > 1 else 0.0
    sharpe = (mean_acc / std_r * _SQRT_8760) if std_r > 0.0 else 0.0

    total_nonzero = win_count + loss_count
    win_rate = (nb.float64(win_count) / nb.float64(total_nonzero) * 100.0) if total_nonzero > 0 else 0.0

    gross_loss = -loss_sum if loss_sum < 0.0 else 1e-10
    profit_factor = win_sum / gross_loss if gross_loss > 1e-10 else win_sum / 1e-10

    return total_return, sharpe, max_dd * 100.0, win_rate, nb.float64(num_trades), profit_factor


@nb.njit(parallel=True, cache=True)
def backtest_batch(close, signals_2d, fee, slippage, initial_capital, fraction):
    """Run ALL backtests in parallel across CPU cores.

    Args:
        close: 1D array of close prices (length M).
        signals_2d: 2D array (N x M) of signals, one row per combo.
        fee: fee as decimal (e.g. 0.00075).
        slippage: slippage as decimal (e.g. 0.0005).
        initial_capital: starting cash (e.g. 10000.0).
        fraction: fraction of equity per trade, 0-1 (e.g. 1.0).

    Returns:
        2D array (N x 6) of metrics per combo.
    """
    n_combos = signals_2d.shape[0]
    n = len(close) - 1
    out = np.empty((n_combos, 6))

    for c in nb.prange(n_combos):
        r0, r1, r2, r3, r4, r5 = _backtest_one(
            close, signals_2d[c], fee, slippage, n, initial_capital, fraction)
        out[c, 0] = r0
        out[c, 1] = r1
        out[c, 2] = r2
        out[c, 3] = r3
        out[c, 4] = r4
        out[c, 5] = r5

    return out


@nb.njit(cache=True)
def _backtest_one_managed(close, high, low, sig, fee, slippage, n,
                          initial_capital, fraction, sl_pct, tp_pct, trail_pct):
    """Single backtest with portfolio tracking + position management.

    Stop priority on each bar (conservative): stop-loss -> trailing stop -> take-profit.
    Position sizing: equity * fraction.
    """
    cash = initial_capital
    shares = 0.0       # signed: +N long, -N short
    prev_sig = 0.0

    running_max = initial_capital
    max_dd = 0.0
    win_sum = 0.0
    loss_sum = 0.0
    win_count = nb.int64(0)
    loss_count = nb.int64(0)
    num_trades = nb.int64(0)
    mean_acc = 0.0
    m2_acc = 0.0
    bar_count = nb.int64(0)

    # Position management state
    entry_price = 0.0
    peak_price = 0.0    # best price since entry (for trailing stop)
    # OPT-5: Pre-computed stop levels (set when position opens)
    sl_level = 0.0
    tp_level = 0.0

    # OPT-2: Pre-compute fee/slippage factors
    slip_buy = 1.0 + slippage
    slip_sell = 1.0 - slippage
    cost_mul = 1.0 + fee
    recv_mul = 1.0 - fee

    # OPT-1: Carry forward equity
    equity = initial_capital

    for i in range(n):
        equity_before = equity
        if equity_before <= 0.0:
            break

        cur_sig = sig[i]
        next_close = close[i + 1]
        next_high = high[i + 1]
        next_low = low[i + 1]
        stopped = False

        # --- Check stops on open position ---
        if shares != 0.0:
            exit_price = 0.0

            if shares > 0.0:
                # Long position stops
                # Stop-loss (OPT-5: sl_level pre-computed)
                if sl_pct > 0.0 and next_low <= sl_level:
                    exit_price = sl_level
                    stopped = True

                # Trailing stop
                if not stopped and trail_pct > 0.0:
                    if next_high > peak_price:
                        peak_price = next_high
                    trail_level = peak_price * (1.0 - trail_pct)
                    if next_low <= trail_level:
                        exit_price = trail_level
                        stopped = True

                # Take-profit (OPT-5: tp_level pre-computed)
                if not stopped and tp_pct > 0.0 and next_high >= tp_level:
                    exit_price = tp_level
                    stopped = True

            else:
                # Short position stops
                # Stop-loss (OPT-5: sl_level pre-computed)
                if sl_pct > 0.0 and next_high >= sl_level:
                    exit_price = sl_level
                    stopped = True

                # Trailing stop
                if not stopped and trail_pct > 0.0:
                    if next_low < peak_price:
                        peak_price = next_low
                    trail_level = peak_price * (1.0 + trail_pct)
                    if next_high >= trail_level:
                        exit_price = trail_level
                        stopped = True

                # Take-profit (OPT-5: tp_level pre-computed)
                if not stopped and tp_pct > 0.0 and next_low <= tp_level:
                    exit_price = tp_level
                    stopped = True

            if stopped:
                # Close position at stop price with slippage + fee
                if shares > 0.0:
                    fill = exit_price * slip_sell
                    proceeds = shares * fill
                    cash += proceeds * recv_mul
                else:
                    fill = exit_price * slip_buy
                    cost = (-shares) * fill
                    cash -= cost * cost_mul
                shares = 0.0
                entry_price = 0.0
                peak_price = 0.0

        # --- Signal change: close old position and/or open new ---
        if cur_sig != prev_sig and not (shares == 0.0 and stopped):
            # Close existing position at close[i] if still open
            if shares != 0.0:
                if shares > 0.0:
                    fill = close[i] * slip_sell
                    proceeds = shares * fill
                    cash += proceeds * recv_mul
                else:
                    fill = close[i] * slip_buy
                    cost = (-shares) * fill
                    cash -= cost * cost_mul
                shares = 0.0
                entry_price = 0.0
                peak_price = 0.0

            # Open new position
            equity_now = cash
            if cur_sig != 0.0 and equity_now > 0.0:
                trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    # Long entry: BUY  (OPT-3: skip qty*fill)
                    fill = close[i] * slip_buy
                    qty = trade_value / fill
                    cash -= trade_value * cost_mul
                    shares = qty
                else:
                    # Short entry: SELL  (OPT-3: skip qty*fill)
                    fill = close[i] * slip_sell
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = close[i]
                peak_price = close[i]
                # OPT-5: Pre-compute stop levels for this position
                if cur_sig > 0.0:
                    sl_level = entry_price * (1.0 - sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 + tp_pct) if tp_pct > 0.0 else 0.0
                else:
                    sl_level = entry_price * (1.0 + sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 - tp_pct) if tp_pct > 0.0 else 0.0
                num_trades += 1

        # If stopped and signal wants re-entry, re-enter at next_close
        if stopped and cur_sig != 0.0 and shares == 0.0:
            equity_now = cash
            if equity_now > 0.0:
                trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    # Long entry: BUY  (OPT-3: skip qty*fill)
                    fill = next_close * slip_buy
                    qty = trade_value / fill
                    cash -= trade_value * cost_mul
                    shares = qty
                else:
                    # Short entry: SELL  (OPT-3: skip qty*fill)
                    fill = next_close * slip_sell
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = next_close
                peak_price = next_close
                # OPT-5: Pre-compute stop levels for this position
                if cur_sig > 0.0:
                    sl_level = entry_price * (1.0 - sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 + tp_pct) if tp_pct > 0.0 else 0.0
                else:
                    sl_level = entry_price * (1.0 + sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 - tp_pct) if tp_pct > 0.0 else 0.0
                num_trades += 1

        prev_sig = cur_sig

        # --- End-of-bar equity (OPT-1: carried to next bar) ---
        equity = cash + shares * next_close

        # Bar return (OPT-4: unconditional — blown-account break guarantees > 0)
        sr = (equity - equity_before) / equity_before

        # Drawdown (OPT-4: unconditional — running_max always positive)
        if equity > running_max:
            running_max = equity
        dd = (equity - running_max) / running_max
        if dd < max_dd:
            max_dd = dd

        # Win/loss
        if sr > 0.0:
            win_sum += sr
            win_count += 1
        elif sr < 0.0:
            loss_sum += sr
            loss_count += 1

        # Welford's online variance
        bar_count += 1
        k = nb.float64(bar_count)
        delta = sr - mean_acc
        mean_acc += delta / k
        m2_acc += delta * (sr - mean_acc)

    # Final metrics
    final_equity = cash + shares * close[min(n, len(close) - 1)]
    total_return = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital > 0.0 else 0.0

    std_r = (m2_acc / bar_count) ** 0.5 if bar_count > 1 else 0.0
    sharpe = (mean_acc / std_r * _SQRT_8760) if std_r > 0.0 else 0.0

    total_nonzero = win_count + loss_count
    win_rate = (nb.float64(win_count) / nb.float64(total_nonzero) * 100.0) if total_nonzero > 0 else 0.0

    gross_loss = -loss_sum if loss_sum < 0.0 else 1e-10
    profit_factor = win_sum / gross_loss if gross_loss > 1e-10 else win_sum / 1e-10

    return total_return, sharpe, max_dd * 100.0, win_rate, nb.float64(num_trades), profit_factor


@nb.njit(parallel=True, cache=True)
def backtest_batch_managed(close, high, low, signals_2d, fee, slippage,
                           initial_capital, fraction,
                           sig_indices, sl_arr, tp_arr, trail_arr):
    """Run ALL managed backtests in parallel across CPU cores.

    Args:
        close: 1D array of close prices (length M).
        high: 1D array of high prices (length M).
        low: 1D array of low prices (length M).
        signals_2d: 2D array (N_signals x M) of signals.
        fee: fee as decimal (e.g. 0.00075).
        slippage: slippage as decimal (e.g. 0.0005).
        initial_capital: starting cash (e.g. 10000.0).
        fraction: fraction of equity per trade, 0-1.
        sig_indices: 1D int64 array (N_combos,) mapping each combo to a signal row.
        sl_arr: 1D float64 array (N_combos,) stop-loss pct per combo.
        tp_arr: 1D float64 array (N_combos,) take-profit pct per combo.
        trail_arr: 1D float64 array (N_combos,) trailing stop pct per combo.

    Returns:
        2D array (N_combos x 6) of metrics per combo.
    """
    n_combos = len(sig_indices)
    n = len(close) - 1
    out = np.empty((n_combos, 6))

    for c in nb.prange(n_combos):
        si = sig_indices[c]
        r0, r1, r2, r3, r4, r5 = _backtest_one_managed(
            close, high, low, signals_2d[si], fee, slippage, n,
            initial_capital, fraction, sl_arr[c], tp_arr[c], trail_arr[c])
        out[c, 0] = r0
        out[c, 1] = r1
        out[c, 2] = r2
        out[c, 3] = r3
        out[c, 4] = r4
        out[c, 5] = r5

    return out


# Warmup JIT at import time
_dummy_close = np.array([1.0, 1.01, 1.02], dtype=np.float64)
_dummy_high = np.array([1.01, 1.02, 1.03], dtype=np.float64)
_dummy_low = np.array([0.99, 1.00, 1.01], dtype=np.float64)
_dummy_sigs = np.zeros((1, 3), dtype=np.float64)
backtest_batch(_dummy_close, _dummy_sigs, 0.001, 0.0005, 10000.0, 1.0)
backtest_batch_managed(
    _dummy_close, _dummy_high, _dummy_low, _dummy_sigs, 0.001, 0.0005,
    10000.0, 1.0,
    np.zeros(1, dtype=np.int64),
    np.zeros(1, dtype=np.float64),
    np.zeros(1, dtype=np.float64),
    np.zeros(1, dtype=np.float64),
)
del _dummy_close, _dummy_high, _dummy_low, _dummy_sigs


def rank_results(results: list[Result], sort_by: str = "sharpe_ratio") -> list[Result]:
    """Sort results by chosen metric, descending."""
    return sorted(results, key=lambda r: getattr(r, sort_by), reverse=True)
