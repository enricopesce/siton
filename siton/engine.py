"""Backtesting engine — Numba JIT parallel across all CPU cores.

Portfolio model: tracks cash + signed shares (positive=long, negative=short).
Equity at any point = cash + shares * current_price.
Supports fixed-fraction position sizing and per-trade slippage.

Execution convention (professional standard):
  signal from close[i-1]  →  execute at open[i]  →  mark at close[i]
  Stops and intrabar drawdown use high[i] / low[i] (same bar as entry).

Performance tuning:
- error_model='numpy': skips division-by-zero traps (safe: we guard with if-checks)
- inline='always': inlines inner functions into prange loop for better LLVM optimization
- parallel=True with prange: distributes combos across all CPU cores
- cache=True: avoids recompilation on subsequent imports
- Float accumulators: avoids int→float casts in hot loop (Welford, win/loss)
"""

from dataclasses import dataclass

import numba as nb
import numpy as np

# ---------------------------------------------------------------------------
# Numba threading configuration — maximize CPU utilization
# ---------------------------------------------------------------------------
_NUM_CORES = nb.config.NUMBA_DEFAULT_NUM_THREADS
nb.set_num_threads(_NUM_CORES)


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
    equity_curve: np.ndarray | None = None
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    psr: float = 0.0


@nb.njit(cache=True, error_model="numpy")
def _backtest_one(
    close,
    high,
    low,
    open_arr,
    sig,
    fee,
    slippage,
    n,
    initial_capital,
    fraction,
    ann_factor,
    rf_per_bar,
):
    """Single backtest with realistic portfolio tracking and trade-level metrics."""
    cash = initial_capital
    shares = 0.0
    prev_sig = 0.0

    running_max = initial_capital
    max_dd = 0.0
    fbar = 0.0
    fbar_total = 0.0
    num_trades = 0.0
    mean_acc = 0.0
    m2_acc = 0.0

    # Downside variance accumulators for Sortino (BUG-06)
    mean_neg = 0.0
    m2_neg = 0.0
    fbar_neg = 0.0

    # Trade-level P&L tracking
    trade_entry_equity = 0.0
    trade_win_count = 0.0
    trade_loss_count = 0.0
    trade_win_sum = 0.0
    trade_loss_sum = 0.0

    # Pre-compute fee/slippage factors
    slip_buy = 1.0 + slippage
    slip_sell = 1.0 - slippage
    cost_mul = 1.0 + fee
    recv_mul = 1.0 - fee

    equity = initial_capital

    for i in range(n):
        equity_before = equity
        if equity_before <= 0.0:
            break

        inv_equity_before = 1.0 / equity_before

        # --- Signal change: close old, open new at open[i] ---
        cur_sig = sig[i]
        if cur_sig != prev_sig:
            # Close existing position
            if shares > 0.0:
                fill = open_arr[i] * slip_sell
                cash += shares * fill * recv_mul
                shares = 0.0
            elif shares < 0.0:
                fill = open_arr[i] * slip_buy
                cash -= (-shares) * fill * cost_mul
                shares = 0.0

            # Record trade P&L when closing a position (prev_sig != 0 means we had one)
            if prev_sig != 0.0:
                trade_pnl = cash - trade_entry_equity
                if trade_pnl > 0.0:
                    trade_win_count += 1.0
                    trade_win_sum += trade_pnl
                elif trade_pnl < 0.0:
                    trade_loss_count += 1.0
                    trade_loss_sum += trade_pnl

            # Open new position at open[i]
            if cur_sig != 0.0 and cash > 0.0:
                trade_entry_equity = cash  # record equity at trade entry
                trade_value = cash * fraction
                if cur_sig > 0.0:
                    # BUG-01 fix: fee embedded in share price, spend exactly trade_value
                    fill = open_arr[i] * slip_buy
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    fill = open_arr[i] * slip_sell
                    shares = -(trade_value / fill)
                    cash += trade_value * recv_mul
                num_trades += 1.0
            prev_sig = cur_sig

        # --- End-of-bar equity (mark at close[i]) ---
        equity = cash + shares * close[i]

        # Bar return for Sharpe calculation
        sr = (equity - equity_before) * inv_equity_before

        # BUG-08: Intrabar worst equity for drawdown (long: use low, short: use high)
        if shares > 0.0:
            intrabar_eq = cash + shares * low[i]
        elif shares < 0.0:
            intrabar_eq = cash + shares * high[i]
        else:
            intrabar_eq = equity

        # Margin call: forced liquidation if short equity hits 0 intrabar
        if shares < 0.0 and intrabar_eq <= 0.0:
            bankrupt_price = cash / (-shares)
            fill = bankrupt_price * slip_buy
            cash -= (-shares) * fill * cost_mul
            cash = max(cash, 0.0)
            trade_pnl = cash - trade_entry_equity
            if trade_pnl > 0.0:
                trade_win_count += 1.0
                trade_win_sum += trade_pnl
            elif trade_pnl < 0.0:
                trade_loss_count += 1.0
                trade_loss_sum += trade_pnl
            shares = 0.0
            equity = cash
            sr = (equity - equity_before) * inv_equity_before
            intrabar_eq = equity

        if equity > running_max:
            running_max = equity
        dd = (intrabar_eq - running_max) / running_max
        if dd < max_dd:
            max_dd = dd

        # Welford's online variance — active bars only (exclude flat-period zero-returns)
        fbar_total += 1.0
        is_active = (cur_sig != 0.0) or (shares != 0.0)
        if is_active:
            excess_r = sr - rf_per_bar
            fbar += 1.0
            delta = excess_r - mean_acc
            mean_acc += delta / fbar
            m2_acc += delta * (excess_r - mean_acc)

            if excess_r < 0.0:
                neg_r = -excess_r
                fbar_neg += 1.0
                delta_neg = neg_r - mean_neg
                mean_neg += delta_neg / fbar_neg
                m2_neg += delta_neg * (neg_r - mean_neg)

    # Close remaining position notionally for trade P&L
    final_equity = equity if fbar_total > 0.0 else initial_capital
    if shares != 0.0:
        trade_pnl = final_equity - trade_entry_equity
        if trade_pnl > 0.0:
            trade_win_count += 1.0
            trade_win_sum += trade_pnl
        elif trade_pnl < 0.0:
            trade_loss_count += 1.0
            trade_loss_sum += trade_pnl

    # Final metrics
    total_return = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital > 0.0 else 0.0

    std_r = (m2_acc / (fbar - 1.0)) ** 0.5 if fbar > 1.0 else 0.0
    sharpe = (mean_acc / std_r * ann_factor) if std_r > 1e-12 else 0.0

    win_rate = (trade_win_count / num_trades * 100.0) if num_trades > 0.0 else 0.0

    # BUG-14: Remove 999.999 profit factor cap
    gross_loss = -trade_loss_sum if trade_loss_sum < 0.0 else 0.0
    if gross_loss > 0.0:
        profit_factor = trade_win_sum / gross_loss
    elif trade_win_sum > 0.0:
        profit_factor = 9999.0
    else:
        profit_factor = 0.0

    # BUG-06: Sortino ratio
    downside_std = (m2_neg / (fbar_neg - 1.0)) ** 0.5 if fbar_neg > 1.0 else 0.0
    sortino = (mean_acc / downside_std * ann_factor) if downside_std > 1e-12 else 0.0

    # BUG-06: Calmar ratio — annualized return / abs(max_drawdown)
    # Linear extrapolation avoids overflow for short windows and is accurate for long ones
    if fbar_total > 0.0 and initial_capital > 0.0:
        cagr = (final_equity / initial_capital - 1.0) * (ann_factor * ann_factor / fbar_total)
    else:
        cagr = 0.0
    calmar = (cagr * 100.0 / (-max_dd * 100.0)) if max_dd < -1e-10 else 0.0

    return (
        total_return,
        sharpe,
        max_dd * 100.0,
        win_rate,
        num_trades,
        profit_factor,
        sortino,
        calmar,
    )


@nb.njit(parallel=True, cache=True, error_model="numpy")
def backtest_batch(
    close,
    high,
    low,
    open_arr,
    signals_2d,
    fee,
    slippage,
    initial_capital,
    fraction,
    ann_factor,
    rf_per_bar,
):
    """Run ALL backtests in parallel across CPU cores.

    Args:
        close: 1D array of close prices (length M).
        high: 1D array of high prices (length M).
        low: 1D array of low prices (length M).
        open_arr: 1D array of open prices (length M).
        signals_2d: 2D array (N x M) of signals, one row per combo.
        fee: fee as decimal (e.g. 0.00075).
        slippage: slippage as decimal (e.g. 0.0005).
        initial_capital: starting cash (e.g. 10000.0).
        fraction: fraction of equity per trade, 0-1 (e.g. 1.0).
        ann_factor: sqrt(periods_per_year) for Sharpe annualization.
        rf_per_bar: risk-free rate per bar for Sharpe calculation.

    Returns:
        2D array (N x 8) of metrics per combo.
    """
    n_combos = signals_2d.shape[0]
    n = len(close)
    out = np.empty((n_combos, 8))

    for c in nb.prange(n_combos):
        r0, r1, r2, r3, r4, r5, r6, r7 = _backtest_one(
            close,
            high,
            low,
            open_arr,
            signals_2d[c],
            fee,
            slippage,
            n,
            initial_capital,
            fraction,
            ann_factor,
            rf_per_bar,
        )
        out[c, 0] = r0
        out[c, 1] = r1
        out[c, 2] = r2
        out[c, 3] = r3
        out[c, 4] = r4
        out[c, 5] = r5
        out[c, 6] = r6
        out[c, 7] = r7

    return out


@nb.njit(cache=True, error_model="numpy")
def _backtest_one_managed(
    close,
    high,
    low,
    open_arr,
    sig,
    fee,
    slippage,
    n,
    initial_capital,
    fraction,
    ann_factor,
    rf_per_bar,
    borrow_rate_per_bar,
    sl_pct,
    tp_pct,
    trail_pct,
    risk_per_trade,
):
    """Single backtest with portfolio tracking + position management + trade-level metrics."""
    cash = initial_capital
    shares = 0.0  # signed: +N long, -N short
    prev_sig = 0.0

    running_max = initial_capital
    max_dd = 0.0
    num_trades = 0.0
    mean_acc = 0.0
    m2_acc = 0.0
    fbar = 0.0
    fbar_total = 0.0

    # BUG-06: Downside variance accumulators for Sortino
    mean_neg = 0.0
    m2_neg = 0.0
    fbar_neg = 0.0

    # Trade-level P&L tracking
    trade_entry_equity = 0.0
    trade_win_count = 0.0
    trade_loss_count = 0.0
    trade_win_sum = 0.0
    trade_loss_sum = 0.0
    in_trade = False

    # Position management state
    entry_price = 0.0
    peak_price = 0.0  # best price since entry (for trailing stop)
    sl_level = 0.0
    tp_level = 0.0

    # Pre-compute fee/slippage factors
    slip_buy = 1.0 + slippage
    slip_sell = 1.0 - slippage
    cost_mul = 1.0 + fee
    recv_mul = 1.0 - fee

    equity = initial_capital

    for i in range(n):
        equity_before = equity
        if equity_before <= 0.0:
            break

        cur_sig = sig[i]
        cur_high = high[i]
        cur_low = low[i]
        stopped = False

        # --- Signal change: close old position and/or open new at open[i] ---
        if cur_sig != prev_sig:
            # Close existing position at open[i] if still open
            if shares != 0.0:
                if shares > 0.0:
                    fill = open_arr[i] * slip_sell
                    proceeds = shares * fill
                    cash += proceeds * recv_mul
                else:
                    fill = open_arr[i] * slip_buy
                    cost = (-shares) * fill
                    cash -= cost * cost_mul
                shares = 0.0
                entry_price = 0.0
                peak_price = 0.0

                # Record trade P&L on signal-driven close
                if in_trade:
                    trade_pnl = cash - trade_entry_equity
                    if trade_pnl > 0.0:
                        trade_win_count += 1.0
                        trade_win_sum += trade_pnl
                    elif trade_pnl < 0.0:
                        trade_loss_count += 1.0
                        trade_loss_sum += trade_pnl
                    in_trade = False

            # Open new position at open[i]
            equity_now = cash
            if cur_sig != 0.0 and equity_now > 0.0:
                trade_entry_equity = cash
                in_trade = True
                if risk_per_trade > 0.0 and sl_pct > 0.0:
                    trade_value = equity_now * risk_per_trade / sl_pct
                    trade_value = min(trade_value, equity_now * fraction)
                else:
                    trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    # BUG-01 fix: fee embedded in share price, spend exactly trade_value
                    fill = open_arr[i] * slip_buy
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    fill = open_arr[i] * slip_sell
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = fill
                peak_price = fill
                if cur_sig > 0.0:
                    sl_level = entry_price * (1.0 - sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 + tp_pct) if tp_pct > 0.0 else 0.0
                else:
                    sl_level = entry_price * (1.0 + sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 - tp_pct) if tp_pct > 0.0 else 0.0
                num_trades += 1.0

        # --- Check stops on open position using current bar's high/low ---
        if shares != 0.0:
            exit_price = 0.0

            # Margin call: forced liquidation if short equity hits 0 intrabar
            if shares < 0.0:
                bankrupt_price = cash / (-shares)
                if cur_high >= bankrupt_price:
                    exit_price = bankrupt_price
                    stopped = True

            if shares > 0.0:
                # Long position stops
                if sl_pct > 0.0 and cur_low <= sl_level:
                    exit_price = sl_level
                    stopped = True

                if not stopped and trail_pct > 0.0:
                    if cur_high > peak_price:
                        peak_price = cur_high
                    trail_level = peak_price * (1.0 - trail_pct)
                    if cur_low <= trail_level:
                        exit_price = trail_level
                        stopped = True

                if not stopped and tp_pct > 0.0 and cur_high >= tp_level:
                    exit_price = tp_level
                    stopped = True

            elif not stopped:
                # Short position stops
                if sl_pct > 0.0 and cur_high >= sl_level:
                    exit_price = sl_level
                    stopped = True

                if not stopped and trail_pct > 0.0:
                    if cur_low < peak_price:
                        peak_price = cur_low
                    trail_level = peak_price * (1.0 + trail_pct)
                    if cur_high >= trail_level:
                        exit_price = trail_level
                        stopped = True

                if not stopped and tp_pct > 0.0 and cur_low <= tp_level:
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

                # Record trade P&L on stop exit
                if in_trade:
                    trade_pnl = cash - trade_entry_equity
                    if trade_pnl > 0.0:
                        trade_win_count += 1.0
                        trade_win_sum += trade_pnl
                    elif trade_pnl < 0.0:
                        trade_loss_count += 1.0
                        trade_loss_sum += trade_pnl
                    in_trade = False

        # If stopped and signal wants re-entry, re-enter at close[i]
        if stopped and cur_sig != 0.0 and shares == 0.0:
            equity_now = cash
            if equity_now > 0.0:
                trade_entry_equity = cash
                in_trade = True
                if risk_per_trade > 0.0 and sl_pct > 0.0:
                    trade_value = equity_now * risk_per_trade / sl_pct
                    trade_value = min(trade_value, equity_now * fraction)
                else:
                    trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    # BUG-01 fix: fee embedded in share price, spend exactly trade_value
                    fill = close[i] * slip_buy
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    fill = close[i] * slip_sell
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = fill
                peak_price = fill
                if cur_sig > 0.0:
                    sl_level = entry_price * (1.0 - sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 + tp_pct) if tp_pct > 0.0 else 0.0
                else:
                    sl_level = entry_price * (1.0 + sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 - tp_pct) if tp_pct > 0.0 else 0.0
                num_trades += 1.0

        prev_sig = cur_sig

        # --- End-of-bar equity (mark at close[i]) ---
        equity = cash + shares * close[i]

        # BUG-10: Short borrow cost — deduct per bar proportional to short notional
        if shares < 0.0:
            cash -= (-shares) * close[i] * borrow_rate_per_bar

        # Bar return for Sharpe
        sr = (equity - equity_before) / equity_before

        # Drawdown
        if equity > running_max:
            running_max = equity
        dd = (equity - running_max) / running_max
        if dd < max_dd:
            max_dd = dd

        # Welford's online variance — active bars only (exclude flat-period zero-returns)
        fbar_total += 1.0
        is_active = (cur_sig != 0.0) or (shares != 0.0)
        if is_active:
            excess_r = sr - rf_per_bar
            fbar += 1.0
            delta = excess_r - mean_acc
            mean_acc += delta / fbar
            m2_acc += delta * (excess_r - mean_acc)

            if excess_r < 0.0:
                neg_r = -excess_r
                fbar_neg += 1.0
                delta_neg = neg_r - mean_neg
                mean_neg += delta_neg / fbar_neg
                m2_neg += delta_neg * (neg_r - mean_neg)

    # Close remaining position notionally for trade P&L
    final_equity = equity if fbar_total > 0.0 else initial_capital
    if in_trade and shares != 0.0:
        trade_pnl = final_equity - trade_entry_equity
        if trade_pnl > 0.0:
            trade_win_count += 1.0
            trade_win_sum += trade_pnl
        elif trade_pnl < 0.0:
            trade_loss_count += 1.0
            trade_loss_sum += trade_pnl

    # Final metrics
    total_return = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital > 0.0 else 0.0

    std_r = (m2_acc / (fbar - 1.0)) ** 0.5 if fbar > 1.0 else 0.0
    sharpe = (mean_acc / std_r * ann_factor) if std_r > 1e-12 else 0.0

    win_rate = (trade_win_count / num_trades * 100.0) if num_trades > 0.0 else 0.0

    # BUG-14: Remove 999.999 profit factor cap
    gross_loss = -trade_loss_sum if trade_loss_sum < 0.0 else 0.0
    if gross_loss > 0.0:
        profit_factor = trade_win_sum / gross_loss
    elif trade_win_sum > 0.0:
        profit_factor = 9999.0
    else:
        profit_factor = 0.0

    # BUG-06: Sortino ratio
    downside_std = (m2_neg / (fbar_neg - 1.0)) ** 0.5 if fbar_neg > 1.0 else 0.0
    sortino = (mean_acc / downside_std * ann_factor) if downside_std > 1e-12 else 0.0

    # BUG-06: Calmar ratio
    if fbar_total > 0.0 and initial_capital > 0.0 and final_equity > 0.0:
        cagr = (final_equity / initial_capital) ** (ann_factor * ann_factor / fbar_total) - 1.0
    else:
        cagr = 0.0
    calmar = (cagr * 100.0 / (-max_dd * 100.0)) if max_dd < -1e-10 else 0.0

    return (
        total_return,
        sharpe,
        max_dd * 100.0,
        win_rate,
        num_trades,
        profit_factor,
        sortino,
        calmar,
    )


@nb.njit(parallel=True, cache=True, error_model="numpy")
def backtest_batch_managed(
    close,
    high,
    low,
    open_arr,
    signals_2d,
    fee,
    slippage,
    initial_capital,
    fraction,
    ann_factor,
    rf_per_bar,
    borrow_rate_per_bar,
    sig_indices,
    sl_arr,
    tp_arr,
    trail_arr,
    risk_per_trade,
):
    """Run ALL managed backtests in parallel across CPU cores.

    Args:
        close: 1D array of close prices (length M).
        high: 1D array of high prices (length M).
        low: 1D array of low prices (length M).
        open_arr: 1D array of open prices (length M).
        signals_2d: 2D array (N_signals x M) of signals.
        fee: fee as decimal (e.g. 0.00075).
        slippage: slippage as decimal (e.g. 0.0005).
        initial_capital: starting cash (e.g. 10000.0).
        fraction: fraction of equity per trade, 0-1.
        ann_factor: sqrt(periods_per_year) for Sharpe annualization.
        rf_per_bar: risk-free rate per bar for Sharpe calculation.
        borrow_rate_per_bar: daily borrow rate for short positions (BUG-10).
        sig_indices: 1D int64 array (N_combos,) mapping each combo to a signal row.
        sl_arr: 1D float64 array (N_combos,) stop-loss pct per combo.
        tp_arr: 1D float64 array (N_combos,) take-profit pct per combo.
        trail_arr: 1D float64 array (N_combos,) trailing stop pct per combo.
        risk_per_trade: fraction of equity to risk per trade (0 = disabled).

    Returns:
        2D array (N_combos x 8) of metrics per combo.
    """
    n_combos = len(sig_indices)
    n = len(close)
    out = np.empty((n_combos, 8))

    for c in nb.prange(n_combos):
        si = sig_indices[c]
        r0, r1, r2, r3, r4, r5, r6, r7 = _backtest_one_managed(
            close,
            high,
            low,
            open_arr,
            signals_2d[si],
            fee,
            slippage,
            n,
            initial_capital,
            fraction,
            ann_factor,
            rf_per_bar,
            borrow_rate_per_bar,
            sl_arr[c],
            tp_arr[c],
            trail_arr[c],
            risk_per_trade,
        )
        out[c, 0] = r0
        out[c, 1] = r1
        out[c, 2] = r2
        out[c, 3] = r3
        out[c, 4] = r4
        out[c, 5] = r5
        out[c, 6] = r6
        out[c, 7] = r7

    return out


# ---------------------------------------------------------------------------
# ATR-managed engine — stop levels scale with ATR at entry bar
# ---------------------------------------------------------------------------


@nb.njit(cache=True, error_model="numpy")
def _backtest_one_atr_managed(
    close,
    high,
    low,
    open_arr,
    sig,
    fee,
    slippage,
    n,
    initial_capital,
    fraction,
    ann_factor,
    rf_per_bar,
    borrow_rate_per_bar,
    atr_arr,
    sl_mult,
    tp_mult,
    trail_mult,
    risk_per_trade,
):
    """Single backtest with ATR-scaled stops.

    stop_loss   = entry_price ± sl_mult   × ATR[entry_bar]   (0 disables)
    take_profit = entry_price ∓ tp_mult   × ATR[entry_bar]   (0 disables)
    trailing    = peak_price  ∓ trail_mult × ATR[current_bar] (0 disables)

    Stops are automatically disabled for entries on NaN ATR bars (warmup).
    """
    cash = initial_capital
    shares = 0.0
    prev_sig = 0.0

    running_max = initial_capital
    max_dd = 0.0
    num_trades = 0.0
    mean_acc = 0.0
    m2_acc = 0.0
    fbar = 0.0
    fbar_total = 0.0

    mean_neg = 0.0
    m2_neg = 0.0
    fbar_neg = 0.0

    trade_entry_equity = 0.0
    trade_win_count = 0.0
    trade_loss_count = 0.0
    trade_win_sum = 0.0
    trade_loss_sum = 0.0
    in_trade = False

    entry_price = 0.0
    peak_price = 0.0
    sl_level = 0.0
    tp_level = 0.0

    slip_buy = 1.0 + slippage
    slip_sell = 1.0 - slippage
    cost_mul = 1.0 + fee
    recv_mul = 1.0 - fee

    equity = initial_capital

    for i in range(n):
        equity_before = equity
        if equity_before <= 0.0:
            break

        cur_sig = sig[i]
        cur_high = high[i]
        cur_low = low[i]
        stopped = False

        if cur_sig != prev_sig:
            if shares != 0.0:
                if shares > 0.0:
                    fill = open_arr[i] * slip_sell
                    cash += shares * fill * recv_mul
                else:
                    fill = open_arr[i] * slip_buy
                    cash -= (-shares) * fill * cost_mul
                shares = 0.0
                entry_price = 0.0
                peak_price = 0.0
                if in_trade:
                    trade_pnl = cash - trade_entry_equity
                    if trade_pnl > 0.0:
                        trade_win_count += 1.0
                        trade_win_sum += trade_pnl
                    elif trade_pnl < 0.0:
                        trade_loss_count += 1.0
                        trade_loss_sum += trade_pnl
                    in_trade = False

            equity_now = cash
            if cur_sig != 0.0 and equity_now > 0.0:
                trade_entry_equity = cash
                in_trade = True
                # Compute fill first (needed for ATR-based risk sizing)
                if cur_sig > 0.0:
                    fill = open_arr[i] * slip_buy
                else:
                    fill = open_arr[i] * slip_sell
                atr_i = atr_arr[i]
                has_atr = (not np.isnan(atr_i)) and atr_i > 0.0
                if risk_per_trade > 0.0 and sl_mult > 0.0 and has_atr:
                    stop_dist = sl_mult * atr_i
                    trade_value = equity_now * risk_per_trade * fill / stop_dist
                    trade_value = min(trade_value, equity_now * fraction)
                else:
                    trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = fill
                peak_price = fill
                # ATR-scaled stop levels — disabled (0) when ATR is NaN/non-positive
                if cur_sig > 0.0:
                    sl_level = (
                        (entry_price - sl_mult * atr_i) if (sl_mult > 0.0 and has_atr) else 0.0
                    )
                    tp_level = (
                        (entry_price + tp_mult * atr_i) if (tp_mult > 0.0 and has_atr) else 0.0
                    )
                else:
                    sl_level = (
                        (entry_price + sl_mult * atr_i) if (sl_mult > 0.0 and has_atr) else 0.0
                    )
                    tp_level = (
                        (entry_price - tp_mult * atr_i) if (tp_mult > 0.0 and has_atr) else 0.0
                    )
                num_trades += 1.0

        if shares != 0.0:
            exit_price = 0.0

            if shares < 0.0:
                bankrupt_price = cash / (-shares)
                if cur_high >= bankrupt_price:
                    exit_price = bankrupt_price
                    stopped = True

            if shares > 0.0:
                if sl_level > 0.0 and cur_low <= sl_level:
                    exit_price = sl_level
                    stopped = True

                if not stopped and trail_mult > 0.0:
                    cur_atr = atr_arr[i]
                    if (not np.isnan(cur_atr)) and cur_atr > 0.0:
                        if cur_high > peak_price:
                            peak_price = cur_high
                        trail_level = peak_price - trail_mult * cur_atr
                        if cur_low <= trail_level:
                            exit_price = trail_level
                            stopped = True

                if not stopped and tp_level > 0.0 and cur_high >= tp_level:
                    exit_price = tp_level
                    stopped = True

            elif not stopped:
                if sl_level > 0.0 and cur_high >= sl_level:
                    exit_price = sl_level
                    stopped = True

                if not stopped and trail_mult > 0.0:
                    cur_atr = atr_arr[i]
                    if (not np.isnan(cur_atr)) and cur_atr > 0.0:
                        if cur_low < peak_price:
                            peak_price = cur_low
                        trail_level = peak_price + trail_mult * cur_atr
                        if cur_high >= trail_level:
                            exit_price = trail_level
                            stopped = True

                if not stopped and tp_level > 0.0 and cur_low <= tp_level:
                    exit_price = tp_level
                    stopped = True

            if stopped:
                if shares > 0.0:
                    fill = exit_price * slip_sell
                    cash += shares * fill * recv_mul
                else:
                    fill = exit_price * slip_buy
                    cash -= (-shares) * fill * cost_mul
                shares = 0.0
                entry_price = 0.0
                peak_price = 0.0
                if in_trade:
                    trade_pnl = cash - trade_entry_equity
                    if trade_pnl > 0.0:
                        trade_win_count += 1.0
                        trade_win_sum += trade_pnl
                    elif trade_pnl < 0.0:
                        trade_loss_count += 1.0
                        trade_loss_sum += trade_pnl
                    in_trade = False

        if stopped and cur_sig != 0.0 and shares == 0.0:
            equity_now = cash
            if equity_now > 0.0:
                trade_entry_equity = cash
                in_trade = True
                # Compute fill first (needed for ATR-based risk sizing)
                if cur_sig > 0.0:
                    fill = close[i] * slip_buy
                else:
                    fill = close[i] * slip_sell
                atr_i = atr_arr[i]
                has_atr = (not np.isnan(atr_i)) and atr_i > 0.0
                if risk_per_trade > 0.0 and sl_mult > 0.0 and has_atr:
                    stop_dist = sl_mult * atr_i
                    trade_value = equity_now * risk_per_trade * fill / stop_dist
                    trade_value = min(trade_value, equity_now * fraction)
                else:
                    trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = fill
                peak_price = fill
                if cur_sig > 0.0:
                    sl_level = (
                        (entry_price - sl_mult * atr_i) if (sl_mult > 0.0 and has_atr) else 0.0
                    )
                    tp_level = (
                        (entry_price + tp_mult * atr_i) if (tp_mult > 0.0 and has_atr) else 0.0
                    )
                else:
                    sl_level = (
                        (entry_price + sl_mult * atr_i) if (sl_mult > 0.0 and has_atr) else 0.0
                    )
                    tp_level = (
                        (entry_price - tp_mult * atr_i) if (tp_mult > 0.0 and has_atr) else 0.0
                    )
                num_trades += 1.0

        prev_sig = cur_sig

        equity = cash + shares * close[i]
        if shares < 0.0:
            cash -= (-shares) * close[i] * borrow_rate_per_bar

        sr = (equity - equity_before) / equity_before

        if equity > running_max:
            running_max = equity
        dd = (equity - running_max) / running_max
        if dd < max_dd:
            max_dd = dd

        fbar_total += 1.0
        is_active = (cur_sig != 0.0) or (shares != 0.0)
        if is_active:
            excess_r = sr - rf_per_bar
            fbar += 1.0
            delta = excess_r - mean_acc
            mean_acc += delta / fbar
            m2_acc += delta * (excess_r - mean_acc)
            if excess_r < 0.0:
                neg_r = -excess_r
                fbar_neg += 1.0
                delta_neg = neg_r - mean_neg
                mean_neg += delta_neg / fbar_neg
                m2_neg += delta_neg * (neg_r - mean_neg)

    final_equity = equity if fbar_total > 0.0 else initial_capital
    if in_trade and shares != 0.0:
        trade_pnl = final_equity - trade_entry_equity
        if trade_pnl > 0.0:
            trade_win_count += 1.0
            trade_win_sum += trade_pnl
        elif trade_pnl < 0.0:
            trade_loss_count += 1.0
            trade_loss_sum += trade_pnl

    total_return = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital > 0.0 else 0.0
    std_r = (m2_acc / (fbar - 1.0)) ** 0.5 if fbar > 1.0 else 0.0
    sharpe = (mean_acc / std_r * ann_factor) if std_r > 1e-12 else 0.0
    win_rate = (trade_win_count / num_trades * 100.0) if num_trades > 0.0 else 0.0

    gross_loss = -trade_loss_sum if trade_loss_sum < 0.0 else 0.0
    if gross_loss > 0.0:
        profit_factor = trade_win_sum / gross_loss
    elif trade_win_sum > 0.0:
        profit_factor = 9999.0
    else:
        profit_factor = 0.0

    downside_std = (m2_neg / (fbar_neg - 1.0)) ** 0.5 if fbar_neg > 1.0 else 0.0
    sortino = (mean_acc / downside_std * ann_factor) if downside_std > 1e-12 else 0.0

    if fbar_total > 0.0 and initial_capital > 0.0 and final_equity > 0.0:
        cagr = (final_equity / initial_capital) ** (ann_factor * ann_factor / fbar_total) - 1.0
    else:
        cagr = 0.0
    calmar = (cagr * 100.0 / (-max_dd * 100.0)) if max_dd < -1e-10 else 0.0

    return (
        total_return,
        sharpe,
        max_dd * 100.0,
        win_rate,
        num_trades,
        profit_factor,
        sortino,
        calmar,
    )


@nb.njit(parallel=True, cache=True, error_model="numpy")
def backtest_batch_atr_managed(
    close,
    high,
    low,
    open_arr,
    signals_2d,
    atr_arr,
    fee,
    slippage,
    initial_capital,
    fraction,
    ann_factor,
    rf_per_bar,
    borrow_rate_per_bar,
    sig_indices,
    sl_mult_arr,
    tp_mult_arr,
    trail_mult_arr,
    risk_per_trade,
):
    """Run ALL ATR-managed backtests in parallel across CPU cores.

    Args:
        atr_arr: 1D float64 array of ATR values (NaN for warmup bars).
        sig_indices: 1D int64 array mapping each combo to a signals_2d row.
        sl_mult_arr: ATR stop-loss multipliers per combo (0 = disabled).
        tp_mult_arr: ATR take-profit multipliers per combo (0 = disabled).
        trail_mult_arr: ATR trailing-stop multipliers per combo (0 = disabled).
        risk_per_trade: fraction of equity to risk per trade (0 = disabled).

    Returns:
        2D array (N_combos x 8) of metrics per combo.
    """
    n_combos = len(sig_indices)
    n = len(close)
    out = np.empty((n_combos, 8))

    for c in nb.prange(n_combos):
        si = sig_indices[c]
        r0, r1, r2, r3, r4, r5, r6, r7 = _backtest_one_atr_managed(
            close,
            high,
            low,
            open_arr,
            signals_2d[si],
            fee,
            slippage,
            n,
            initial_capital,
            fraction,
            ann_factor,
            rf_per_bar,
            borrow_rate_per_bar,
            atr_arr,
            sl_mult_arr[c],
            tp_mult_arr[c],
            trail_mult_arr[c],
            risk_per_trade,
        )
        out[c, 0] = r0
        out[c, 1] = r1
        out[c, 2] = r2
        out[c, 3] = r3
        out[c, 4] = r4
        out[c, 5] = r5
        out[c, 6] = r6
        out[c, 7] = r7

    return out


# ---------------------------------------------------------------------------
# Equity curve variants — NOT parallelized, only for top N results
# ---------------------------------------------------------------------------


@nb.njit(cache=True, error_model="numpy")
def backtest_one_equity(
    close,
    high,
    low,
    open_arr,
    sig,
    fee,
    slippage,
    initial_capital,
    fraction,
    ann_factor,
    rf_per_bar,
):
    """Same as _backtest_one but also returns an equity curve array."""
    n = len(close)
    curve = np.empty(n, dtype=np.float64)

    cash = initial_capital
    shares = 0.0
    prev_sig = 0.0
    equity = initial_capital

    slip_buy = 1.0 + slippage
    slip_sell = 1.0 - slippage
    cost_mul = 1.0 + fee
    recv_mul = 1.0 - fee

    for i in range(n):
        if equity <= 0.0:
            for j in range(i, n):
                curve[j] = 0.0
            break

        cur_sig = sig[i]
        if cur_sig != prev_sig:
            if shares > 0.0:
                fill = open_arr[i] * slip_sell
                cash += shares * fill * recv_mul
                shares = 0.0
            elif shares < 0.0:
                fill = open_arr[i] * slip_buy
                cash -= (-shares) * fill * cost_mul
                shares = 0.0

            if cur_sig != 0.0 and cash > 0.0:
                trade_value = cash * fraction
                if cur_sig > 0.0:
                    # BUG-01 fix: fee embedded in share price
                    fill = open_arr[i] * slip_buy
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    fill = open_arr[i] * slip_sell
                    shares = -(trade_value / fill)
                    cash += trade_value * recv_mul
            prev_sig = cur_sig

        # Margin call: forced liquidation if short equity hits 0 intrabar
        if shares < 0.0:
            if cash + shares * high[i] <= 0.0:
                bankrupt_price = cash / (-shares)
                fill = bankrupt_price * slip_buy
                cash -= (-shares) * fill * cost_mul
                cash = max(cash, 0.0)
                shares = 0.0

        equity = cash + shares * close[i]
        curve[i] = equity

    return curve


@nb.njit(cache=True, error_model="numpy")
def backtest_one_managed_equity(
    close,
    high,
    low,
    open_arr,
    sig,
    fee,
    slippage,
    initial_capital,
    fraction,
    ann_factor,
    rf_per_bar,
    borrow_rate_per_bar,
    sl_pct,
    tp_pct,
    trail_pct,
    risk_per_trade,
):
    """Same as _backtest_one_managed but also returns an equity curve array."""
    n = len(close)
    curve = np.empty(n, dtype=np.float64)

    cash = initial_capital
    shares = 0.0
    prev_sig = 0.0
    equity = initial_capital

    entry_price = 0.0
    peak_price = 0.0
    sl_level = 0.0
    tp_level = 0.0

    slip_buy = 1.0 + slippage
    slip_sell = 1.0 - slippage
    cost_mul = 1.0 + fee
    recv_mul = 1.0 - fee

    for i in range(n):
        if equity <= 0.0:
            for j in range(i, n):
                curve[j] = 0.0
            break

        cur_sig = sig[i]
        cur_high = high[i]
        cur_low = low[i]
        stopped = False

        # Signal change — execute at open[i]
        if cur_sig != prev_sig:
            if shares != 0.0:
                if shares > 0.0:
                    fill = open_arr[i] * slip_sell
                    cash += shares * fill * recv_mul
                else:
                    fill = open_arr[i] * slip_buy
                    cash -= (-shares) * fill * cost_mul
                shares = 0.0
                entry_price = 0.0
                peak_price = 0.0

            equity_now = cash
            if cur_sig != 0.0 and equity_now > 0.0:
                if risk_per_trade > 0.0 and sl_pct > 0.0:
                    trade_value = equity_now * risk_per_trade / sl_pct
                    trade_value = min(trade_value, equity_now * fraction)
                else:
                    trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    # BUG-01 fix: fee embedded in share price
                    fill = open_arr[i] * slip_buy
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    fill = open_arr[i] * slip_sell
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = fill
                peak_price = fill
                if cur_sig > 0.0:
                    sl_level = entry_price * (1.0 - sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 + tp_pct) if tp_pct > 0.0 else 0.0
                else:
                    sl_level = entry_price * (1.0 + sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 - tp_pct) if tp_pct > 0.0 else 0.0

        # Check stops — only if position is still open after signal processing
        if shares != 0.0:
            exit_price = 0.0

            # Margin call: forced liquidation if short equity hits 0 intrabar
            if shares < 0.0:
                bankrupt_price = cash / (-shares)
                if cur_high >= bankrupt_price:
                    exit_price = bankrupt_price
                    stopped = True

            if shares > 0.0:
                if sl_pct > 0.0 and cur_low <= sl_level:
                    exit_price = sl_level
                    stopped = True
                if not stopped and trail_pct > 0.0:
                    if cur_high > peak_price:
                        peak_price = cur_high
                    trail_level = peak_price * (1.0 - trail_pct)
                    if cur_low <= trail_level:
                        exit_price = trail_level
                        stopped = True
                if not stopped and tp_pct > 0.0 and cur_high >= tp_level:
                    exit_price = tp_level
                    stopped = True
            elif not stopped:
                if sl_pct > 0.0 and cur_high >= sl_level:
                    exit_price = sl_level
                    stopped = True
                if not stopped and trail_pct > 0.0:
                    if cur_low < peak_price:
                        peak_price = cur_low
                    trail_level = peak_price * (1.0 + trail_pct)
                    if cur_high >= trail_level:
                        exit_price = trail_level
                        stopped = True
                if not stopped and tp_pct > 0.0 and cur_low <= tp_level:
                    exit_price = tp_level
                    stopped = True

            if stopped:
                if shares > 0.0:
                    fill = exit_price * slip_sell
                    cash += shares * fill * recv_mul
                else:
                    fill = exit_price * slip_buy
                    cash -= (-shares) * fill * cost_mul
                shares = 0.0
                entry_price = 0.0
                peak_price = 0.0

        # Stopped re-entry at close[i]
        if stopped and cur_sig != 0.0 and shares == 0.0:
            equity_now = cash
            if equity_now > 0.0:
                if risk_per_trade > 0.0 and sl_pct > 0.0:
                    trade_value = equity_now * risk_per_trade / sl_pct
                    trade_value = min(trade_value, equity_now * fraction)
                else:
                    trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    # BUG-01 fix: fee embedded in share price
                    fill = close[i] * slip_buy
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    fill = close[i] * slip_sell
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = fill
                peak_price = fill
                if cur_sig > 0.0:
                    sl_level = entry_price * (1.0 - sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 + tp_pct) if tp_pct > 0.0 else 0.0
                else:
                    sl_level = entry_price * (1.0 + sl_pct) if sl_pct > 0.0 else 0.0
                    tp_level = entry_price * (1.0 - tp_pct) if tp_pct > 0.0 else 0.0

        prev_sig = cur_sig
        equity = cash + shares * close[i]

        # BUG-10: Short borrow cost deduction
        if shares < 0.0:
            cash -= (-shares) * close[i] * borrow_rate_per_bar

        curve[i] = equity

    return curve


@nb.njit(cache=True, error_model="numpy")
def backtest_one_atr_managed_equity(
    close,
    high,
    low,
    open_arr,
    sig,
    fee,
    slippage,
    initial_capital,
    fraction,
    ann_factor,
    rf_per_bar,
    borrow_rate_per_bar,
    atr_arr,
    sl_mult,
    tp_mult,
    trail_mult,
    risk_per_trade,
):
    """Same as _backtest_one_atr_managed but returns an equity curve array."""
    n = len(close)
    curve = np.empty(n, dtype=np.float64)

    cash = initial_capital
    shares = 0.0
    prev_sig = 0.0
    equity = initial_capital

    entry_price = 0.0
    peak_price = 0.0
    sl_level = 0.0
    tp_level = 0.0

    slip_buy = 1.0 + slippage
    slip_sell = 1.0 - slippage
    cost_mul = 1.0 + fee
    recv_mul = 1.0 - fee

    for i in range(n):
        if equity <= 0.0:
            for j in range(i, n):
                curve[j] = 0.0
            break

        cur_sig = sig[i]
        cur_high = high[i]
        cur_low = low[i]
        stopped = False

        if cur_sig != prev_sig:
            if shares != 0.0:
                if shares > 0.0:
                    fill = open_arr[i] * slip_sell
                    cash += shares * fill * recv_mul
                else:
                    fill = open_arr[i] * slip_buy
                    cash -= (-shares) * fill * cost_mul
                shares = 0.0
                entry_price = 0.0
                peak_price = 0.0

            equity_now = cash
            if cur_sig != 0.0 and equity_now > 0.0:
                # Compute fill first (needed for ATR-based risk sizing)
                if cur_sig > 0.0:
                    fill = open_arr[i] * slip_buy
                else:
                    fill = open_arr[i] * slip_sell
                atr_i = atr_arr[i]
                has_atr = (not np.isnan(atr_i)) and atr_i > 0.0
                if risk_per_trade > 0.0 and sl_mult > 0.0 and has_atr:
                    stop_dist = sl_mult * atr_i
                    trade_value = equity_now * risk_per_trade * fill / stop_dist
                    trade_value = min(trade_value, equity_now * fraction)
                else:
                    trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = fill
                peak_price = fill
                if cur_sig > 0.0:
                    sl_level = (
                        (entry_price - sl_mult * atr_i) if (sl_mult > 0.0 and has_atr) else 0.0
                    )
                    tp_level = (
                        (entry_price + tp_mult * atr_i) if (tp_mult > 0.0 and has_atr) else 0.0
                    )
                else:
                    sl_level = (
                        (entry_price + sl_mult * atr_i) if (sl_mult > 0.0 and has_atr) else 0.0
                    )
                    tp_level = (
                        (entry_price - tp_mult * atr_i) if (tp_mult > 0.0 and has_atr) else 0.0
                    )

        if shares != 0.0:
            exit_price = 0.0

            if shares < 0.0:
                bankrupt_price = cash / (-shares)
                if cur_high >= bankrupt_price:
                    exit_price = bankrupt_price
                    stopped = True

            if shares > 0.0:
                if sl_level > 0.0 and cur_low <= sl_level:
                    exit_price = sl_level
                    stopped = True
                if not stopped and trail_mult > 0.0:
                    cur_atr = atr_arr[i]
                    if (not np.isnan(cur_atr)) and cur_atr > 0.0:
                        if cur_high > peak_price:
                            peak_price = cur_high
                        trail_level = peak_price - trail_mult * cur_atr
                        if cur_low <= trail_level:
                            exit_price = trail_level
                            stopped = True
                if not stopped and tp_level > 0.0 and cur_high >= tp_level:
                    exit_price = tp_level
                    stopped = True

            elif not stopped:
                if sl_level > 0.0 and cur_high >= sl_level:
                    exit_price = sl_level
                    stopped = True
                if not stopped and trail_mult > 0.0:
                    cur_atr = atr_arr[i]
                    if (not np.isnan(cur_atr)) and cur_atr > 0.0:
                        if cur_low < peak_price:
                            peak_price = cur_low
                        trail_level = peak_price + trail_mult * cur_atr
                        if cur_high >= trail_level:
                            exit_price = trail_level
                            stopped = True
                if not stopped and tp_level > 0.0 and cur_low <= tp_level:
                    exit_price = tp_level
                    stopped = True

            if stopped:
                if shares > 0.0:
                    fill = exit_price * slip_sell
                    cash += shares * fill * recv_mul
                else:
                    fill = exit_price * slip_buy
                    cash -= (-shares) * fill * cost_mul
                shares = 0.0
                entry_price = 0.0
                peak_price = 0.0

        if stopped and cur_sig != 0.0 and shares == 0.0:
            equity_now = cash
            if equity_now > 0.0:
                # Compute fill first (needed for ATR-based risk sizing)
                if cur_sig > 0.0:
                    fill = close[i] * slip_buy
                else:
                    fill = close[i] * slip_sell
                atr_i = atr_arr[i]
                has_atr = (not np.isnan(atr_i)) and atr_i > 0.0
                if risk_per_trade > 0.0 and sl_mult > 0.0 and has_atr:
                    stop_dist = sl_mult * atr_i
                    trade_value = equity_now * risk_per_trade * fill / stop_dist
                    trade_value = min(trade_value, equity_now * fraction)
                else:
                    trade_value = equity_now * fraction
                if cur_sig > 0.0:
                    shares = trade_value / (fill * cost_mul)
                    cash -= trade_value
                else:
                    qty = trade_value / fill
                    cash += trade_value * recv_mul
                    shares = -qty
                entry_price = fill
                peak_price = fill
                if cur_sig > 0.0:
                    sl_level = (
                        (entry_price - sl_mult * atr_i) if (sl_mult > 0.0 and has_atr) else 0.0
                    )
                    tp_level = (
                        (entry_price + tp_mult * atr_i) if (tp_mult > 0.0 and has_atr) else 0.0
                    )
                else:
                    sl_level = (
                        (entry_price + sl_mult * atr_i) if (sl_mult > 0.0 and has_atr) else 0.0
                    )
                    tp_level = (
                        (entry_price - tp_mult * atr_i) if (tp_mult > 0.0 and has_atr) else 0.0
                    )

        prev_sig = cur_sig
        equity = cash + shares * close[i]
        if shares < 0.0:
            cash -= (-shares) * close[i] * borrow_rate_per_bar

        curve[i] = equity

    return curve


# Warmup JIT at import time
_dummy_close = np.array([1.0, 1.01, 1.02], dtype=np.float64)
_dummy_high = np.array([1.01, 1.02, 1.03], dtype=np.float64)
_dummy_low = np.array([0.99, 1.00, 1.01], dtype=np.float64)
_dummy_open = np.array([1.00, 1.005, 1.015], dtype=np.float64)
_dummy_sigs = np.zeros((1, 3), dtype=np.float64)
_dummy_ann = np.float64(8760.0) ** 0.5
backtest_batch(
    _dummy_close,
    _dummy_high,
    _dummy_low,
    _dummy_open,
    _dummy_sigs,
    0.001,
    0.0005,
    10000.0,
    1.0,
    _dummy_ann,
    0.0,
)
backtest_batch_managed(
    _dummy_close,
    _dummy_high,
    _dummy_low,
    _dummy_open,
    _dummy_sigs,
    0.001,
    0.0005,
    10000.0,
    1.0,
    _dummy_ann,
    0.0,
    0.0,
    np.zeros(1, dtype=np.int64),
    np.zeros(1, dtype=np.float64),
    np.zeros(1, dtype=np.float64),
    np.zeros(1, dtype=np.float64),
    0.0,
)
_dummy_sig1d = np.zeros(3, dtype=np.float64)
backtest_one_equity(
    _dummy_close,
    _dummy_high,
    _dummy_low,
    _dummy_open,
    _dummy_sig1d,
    0.001,
    0.0005,
    10000.0,
    1.0,
    _dummy_ann,
    0.0,
)
backtest_one_managed_equity(
    _dummy_close,
    _dummy_high,
    _dummy_low,
    _dummy_open,
    _dummy_sig1d,
    0.001,
    0.0005,
    10000.0,
    1.0,
    _dummy_ann,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
_dummy_atr3 = np.array([0.001, 0.001, 0.001], dtype=np.float64)
backtest_batch_atr_managed(
    _dummy_close,
    _dummy_high,
    _dummy_low,
    _dummy_open,
    _dummy_sigs,
    _dummy_atr3,
    0.001,
    0.0005,
    10000.0,
    1.0,
    _dummy_ann,
    0.0,
    0.0,
    np.zeros(1, dtype=np.int64),
    np.zeros(1, dtype=np.float64),
    np.zeros(1, dtype=np.float64),
    np.zeros(1, dtype=np.float64),
    0.0,
)
backtest_one_atr_managed_equity(
    _dummy_close,
    _dummy_high,
    _dummy_low,
    _dummy_open,
    _dummy_sig1d,
    0.001,
    0.0005,
    10000.0,
    1.0,
    _dummy_ann,
    0.0,
    0.0,
    _dummy_atr3,
    1.5,
    3.0,
    0.0,
    0.0,
)
del (
    _dummy_close,
    _dummy_high,
    _dummy_low,
    _dummy_open,
    _dummy_sigs,
    _dummy_sig1d,
    _dummy_ann,
    _dummy_atr3,
)


def rank_results(results: list[Result], sort_by: str = "sharpe_ratio") -> list[Result]:
    """Sort results by chosen metric, descending. Zero-trade results rank last."""
    return sorted(results, key=lambda r: (r.num_trades > 0, getattr(r, sort_by)), reverse=True)
