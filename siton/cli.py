"""CLI entry point for Siton."""

import argparse
import importlib.util
import sys
import time
import warnings

import numpy as np

from siton.engine import backtest_batch, backtest_batch_managed, Result, rank_results
from siton.indicators import expand_grid, clear_cache
from siton.data import fetch_ohlcv, load_csv, generate_sample


def _load_data(args):
    """Load OHLCV data based on CLI args. Returns a Polars DataFrame."""
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if args.demo:
        print(f"\n[*] Generating synthetic data (10000 hourly candles)...")
        df = generate_sample(n=10000)
        print(f"    Simulated BTC-like price from ${df['close'][0]:.0f} to ${df['close'][-1]:.0f}")
    elif args.csv:
        print(f"\n[*] Loading data from {args.csv}...")
        df = load_csv(args.csv, start=start, end=end)
    else:
        date_desc = ""
        if start and end:
            date_desc = f" from {start} to {end}"
        elif start:
            date_desc = f" from {start}"
        elif end:
            date_desc = f" until {end}"
        limit_desc = f"{args.limit} " if not start else ""
        print(f"\n[*] Fetching {limit_desc}{args.timeframe} candles for {args.symbol} from {args.exchange}{date_desc}...")
        df = fetch_ohlcv(args.symbol, args.timeframe, args.exchange, args.limit,
                         start=start, end=end)
    return df


def _df_to_numpy(df):
    """Extract OHLCV arrays from a Polars DataFrame."""
    return {
        "open":   df["open"].to_numpy().astype(np.float64),
        "high":   df["high"].to_numpy().astype(np.float64),
        "low":    df["low"].to_numpy().astype(np.float64),
        "close":  df["close"].to_numpy().astype(np.float64),
        "volume": df["volume"].to_numpy().astype(np.float64),
    }


def _display_results(ranked, top, sort):
    """Print the ranked results table and best strategy summary."""
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


def _load_sdk_file(filepath):
    """Dynamically import an SDK strategy file and return its STRATEGY variable."""
    spec = importlib.util.spec_from_file_location("_sdk_strategy", filepath)
    mod = importlib.util.module_from_spec(spec)
    # Prevent the module's __main__ block from running during import
    mod.__name__ = "_sdk_strategy"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spec.loader.exec_module(mod)
    if not hasattr(mod, "STRATEGY"):
        print(f"Error: {filepath} does not define a STRATEGY variable.", file=sys.stderr)
        sys.exit(1)
    return mod.STRATEGY


def _run_sdk_strategy(strategy, data, close, t1):
    """Run an SDK Strategy through the standard pipeline."""
    import itertools
    from siton.sdk import Strategy as SDKStrategy

    strategies = [strategy] if isinstance(strategy, SDKStrategy) else list(strategy)

    combos = []
    combo_strat = []
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

    clear_cache()
    n_combos = len(combos)
    signals_2d = np.empty((n_combos, len(close)), dtype=np.float64)
    for i, (name, params) in enumerate(combos):
        signals_2d[i] = strat_lookup[name](data, **params)
    clear_cache()

    # Read execution params from the first Strategy object
    first = strategies[0]
    fee_pct = first.fee if first.fee is not None else 0.075
    slip_pct = first.slippage
    fee = fee_pct / 100.0
    slippage = slip_pct / 100.0
    capital = first.capital
    fraction = first.fraction

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

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Siton - Fast Crypto Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  siton my_strategy.py --demo
  siton my_strategy.py -s BTC/USDT -t 1h --start 2024-01-01 --end 2024-06-30
  siton my_strategy.py --csv data.csv

Execution parameters (fee, slippage, capital, fraction, top, sort) are
defined in the Strategy file.
        """,
    )
    parser.add_argument("strategy", help="Strategy file to run (must define STRATEGY variable)")
    parser.add_argument("-s", "--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("-t", "--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("-e", "--exchange", default="binance", help="Exchange (default: binance)")
    parser.add_argument("-n", "--limit", type=int, default=5000, help="Number of candles (default: 5000)")
    parser.add_argument("--start", metavar="DATE", help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", metavar="DATE", help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--csv", help="Load OHLCV from CSV file instead of exchange")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no API needed)")

    args = parser.parse_args()

    # --- Load strategy ---
    strategy = _load_sdk_file(args.strategy)

    # --- Load data ---
    print("=" * 70)
    print("  SITON - Fast Crypto Strategy Backtester")
    print("=" * 70)

    t0 = time.perf_counter()
    df = _load_data(args)
    print(f"    {len(df)} candles loaded in {time.perf_counter() - t0:.2f}s")

    data = _df_to_numpy(df)
    close = data["close"]

    t1 = time.perf_counter()
    results = _run_sdk_strategy(strategy, data, close, t1)

    total_combos = len(results)
    elapsed = time.perf_counter() - t1
    print(f"    Done in {elapsed:.4f}s ({total_combos / max(elapsed, 0.0001):.0f} backtests/sec)")

    # --- Rank & display ---
    from siton.sdk import Strategy as SDKStrategy
    strategies = [strategy] if isinstance(strategy, SDKStrategy) else list(strategy)
    first = strategies[0]
    sort = first.sort
    top = first.top

    ranked = rank_results(results, sort_by=sort)
    _display_results(ranked, top, sort)

    total_time = time.perf_counter() - t0
    print(f"\nTotal time: {total_time:.2f}s")
