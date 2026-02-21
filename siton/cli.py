"""CLI entry point for Siton."""

import argparse
import importlib.util
import sys
import time
import warnings

import numpy as np

from siton.data import fetch_ohlcv, generate_sample, load_csv
from siton.engine import rank_results
from siton.sdk import run as sdk_run


def _load_data(args):
    """Load OHLCV data based on CLI args. Returns a Polars DataFrame."""
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if args.demo:
        print("\n[*] Generating synthetic data (10000 hourly candles)...")
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


def _display_results(ranked, top, sort, buy_hold_pct=None):
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

    if buy_hold_pct is not None:
        alpha = best.total_return_pct - buy_hold_pct
        print(f"\n  Buy & Hold: {buy_hold_pct:+.2f}%")
        print(f"  Alpha (vs B&H): {alpha:+.2f}%")

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
    parser.add_argument("--validate", action="store_true", help="Enable walk-forward validation")
    parser.add_argument("--train-ratio", type=float, default=None, help="Train/test split ratio (default: 0.7)")

    args = parser.parse_args()

    # --- Load strategy ---
    strategy = _load_sdk_file(args.strategy)

    from siton.sdk import Strategy as SDKStrategy
    strategies = [strategy] if isinstance(strategy, SDKStrategy) else list(strategy)

    # --- Load data ---
    print("=" * 70)
    print("  SITON - Fast Crypto Strategy Backtester")
    print("=" * 70)

    t0 = time.perf_counter()
    df = _load_data(args)
    print(f"    {len(df)} candles loaded in {time.perf_counter() - t0:.2f}s")

    data = _df_to_numpy(df)
    close = data["close"]

    # Buy & hold benchmark
    buy_hold_pct = (close[-1] / close[0] - 1.0) * 100.0

    # Apply CLI overrides to strategy
    first = strategies[0]
    if args.validate:
        first.validate = True
    if args.train_ratio is not None:
        first.train_ratio = args.train_ratio

    t1 = time.perf_counter()

    print("\n[*] Running backtest...")
    results = sdk_run(strategies, data, timeframe=args.timeframe)

    total_elapsed = time.perf_counter() - t1
    sort = first.sort
    top = first.top

    if isinstance(results, dict):
        # Walk-forward validation: train + test results
        train_top = results["train"]
        test_top = results["test"]
        stability = results.get("stability", [])
        split = results.get("split", 0)
        n_bars = len(close)

        print(f"    Done in {total_elapsed:.4f}s")

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
        print("  OUT-OF-SAMPLE (TEST) — IS RANK ORDER (no OOS re-ranking)")
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
            print("\n  PARAMETER STABILITY (top-1 as IS window expands)")
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
        n_results = len(results)
        print(f"    Done in {total_elapsed:.4f}s ({n_results / max(total_elapsed, 0.0001):.0f} backtests/sec)")

        ranked = rank_results(results, sort_by=sort)
        _display_results(ranked, top, sort, buy_hold_pct=buy_hold_pct)

    total_time = time.perf_counter() - t0
    print(f"\nTotal time: {total_time:.2f}s")
