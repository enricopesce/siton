"""Example 5 — Multi-Strategy Purged K-Fold CV (Expert)
=======================================================
Compare three different strategy families head-to-head using purged
K-fold cross-validation, which gives K independent OOS estimates per
combo and a cv_consistency metric to detect overfitting.

Concepts introduced:
  - Multiple Strategy objects passed to backtest() / run()
  - Signal.majority(*signals) — majority vote across N signals
  - cv=True + cv_folds — purged K-fold cross-validation
  - cv_consistency — fraction of folds with positive OOS Sharpe
  - Deflated Sharpe Ratio (PSR column) to penalise multiple testing
  - bollinger_bands(), obv(), channel_breakout() signals
  - Programmatic run() API — inspect Result objects directly

Run:
    python examples/05_multi_strategy_walkforward.py --demo
    siton examples/05_multi_strategy_walkforward.py --demo --cv
"""

from siton.sdk import (
    ema_cross, macd_signal, adx, rsi,
    bollinger_bands, obv, channel_breakout,
    Signal, Strategy, backtest, run,
)
from siton.data import generate_sample

# ── Strategy A: Trend following ───────────────────────────────────────────────
trend_entry = ema_cross(fast=[8, 12], slow=[26, 50]).filter_by(
    adx(periods=[14], thresholds=[20, 25])
)
strat_a = Strategy(
    "TrendFollowing",
    signal=trend_entry,
    long_only=True,
    stop_loss=[3.0, 5.0],
    top=5,
    sort="sharpe_ratio",
    cv=True,
    cv_folds=5,
    purge_bars=50,
)

# ── Strategy B: Mean reversion ────────────────────────────────────────────────
mean_rev = bollinger_bands(windows=[20], num_std=[2.0, 2.5]).filter_by(
    ~rsi(periods=[14], oversold=[30], overbought=[70])
)
strat_b = Strategy(
    "MeanReversion",
    signal=mean_rev,
    long_only=True,
    stop_loss=[2.0, 3.0],
    top=5,
    sort="sharpe_ratio",
    cv=True,
    cv_folds=5,
    purge_bars=50,
)

# ── Strategy C: Majority vote ensemble ───────────────────────────────────────
# Three independent signals; at least two must agree for a trade.
ensemble = Signal.majority(
    ema_cross(fast=[12], slow=[26]),
    macd_signal(fast=[12], slow=[26], signal_period=[9]),
    obv(ma_periods=[20]),
)
strat_c = Strategy(
    "MajorityVote",
    signal=ensemble,
    long_only=True,
    top=5,
    sort="sharpe_ratio",
    cv=True,
    cv_folds=5,
    purge_bars=50,
)

# ── STRATEGY list (used by siton CLI) ─────────────────────────────────────────
STRATEGY = strat_a   # default when invoked via `siton` CLI (single strategy)

# ── Programmatic comparison (run directly) ────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo", action="store_true")
    args, _ = parser.parse_known_args()

    if args.demo:
        print("Generating synthetic data (10 000 hourly candles)…")
        df = generate_sample(n=10_000)

        for strat in [strat_a, strat_b, strat_c]:
            print(f"\n{'=' * 60}")
            print(f"  {strat.name}")
            print(f"{'=' * 60}")
            results = run(strat, df, timeframe="1h")
            if isinstance(results, dict):
                aggregated = results["aggregated"]
                n_folds = len(results["folds"])
                print(f"  {'Metric':<18} {'Aggregated OOS':>14}")
                print(f"  {'-'*34}")
                if aggregated:
                    best = aggregated[0]
                    cons = f"{int(best.cv_consistency * n_folds)}/{n_folds}"
                    print(f"  {'Total Return %':<18} {best.total_return_pct:>+13.2f}%")
                    print(f"  {'Sharpe Ratio':<18} {best.sharpe_ratio:>14.3f}")
                    print(f"  {'Max Drawdown %':<18} {best.max_drawdown_pct:>14.2f}")
                    print(f"  {'Win Rate %':<18} {best.win_rate_pct:>13.2f}%")
                    print(f"  {'Trades (total)':<18} {best.num_trades:>14}")
                    print(f"  {'PSR':<18} {best.psr:>14.3f}")
                    print(f"  {'Consistency':<18} {cons:>14}")
                    print(f"  Best params: {best.params}")
            else:
                for r in results[:3]:
                    print(f"  {r.strategy}: return={r.total_return_pct:+.2f}% sharpe={r.sharpe_ratio:.3f} psr={r.psr:.3f}")
    else:
        # Fallback: use CLI runner for the first strategy
        backtest(strat_a)
