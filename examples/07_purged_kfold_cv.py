"""Example 7 — Purged K-Fold Cross-Validation (López de Prado, AFML)
=====================================================================
Purged K-fold CV divides the data into K folds and uses each fold as
the out-of-sample (OOS) test set in turn, with an anchored expanding
training window.  A "purge gap" of N bars is removed from the end of
each training window to prevent indicator warm-up contamination at
the train/test boundary.

The result is K OOS Sharpe estimates per parameter combo, aggregated
to a single robust ranking.  The ``cv_consistency`` metric reports the
fraction of folds with positive OOS Sharpe — a quick sanity check for
overfitting (e.g. 4/5 means the combo was profitable in 4 of 5 folds).

Concepts introduced:
  - Strategy(..., cv=True, cv_folds=5, purge_bars=50)
  - run() returning {"folds": [...], "aggregated": [...], ...}
  - Result.cv_consistency field
  - --cv / --cv-folds / --purge-bars CLI flags

Run:
    python examples/07_purged_kfold_cv.py --demo
    siton examples/07_purged_kfold_cv.py --demo
    siton examples/07_purged_kfold_cv.py --demo --cv --cv-folds 5 --purge-bars 50
    siton examples/07_purged_kfold_cv.py -s BTC/USDT -t 1h --start 2024-01-01
"""

from siton.sdk import ema_cross, adx, Strategy, backtest, run
from siton.data import generate_sample

# ── Signal: EMA crossover filtered by ADX trend strength ─────────────────────
# 2 × 2 = 4 EMA combos × 3 ADX thresholds = 12 total parameter combos
signal = ema_cross(fast=[8, 12], slow=[26, 50]) & adx(periods=[14], thresholds=[20, 25, 30])

# ── Strategy: enable purged K-fold CV ────────────────────────────────────────
STRATEGY = Strategy(
    "TrendFollowing",
    signal=signal,
    top=5,
    sort="sharpe_ratio",
    long_only=True,
    # Purged K-Fold CV settings
    cv=True,
    cv_folds=5,
    purge_bars=50,
)

# ── Programmatic API demo ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        print("=" * 70)
        print("  Purged K-Fold CV — Programmatic API demo")
        print("=" * 70)

        df = generate_sample(n=5000)
        results = run(STRATEGY, df)

        print(f"\nResult type: {type(results).__name__}")
        print(f"Keys: {list(results.keys())}")
        print(f"Folds run: {len(results['folds'])}")

        print("\nAggregated OOS top results:")
        for i, r in enumerate(results["aggregated"], 1):
            n_folds = len(results["folds"])
            cons_str = f"{int(round(r.cv_consistency * n_folds))}/{n_folds}"
            params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            print(
                f"  {i}. Sharpe={r.sharpe_ratio:.3f}  Return={r.total_return_pct:+.2f}%"
                f"  Consistency={cons_str}  [{params_str}]"
            )

        print()
        print("  cv_consistency interpretation:")
        print("  - 5/5: profitable in every fold (very robust)")
        print("  - 3/5: profitable in majority of folds (acceptable)")
        print("  - 1/5: mostly overfitting (avoid this combo)")

        print()
        print("─" * 70)
        print("  Now running via backtest() (full CLI output):")
        print("─" * 70)

    backtest(STRATEGY)
