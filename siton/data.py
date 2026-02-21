"""Data ingestion — exchange via ccxt, CSV, or synthetic generator."""

import math
import warnings

import numpy as np
import polars as pl


def _parse_date_ms(date_str: str) -> int:
    """Parse a date string (YYYY-MM-DD) to epoch milliseconds."""
    from datetime import datetime, timezone

    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _check_gaps(df: pl.DataFrame, timeframe: str, max_gap_factor: float = 2.5) -> None:
    """BUG-05: Warn on temporal gaps and invalid OHLCV values."""
    from siton.sdk import _TIMEFRAME_SECONDS

    tf_ms = _TIMEFRAME_SECONDS.get(timeframe, 3600) * 1000
    ts = df["timestamp"].to_numpy()
    if len(ts) < 2:
        return
    diffs = np.diff(ts)
    gaps = np.where(diffs > tf_ms * max_gap_factor)[0]
    if len(gaps):
        warnings.warn(
            f"{len(gaps)} temporal gap(s) detected "
            f"(worst: {diffs[gaps].max() / tf_ms:.1f}× bar size at index {gaps[0]}). "
            f"Results may be inaccurate.",
            stacklevel=3,
        )
    arr = {c: df[c].to_numpy() for c in ["open", "high", "low", "close", "volume"]}
    bad_hl = np.any(arr["high"] < arr["low"])
    bad_ch = np.any(arr["close"] > arr["high"])
    bad_cl = np.any(arr["close"] < arr["low"])
    bad_vol = np.any(arr["volume"] < 0)
    if bad_hl or bad_ch or bad_cl or bad_vol:
        warnings.warn(
            "OHLCV sanity check failed: high<low, close out of range, or negative volume.",
            stacklevel=3,
        )


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    exchange_id: str = "binance",
    limit: int = 5000,
    start: str | None = None,
    end: str | None = None,
) -> pl.DataFrame:
    """Fetch OHLCV candles from exchange via ccxt (paginates automatically).

    Args:
        start: Optional start date as YYYY-MM-DD (inclusive).
        end: Optional end date as YYYY-MM-DD (inclusive, fetches through end of day).
    """
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()

    tf_seconds = exchange.parse_timeframe(timeframe)
    tf_ms = tf_seconds * 1000
    batch = 1000

    if start:
        since = _parse_date_ms(start)
    else:
        since = exchange.milliseconds() - limit * tf_ms

    end_ms = _parse_date_ms(end) + 86_400_000 if end else None  # end of day

    # When date range is given, fetch until end (ignore limit)
    use_limit = end_ms is None
    all_ohlcv = []

    while True:
        if use_limit and len(all_ohlcv) >= limit:
            break
        remaining = limit - len(all_ohlcv) if use_limit else batch
        fetch_size = min(remaining, batch)
        chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=fetch_size)
        if not chunk:
            break
        # Stop if we've passed the end date
        if end_ms and chunk[0][0] >= end_ms:
            break
        all_ohlcv.extend(chunk)
        since = chunk[-1][0] + 1
        if end_ms and since >= end_ms:
            break

    seen = set()
    unique = []
    for c in all_ohlcv:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)
    all_ohlcv = unique

    # Trim to date range
    if end_ms:
        all_ohlcv = [c for c in all_ohlcv if c[0] < end_ms]
    if use_limit:
        all_ohlcv = all_ohlcv[:limit]

    df = (
        pl.DataFrame(
            all_ohlcv,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row",
        )
        .with_columns(
            (pl.col("timestamp").cast(pl.Int64) * 1000).cast(pl.Datetime("us")).alias("datetime"),
        )
        .sort("timestamp")
    )

    _check_gaps(df, timeframe)
    return df


def load_csv(
    path: str, start: str | None = None, end: str | None = None, timeframe: str = "1h"
) -> pl.DataFrame:
    """Load OHLCV from CSV. Expects columns: timestamp,open,high,low,close,volume."""
    df = pl.read_csv(path).sort("timestamp")
    if start:
        df = df.filter(pl.col("timestamp") >= _parse_date_ms(start))
    if end:
        df = df.filter(pl.col("timestamp") < _parse_date_ms(end) + 86_400_000)
    _check_gaps(df, timeframe)
    return df


def generate_sample(n: int = 10000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic OHLCV with GARCH volatility and fat-tailed returns."""
    if n == 0:
        return pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Int64),
                "open": pl.Series([], dtype=pl.Float64),
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
                "volume": pl.Series([], dtype=pl.Float64),
            }
        )

    rng = np.random.default_rng(seed)

    # GARCH(1,1)-like variance process for realistic volatility clustering
    omega, alpha_g, beta_g = 4e-6, 0.10, 0.85
    h = np.empty(n)
    h[0] = 0.02**2  # initial variance (~2% per bar)
    eps = rng.standard_t(df=5, size=n)  # fat-tailed Student-t innovations
    log_returns = np.empty(n)
    log_returns[0] = math.sqrt(h[0]) * eps[0]
    for i in range(1, n):
        h[i] = omega + alpha_g * log_returns[i - 1] ** 2 + beta_g * h[i - 1]
        log_returns[i] = math.sqrt(h[i]) * eps[i]

    # Use log-returns to guarantee positive prices (exp is always > 0)
    close = 30_000.0 * np.exp(np.cumsum(log_returns))
    returns = np.exp(log_returns) - 1.0  # simple returns for volume scaling

    # Open: previous close + overnight gap (random, usually small)
    open_ = np.empty(n)
    open_[0] = close[0] * (1 + rng.normal(0, 0.003))
    gap = rng.normal(0, 0.002, n - 1)
    open_[1:] = close[:-1] * (1 + gap)

    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.008, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.008, n))

    # Volume: correlated with absolute returns (higher vol → higher volume)
    base_vol = rng.lognormal(6.0, 1.0, n)  # heavy-tailed base
    vol_factor = 1 + 5 * np.abs(returns) / 0.02  # spike on large moves
    volume = base_vol * vol_factor

    timestamps = np.arange(n, dtype=np.int64) * 3_600_000

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
