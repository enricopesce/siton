"""Data ingestion — exchange via ccxt, CSV, or synthetic generator."""

import math
import os
import warnings
from pathlib import Path

import numpy as np
import polars as pl

# Timeframe → seconds (mirrors sdk._TIMEFRAME_SECONDS; kept local to avoid circular import)
_TF_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


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


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(
    exchange: str,
    symbol: str,
    timeframe: str,
    start: str | None,
    end: str | None,
    limit: int,
) -> Path:
    """Return the parquet cache file path for a given fetch request.

    Layout: $SITON_CACHE_DIR/{exchange}/{symbol}/{timeframe}/{key}.parquet
    Key:
      - Fixed range  : "{start}_{end}"   — immutable, cached forever
      - Open-ended   : "{start}_live"    — incrementally extended
      - Limit-only   : "last_{limit}"    — sliding window, refreshed when stale
    """
    root = Path(os.environ.get("SITON_CACHE_DIR", Path.home() / ".cache" / "siton"))
    sym = symbol.replace("/", "-")
    if start and end:
        fname = f"{start}_{end}.parquet"
    elif start:
        fname = f"{start}_live.parquet"
    else:
        fname = f"last_{limit}.parquet"
    return root / exchange / sym / timeframe / fname


def _fetch_raw(
    exchange,
    symbol: str,
    timeframe: str,
    since: int,
    end_ms: int | None,
    batch: int,
    limit: int | None = None,
) -> list:
    """Paginate ccxt.fetch_ohlcv and return raw [[ts,o,h,l,c,v], ...] rows."""
    all_ohlcv: list = []
    use_limit = limit is not None
    while True:
        if use_limit and len(all_ohlcv) >= limit:
            break
        fetch_size = min((limit - len(all_ohlcv)) if use_limit else batch, batch)
        chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=fetch_size)
        if not chunk:
            break
        if end_ms and chunk[0][0] >= end_ms:
            break
        all_ohlcv.extend(chunk)
        since = chunk[-1][0] + 1
        if end_ms and since >= end_ms:
            break
    return all_ohlcv


def _raw_to_df(ohlcv: list) -> pl.DataFrame:
    """Convert raw ccxt rows to a sorted Polars DataFrame."""
    return (
        pl.DataFrame(
            ohlcv,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row",
        )
        .with_columns(
            (pl.col("timestamp").cast(pl.Int64) * 1000).cast(pl.Datetime("us")).alias("datetime"),
        )
        .sort("timestamp")
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    exchange_id: str = "binance",
    limit: int = 5000,
    start: str | None = None,
    end: str | None = None,
    cache: bool = True,
    verbose: bool = False,
) -> pl.DataFrame:
    """Fetch OHLCV candles from exchange via ccxt with an intelligent disk cache.

    Cache lives at ~/.cache/siton/ (override with $SITON_CACHE_DIR):
      - Fixed range (start+end): cached permanently — historical data never changes.
      - Open-ended  (start only): incrementally extended — only new bars are fetched.
      - Limit-only  (no dates):   refreshed when last bar is >2 bars behind now.

    Args:
        start:   Optional start date YYYY-MM-DD (inclusive).
        end:     Optional end date YYYY-MM-DD (inclusive, through end of day).
        cache:   Use local disk cache (default True).
        verbose: Print cache/fetch status messages (default False).
    """
    import time as _time

    tf_ms = _TF_SECONDS.get(timeframe, 3600) * 1000
    end_ms = _parse_date_ms(end) + 86_400_000 if end else None
    # Only apply the `limit` cap when no date range is given at all
    use_limit = start is None and end_ms is None

    # ------------------------------------------------------------------
    # Step 1: Try to return directly from cache (no exchange init needed)
    # ------------------------------------------------------------------
    path = _cache_path(exchange_id, symbol, timeframe, start, end, limit) if cache else None
    cached: pl.DataFrame | None = None

    if path and path.exists():
        cached = pl.read_parquet(path)
        last_ts = int(cached["timestamp"][-1])
        now_ms = int(_time.time() * 1000)

        if end is not None:
            # Fixed historical range — always valid, never re-fetch
            if verbose:
                print(f"\n[*] Loaded {len(cached)} {timeframe} candles from cache.")
            _check_gaps(cached, timeframe)
            return cached

        if not use_limit and last_ts >= now_ms - tf_ms:
            # Open-ended: last bar is within one timeframe of now
            if verbose:
                print(f"\n[*] Loaded {len(cached)} {timeframe} candles from cache (up to date).")
            _check_gaps(cached, timeframe)
            return cached

        if use_limit and last_ts >= now_ms - tf_ms * 2:
            # Limit-only: at most 2 bars stale
            if verbose:
                print(f"\n[*] Loaded {len(cached)} {timeframe} candles from cache (up to date).")
            _check_gaps(cached, timeframe)
            return cached.tail(limit)

    # ------------------------------------------------------------------
    # Step 2: Init exchange, then fetch (full or incremental delta)
    # ------------------------------------------------------------------
    import ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()
    tf_ms = exchange.parse_timeframe(timeframe) * 1000  # precise value from exchange
    batch = 1000

    if cached is not None:
        # Incremental update — fetch only candles newer than what we have
        last_ts = int(cached["timestamp"][-1])
        delta_since = last_ts + tf_ms

        if verbose:
            from datetime import datetime, timezone

            last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"\n[*] Cache hit ({len(cached)} candles to {last_dt}), fetching new candles...")

        raw_delta = _fetch_raw(exchange, symbol, timeframe, delta_since, end_ms, batch)

        if raw_delta:
            if verbose:
                print(f"    +{len(raw_delta)} new candles fetched.")
            df_new = _raw_to_df(raw_delta)
            merged = pl.concat([cached, df_new]).unique("timestamp").sort("timestamp")
            if use_limit:
                merged = merged.tail(limit)
            if end_ms:
                merged = merged.filter(pl.col("timestamp") < end_ms)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.write_parquet(path)
            _check_gaps(merged, timeframe)
            return merged

        if verbose:
            print("    No new candles available. Using cached data.")
        _check_gaps(cached, timeframe)
        return cached

    # Full fetch from scratch
    since = _parse_date_ms(start) if start else exchange.milliseconds() - limit * tf_ms

    if verbose:
        date_desc = (
            f" from {start} to {end}"
            if start and end
            else f" from {start}"
            if start
            else f" until {end}"
            if end
            else ""
        )
        limit_desc = f"{limit} " if not start else ""
        print(
            f"\n[*] Fetching {limit_desc}{timeframe} candles for {symbol} "
            f"from {exchange_id}{date_desc}..."
        )

    raw = _fetch_raw(exchange, symbol, timeframe, since, end_ms, batch, limit if use_limit else None)

    # Deduplicate
    seen: set = set()
    unique: list = []
    for c in raw:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)

    if end_ms:
        unique = [c for c in unique if c[0] < end_ms]
    if use_limit:
        unique = unique[:limit]

    df = _raw_to_df(unique)

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

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
