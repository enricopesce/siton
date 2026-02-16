"""Data ingestion — exchange via ccxt, CSV, or synthetic generator."""

import polars as pl


def _parse_date_ms(date_str: str) -> int:
    """Parse a date string (YYYY-MM-DD) to epoch milliseconds."""
    from datetime import datetime, timezone
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


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

    df = pl.DataFrame(
        all_ohlcv,
        schema=["timestamp", "open", "high", "low", "close", "volume"],
        orient="row",
    ).with_columns(
        (pl.col("timestamp").cast(pl.Int64) * 1000).cast(pl.Datetime("us")).alias("datetime"),
    ).sort("timestamp")

    return df


def load_csv(path: str, start: str | None = None, end: str | None = None) -> pl.DataFrame:
    """Load OHLCV from CSV. Expects columns: timestamp,open,high,low,close,volume."""
    df = pl.read_csv(path).sort("timestamp")
    if start:
        df = df.filter(pl.col("timestamp") >= _parse_date_ms(start))
    if end:
        df = df.filter(pl.col("timestamp") < _parse_date_ms(end) + 86_400_000)
    return df


def generate_sample(n: int = 10000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic OHLCV for testing without API keys."""
    import numpy as np

    rng = np.random.default_rng(seed)

    returns = rng.normal(0.0001, 0.02, n)
    close = 30000.0 * np.cumprod(1 + returns)

    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.uniform(100, 10000, n)
    timestamps = np.arange(n) * 3600_000

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
