from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, check OHLC integrity, and return a clean frame."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    df = df[~df.index.duplicated()].sort_index().dropna(subset=REQUIRED)
    if len(df) < 2:
        raise ValueError("Need at least 2 bars")
    if (df[REQUIRED] <= 0).any().any():
        raise ValueError("Prices must be positive")
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLC(V) data from CSV. First column (or 'date'/'datetime') is the timestamp."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    date_col = next((c for c in ("date", "datetime", "timestamp") if c in raw.columns), raw.columns[0])
    raw[date_col] = pd.to_datetime(raw[date_col])
    return validate(raw.set_index(date_col))


def load_yahoo(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Download daily bars via yfinance (optional dependency)."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install with: pip install 'tradebot[yahoo]'") from exc
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return validate(df)


def synthetic(
    bars: int = 1000,
    seed: int | None = 42,
    start_price: float = 100.0,
    annual_drift: float = 0.10,
    annual_vol: float = 0.25,
) -> pd.DataFrame:
    """Geometric-Brownian-motion OHLC data with a business-day index."""
    if bars < 2:
        raise ValueError("bars must be >= 2")
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    rets = rng.normal((annual_drift - 0.5 * annual_vol**2) * dt, annual_vol * np.sqrt(dt), bars)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.002, bars))
    span = np.abs(rng.normal(0, annual_vol * np.sqrt(dt) * 0.6, bars))
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(1_000, 100_000, bars)},
        index=idx,
    )
