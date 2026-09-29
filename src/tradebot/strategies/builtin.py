from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradebot.strategies.base import Strategy, register


@register
@dataclass
class BuyAndHold(Strategy):
    name = "buy_hold"
    description = "Benchmark: always long."

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index)


@register
@dataclass
class SmaCross(Strategy):
    name = "sma_cross"
    description = "Long while the fast SMA is above the slow SMA (trend following)."
    fast: int = 10
    slow: int = 30

    def __post_init__(self) -> None:
        if not 0 < self.fast < self.slow:
            raise ValueError("Require 0 < fast < slow")

    def generate(self, df: pd.DataFrame) -> pd.Series:
        fast = df["close"].rolling(self.fast).mean()
        slow = df["close"].rolling(self.slow).mean()
        return (fast > slow).astype(int)


@register
@dataclass
class RsiReversion(Strategy):
    name = "rsi"
    description = "Buy when RSI drops below `low`, sell when it rises above `high` (mean reversion)."
    period: int = 14
    low: float = 30.0
    high: float = 70.0

    def __post_init__(self) -> None:
        if self.period < 2 or not 0 < self.low < self.high < 100:
            raise ValueError("Require period >= 2 and 0 < low < high < 100")

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(100.0)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._rsi(df["close"]).to_numpy()
        out = np.zeros(len(rsi), dtype=int)
        state = 0
        for i, v in enumerate(rsi):
            if i >= self.period:
                if state == 0 and v < self.low:
                    state = 1
                elif state == 1 and v > self.high:
                    state = 0
            out[i] = state
        return pd.Series(out, index=df.index)


@register
@dataclass
class Bollinger(Strategy):
    name = "bollinger"
    description = "Buy below the lower band, exit at the moving average (mean reversion)."
    window: int = 20
    num_std: float = 2.0

    def __post_init__(self) -> None:
        if self.window < 2 or self.num_std <= 0:
            raise ValueError("Require window >= 2 and num_std > 0")

    def generate(self, df: pd.DataFrame) -> pd.Series:
        mid = df["close"].rolling(self.window).mean()
        lower = mid - self.num_std * df["close"].rolling(self.window).std()
        close = df["close"].to_numpy()
        mid_a, low_a = mid.to_numpy(), lower.to_numpy()
        out = np.zeros(len(df), dtype=int)
        state = 0
        for i in range(len(df)):
            if np.isnan(mid_a[i]):
                continue
            if state == 0 and close[i] < low_a[i]:
                state = 1
            elif state == 1 and close[i] >= mid_a[i]:
                state = 0
            out[i] = state
        return pd.Series(out, index=df.index)


@register
@dataclass
class Breakout(Strategy):
    name = "breakout"
    description = "Donchian channel: long on a new `entry`-bar high, exit on a new `exit`-bar low."
    entry: int = 20
    exit: int = 10

    def __post_init__(self) -> None:
        if self.entry < 2 or self.exit < 2:
            raise ValueError("entry and exit must be >= 2")

    def generate(self, df: pd.DataFrame) -> pd.Series:
        hi = df["high"].rolling(self.entry).max().shift(1).to_numpy()
        lo = df["low"].rolling(self.exit).min().shift(1).to_numpy()
        close = df["close"].to_numpy()
        out = np.zeros(len(df), dtype=int)
        state = 0
        for i in range(len(df)):
            if state == 0 and close[i] > hi[i]:  # NaN compares False
                state = 1
            elif state == 1 and close[i] < lo[i]:
                state = 0
            out[i] = state
        return pd.Series(out, index=df.index)
