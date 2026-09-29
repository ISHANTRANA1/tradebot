from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Fill:
    timestamp: pd.Timestamp
    side: Side
    qty: float
    price: float
    fee: float
    reason: str = "signal"


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round trip (entry + exit)."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float

    @property
    def return_pct(self) -> float:
        return (self.exit_price / self.entry_price - 1.0) * 100.0
