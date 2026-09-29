from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tradebot.models import Fill, Side, Trade


@dataclass
class Portfolio:
    """Long-only single-asset portfolio that records fills and closed trades."""

    cash: float
    qty: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    _entry_time: pd.Timestamp | None = None
    _entry_cost: float = 0.0  # cash spent incl. fees
    _entry_price: float = 0.0

    @property
    def in_position(self) -> bool:
        return self.qty > 0

    @property
    def entry_price(self) -> float | None:
        return self._entry_cost / self.qty if self.in_position else None

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price

    def apply(self, fill: Fill) -> None:
        if fill.side is Side.BUY:
            if self.in_position:
                raise RuntimeError("Already in position")
            cost = fill.qty * fill.price + fill.fee
            if cost > self.cash + 1e-9:
                raise ValueError("Insufficient cash")
            self.cash -= cost
            self.qty = fill.qty
            self._entry_time = fill.timestamp
            self._entry_cost = cost
            self._entry_price = fill.price
        else:
            if not self.in_position or fill.qty > self.qty + 1e-9:
                raise ValueError("Cannot sell more than held")
            proceeds = fill.qty * fill.price - fill.fee
            self.cash += proceeds
            self.trades.append(
                Trade(
                    entry_time=self._entry_time,  # type: ignore[arg-type]
                    exit_time=fill.timestamp,
                    entry_price=self._entry_price,
                    exit_price=fill.price,
                    qty=fill.qty,
                    pnl=proceeds - self._entry_cost,
                )
            )
            self.qty = 0.0
            self._entry_time = None
            self._entry_cost = 0.0
        self.fills.append(fill)
