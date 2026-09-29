from __future__ import annotations

import pandas as pd

from tradebot.config import BacktestConfig
from tradebot.models import Fill, Side


class PaperBroker:
    """Simulated broker: applies slippage and commission. Never touches real money."""

    def __init__(self, config: BacktestConfig) -> None:
        self._cfg = config

    def fill_price(self, side: Side, ref_price: float) -> float:
        s = self._cfg.slippage
        return ref_price * (1 + s) if side is Side.BUY else ref_price * (1 - s)

    def execute(
        self, ts: pd.Timestamp, side: Side, qty: float, ref_price: float, reason: str = "signal"
    ) -> Fill:
        price = self.fill_price(side, ref_price)
        fee = qty * price * self._cfg.commission
        return Fill(ts, side, qty, price, fee, reason)
