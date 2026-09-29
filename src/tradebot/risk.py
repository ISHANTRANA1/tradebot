from __future__ import annotations

from tradebot.config import BacktestConfig


class RiskManager:
    """Position sizing and stop-loss rules."""

    def __init__(self, config: BacktestConfig) -> None:
        self._cfg = config

    def entry_qty(self, cash: float, equity: float, ref_price: float) -> float:
        """Quantity to buy, capped by available cash after slippage and fees."""
        budget = min(cash, equity * self._cfg.risk_fraction)
        unit_cost = ref_price * (1 + self._cfg.slippage) * (1 + self._cfg.commission)
        return max(budget / unit_cost, 0.0)

    def stop_price(self, entry_price: float) -> float | None:
        sl = self._cfg.stop_loss
        return entry_price * (1 - sl) if sl else None
