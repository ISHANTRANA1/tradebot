from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Immutable settings for a backtest run."""

    initial_cash: float = 10_000.0
    commission: float = 0.001      # 0.1% per side
    slippage: float = 0.0005       # 0.05% adverse price move per fill
    risk_fraction: float = 1.0     # share of equity committed per entry (0-1]
    stop_loss: float | None = None  # e.g. 0.05 = exit if price falls 5% below entry
    bars_per_year: int = 252

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 <= self.commission < 0.1:
            raise ValueError("commission must be in [0, 0.1)")
        if not 0 <= self.slippage < 0.1:
            raise ValueError("slippage must be in [0, 0.1)")
        if not 0 < self.risk_fraction <= 1:
            raise ValueError("risk_fraction must be in (0, 1]")
        if self.stop_loss is not None and not 0 < self.stop_loss < 1:
            raise ValueError("stop_loss must be in (0, 1)")
