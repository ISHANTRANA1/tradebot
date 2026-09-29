from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.models import Trade


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction."""
    return float((equity / equity.cummax() - 1.0).min())


def compute(equity: pd.Series, trades: list[Trade], bars_per_year: int = 252) -> dict[str, float]:
    rets = equity.pct_change().dropna()
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = max(len(equity) / bars_per_year, 1e-9)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    std = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(bars_per_year)) if std and std > 0 else 0.0
    wins = [t for t in trades if t.pnl > 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in trades if t.pnl <= 0)
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "trades": float(len(trades)),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "final_equity": float(equity.iloc[-1]),
    }
