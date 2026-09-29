from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradebot import metrics
from tradebot.broker import PaperBroker
from tradebot.config import BacktestConfig
from tradebot.data import validate
from tradebot.models import Fill, Side, Trade
from tradebot.portfolio import Portfolio
from tradebot.risk import RiskManager
from tradebot.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    equity: pd.Series
    fills: list[Fill]
    trades: list[Trade]
    metrics: dict[str, float]
    benchmark: pd.Series  # buy & hold equity on the same data


class Backtester:
    """Bar-by-bar simulation. Signals from bar t-1 execute at the open of bar t."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.cfg = config or BacktestConfig()
        self._broker = PaperBroker(self.cfg)
        self._risk = RiskManager(self.cfg)

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        df = validate(df)
        target = strategy.generate(df).reindex(df.index).fillna(0).astype(int).shift(1).fillna(0).astype(int)
        pf = Portfolio(cash=self.cfg.initial_cash)
        equity: list[float] = []
        blocked = False  # after a stop-out, wait for the signal to reset before re-entering

        for ts, bar in df.iterrows():
            # 1) protective stop, checked against this bar's range
            if pf.in_position and (stop := self._risk.stop_price(pf.entry_price)) is not None:  # type: ignore[arg-type]
                if bar["low"] <= stop:
                    ref = min(bar["open"], stop)  # gap-down fills at the open
                    pf.apply(self._broker.execute(ts, Side.SELL, pf.qty, ref, "stop_loss"))
                    blocked = True
            # 2) act on yesterday's signal at today's open
            want = target.loc[ts]
            if want == 0:
                blocked = False
                if pf.in_position:
                    pf.apply(self._broker.execute(ts, Side.SELL, pf.qty, bar["open"]))
            elif not pf.in_position and not blocked:
                qty = self._risk.entry_qty(pf.cash, pf.equity(bar["open"]), bar["open"])
                if qty > 0:
                    pf.apply(self._broker.execute(ts, Side.BUY, qty, bar["open"]))
            equity.append(pf.equity(bar["close"]))

        if pf.in_position:  # mark-to-market close so the last trade is counted
            last_ts, last = df.index[-1], df.iloc[-1]
            pf.apply(self._broker.execute(last_ts, Side.SELL, pf.qty, last["close"], "end_of_data"))
            equity[-1] = pf.cash

        eq = pd.Series(equity, index=df.index, name="equity")
        bench = self.cfg.initial_cash * df["close"] / df["close"].iloc[0]
        return BacktestResult(
            strategy=str(strategy),
            equity=eq,
            fills=pf.fills,
            trades=pf.trades,
            metrics=metrics.compute(eq, pf.trades, self.cfg.bars_per_year),
            benchmark=bench.rename("buy_hold"),
        )
