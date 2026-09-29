# tradebot

A small, typed **paper-trading and backtesting framework** with a rich terminal interface.
It simulates trades only. It never connects to a broker or spends real money.

## Install

```bash
pip install -e ".[plot,dev]"        # add ,yahoo for live ticker downloads
```

## Use

```bash
tradebot list                                            # available strategies
tradebot backtest sma_cross -p fast=10 -p slow=40        # one strategy on synthetic data
tradebot backtest rsi --stop-loss 0.05 --plot rsi.png    # with a 5% stop and a chart
tradebot compare --bars 1500 --seed 7                    # rank all strategies
tradebot optimize sma_cross -g fast=5,10,20 -g slow=30,50,100 --top 5
tradebot backtest bollinger --source csv --csv prices.csv
tradebot backtest sma_cross --source yahoo --symbol AAPL
```

CSV needs `date, open, high, low, close` columns (volume optional).
`python -m tradebot ...` works too.

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | Frozen, validated `BacktestConfig` |
| `data.py` | CSV / Yahoo / synthetic GBM loaders + validation |
| `strategies/` | `Strategy` ABC, registry, built-ins (SMA cross, RSI, Bollinger, Breakout, Buy & Hold) |
| `risk.py` | Position sizing and stop-loss |
| `broker.py` | Paper broker with slippage and commission |
| `portfolio.py` | Cash, position, fills, closed trades |
| `backtest.py` | Bar-by-bar engine; signals execute at the **next open** (no look-ahead) |
| `metrics.py` | Return, CAGR, Sharpe, drawdown, win rate, profit factor |
| `report.py` / `cli.py` | Rich tables, matplotlib charts, argparse interface |

## Add a strategy

```python
from dataclasses import dataclass
from tradebot.strategies import Strategy, register

@register
@dataclass
class MyStrategy(Strategy):
    name = "my_strategy"
    description = "Long when close > 50-bar mean."
    window: int = 50

    def generate(self, df):
        return (df["close"] > df["close"].rolling(self.window).mean()).astype(int)
```

Return a Series of `1` (long) or `0` (flat) aligned to `df.index`. It then shows up in every CLI command.

## Caveats

Backtests ignore taxes, liquidity limits, and regime change. Optimising on the data you evaluate
on overfits, so validate on different data. Nothing here is financial advice.

## Test

```bash
pytest
```
