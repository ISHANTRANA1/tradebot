from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tradebot.backtest import BacktestResult

console = Console()


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _color(x: float) -> str:
    return "green" if x > 0 else "red" if x < 0 else "white"


def show_result(res: BacktestResult, max_trades: int = 10) -> None:
    m = res.metrics
    bench_ret = res.benchmark.iloc[-1] / res.benchmark.iloc[0] - 1
    t = Table(show_header=False, box=None, padding=(0, 2))
    rows = [
        ("Total return", f"[{_color(m['total_return'])}]{_pct(m['total_return'])}[/]"),
        ("Buy & hold", f"[{_color(bench_ret)}]{_pct(bench_ret)}[/]"),
        ("CAGR", _pct(m["cagr"])),
        ("Sharpe", f"{m['sharpe']:.2f}"),
        ("Max drawdown", f"[red]{_pct(m['max_drawdown'])}[/]"),
        ("Trades", f"{int(m['trades'])}"),
        ("Win rate", f"{m['win_rate'] * 100:.1f}%"),
        ("Profit factor", f"{m['profit_factor']:.2f}"),
        ("Final equity", f"{m['final_equity']:,.2f}"),
    ]
    for k, v in rows:
        t.add_row(f"[bold]{k}[/]", v)
    console.print(Panel(t, title=f"[cyan]{res.strategy}[/]", expand=False))

    if res.trades:
        tt = Table(title=f"Last {min(max_trades, len(res.trades))} trades")
        for col in ("Entry", "Exit", "Entry px", "Exit px", "P&L", "Return"):
            tt.add_column(col, justify="right")
        for tr in res.trades[-max_trades:]:
            c = _color(tr.pnl)
            tt.add_row(
                f"{tr.entry_time:%Y-%m-%d}", f"{tr.exit_time:%Y-%m-%d}",
                f"{tr.entry_price:.2f}", f"{tr.exit_price:.2f}",
                f"[{c}]{tr.pnl:+,.2f}[/]", f"[{c}]{tr.return_pct:+.2f}%[/]",
            )
        console.print(tt)


def show_ranking(results: list[BacktestResult], title: str) -> None:
    t = Table(title=title)
    t.add_column("#", justify="right")
    t.add_column("Strategy")
    for col in ("Return", "Sharpe", "Max DD", "Trades", "Win %"):
        t.add_column(col, justify="right")
    for i, r in enumerate(results, 1):
        m = r.metrics
        t.add_row(
            str(i), r.strategy,
            f"[{_color(m['total_return'])}]{_pct(m['total_return'])}[/]",
            f"{m['sharpe']:.2f}", _pct(m["max_drawdown"]),
            str(int(m["trades"])), f"{m['win_rate'] * 100:.0f}",
        )
    console.print(t)


def plot_equity(results: list[BacktestResult], path: str | Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install with: pip install 'tradebot[plot]'") from exc
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1])
    for r in results:
        ax1.plot(r.equity.index, r.equity, label=r.strategy)
    ax1.plot(results[0].benchmark.index, results[0].benchmark, "k--", alpha=0.5, label="buy & hold")
    ax1.set_ylabel("Equity")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    for r in results:
        ax2.fill_between(r.equity.index, (r.equity / r.equity.cummax() - 1) * 100, alpha=0.3)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
