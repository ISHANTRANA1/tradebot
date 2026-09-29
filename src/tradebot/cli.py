from __future__ import annotations

import argparse
import itertools
import sys
from typing import Sequence

import pandas as pd

from tradebot import __version__, data, report
from tradebot.backtest import Backtester, BacktestResult
from tradebot.config import BacktestConfig
from tradebot.strategies import available, create, parse_params


def _add_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--source", choices=["synthetic", "csv", "yahoo"], default="synthetic")
    g.add_argument("--csv", help="path to OHLC CSV (with --source csv)")
    g.add_argument("--symbol", help="ticker (with --source yahoo)")
    g.add_argument("--bars", type=int, default=1000, help="synthetic bars (default 1000)")
    g.add_argument("--seed", type=int, default=42, help="synthetic RNG seed")
    c = p.add_argument_group("simulation")
    c.add_argument("--cash", type=float, default=10_000.0)
    c.add_argument("--commission", type=float, default=0.001)
    c.add_argument("--slippage", type=float, default=0.0005)
    c.add_argument("--risk", type=float, default=1.0, help="fraction of equity per entry")
    c.add_argument("--stop-loss", type=float, default=None, help="e.g. 0.05 for 5%%")
    p.add_argument("--plot", metavar="PNG", help="save equity/drawdown chart")


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "csv":
        if not args.csv:
            raise SystemExit("--csv is required with --source csv")
        return data.load_csv(args.csv)
    if args.source == "yahoo":
        if not args.symbol:
            raise SystemExit("--symbol is required with --source yahoo")
        return data.load_yahoo(args.symbol)
    return data.synthetic(bars=args.bars, seed=args.seed)


def _config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=args.cash, commission=args.commission, slippage=args.slippage,
        risk_fraction=args.risk, stop_loss=args.stop_loss,
    )


def _maybe_plot(args: argparse.Namespace, results: list[BacktestResult]) -> None:
    if args.plot:
        out = report.plot_equity(results, args.plot)
        report.console.print(f"[dim]Chart saved to {out}[/]")


def cmd_list(_: argparse.Namespace) -> int:
    from rich.table import Table

    t = Table(title="Strategies")
    for c in ("Name", "Parameters (defaults)", "Description"):
        t.add_column(c)
    for name, cls in sorted(available().items()):
        t.add_row(name, ", ".join(f"{k}={v}" for k, v in vars(cls()).items()) or "-", cls.description)
    report.console.print(t)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    df = _load(args)
    strat = create(args.strategy, **parse_params(args.param))
    res = Backtester(_config(args)).run(df, strat)
    report.console.print(f"[dim]{len(df)} bars, {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}[/]")
    report.show_result(res)
    _maybe_plot(args, [res])
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    df = _load(args)
    bt = Backtester(_config(args))
    results = [bt.run(df, cls()) for cls in available().values()]
    results.sort(key=lambda r: r.metrics["sharpe"], reverse=True)
    report.show_ranking(results, "All strategies (default params), ranked by Sharpe")
    _maybe_plot(args, results)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    grid: dict[str, list] = {}
    for g in args.grid:
        key, _, vals = g.partition("=")
        grid[key] = [parse_params([f"x={v}"])["x"] for v in vals.split(",")]
    if not grid:
        raise SystemExit("Provide at least one --grid name=v1,v2,...")
    df = _load(args)
    bt = Backtester(_config(args))
    results: list[BacktestResult] = []
    for combo in itertools.product(*grid.values()):
        try:
            results.append(bt.run(df, create(args.strategy, **dict(zip(grid, combo)))))
        except ValueError:
            continue  # skip invalid combos such as fast >= slow
    if not results:
        raise SystemExit("No valid parameter combinations")
    results.sort(key=lambda r: r.metrics[args.metric], reverse=True)
    report.show_ranking(results[: args.top], f"Top {min(args.top, len(results))} by {args.metric}")
    report.console.print(
        "[yellow]Warning:[/] optimising on the same data you evaluate on overfits. "
        "Validate on a different --seed or date range."
    )
    _maybe_plot(args, results[:3])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradebot", description="Paper-trading & backtesting CLI (no real money).")
    p.add_argument("--version", action="version", version=f"tradebot {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show available strategies").set_defaults(func=cmd_list)

    b = sub.add_parser("backtest", help="run one strategy")
    b.add_argument("strategy", choices=sorted(available()))
    b.add_argument("-p", "--param", action="append", default=[], metavar="K=V", help="strategy parameter")
    _add_data_args(b)
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("compare", help="run every strategy and rank them")
    _add_data_args(c)
    c.set_defaults(func=cmd_compare)

    o = sub.add_parser("optimize", help="grid-search strategy parameters")
    o.add_argument("strategy", choices=sorted(available()))
    o.add_argument("-g", "--grid", action="append", default=[], metavar="K=V1,V2", help="values to try")
    o.add_argument("--metric", default="sharpe", choices=["sharpe", "total_return", "cagr", "profit_factor"])
    o.add_argument("--top", type=int, default=5)
    _add_data_args(o)
    o.set_defaults(func=cmd_optimize)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
        report.console.print(f"[red]Error:[/] {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
