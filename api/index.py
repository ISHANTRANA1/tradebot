"""Vercel serverless entrypoint: FastAPI wrapper around the tradebot engine."""
import sys
from urllib.parse import parse_qs
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from tradebot import data  # noqa: E402
from tradebot.backtest import Backtester, BacktestResult  # noqa: E402
from tradebot.config import BacktestConfig  # noqa: E402
from tradebot.strategies import available, create  # noqa: E402

api = FastAPI(title="tradebot API", description="Paper trading only. Not financial advice.")


class RunRequest(BaseModel):
    strategy: str = "sma_cross"
    params: dict[str, Any] = {}
    bars: int = Field(1000, ge=50, le=5000)
    seed: int = 42
    cash: float = Field(10_000, gt=0, le=1e9)
    commission: float = Field(0.001, ge=0, lt=0.1)
    slippage: float = Field(0.0005, ge=0, lt=0.1)
    risk: float = Field(1.0, gt=0, le=1)
    stop_loss: float | None = Field(None, gt=0, lt=1)


def _config(r: RunRequest) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=r.cash, commission=r.commission, slippage=r.slippage,
        risk_fraction=r.risk, stop_loss=r.stop_loss,
    )


def _series(s, max_points: int = 400) -> dict[str, list]:
    step = max(len(s) // max_points, 1)
    s = s.iloc[::step]
    return {"t": [d.strftime("%Y-%m-%d") for d in s.index], "v": [round(float(v), 2) for v in s]}


def _clean(m: dict[str, float]) -> dict[str, float | None]:
    return {k: (None if v in (float("inf"), float("-inf")) else round(v, 4)) for k, v in m.items()}


def _summary(res: BacktestResult) -> dict[str, Any]:
    return {"strategy": res.strategy, "metrics": _clean(res.metrics)}


@api.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.get("/api/strategies")
def strategies() -> list[dict[str, Any]]:
    return [
        {"name": n, "description": c.description, "params": vars(c())}
        for n, c in sorted(available().items())
    ]


@api.post("/api/backtest")
def backtest(req: RunRequest) -> dict[str, Any]:
    try:
        df = data.synthetic(bars=req.bars, seed=req.seed)
        res = Backtester(_config(req)).run(df, create(req.strategy, **req.params))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip("'\"")) from exc
    return {
        **_summary(res),
        "equity": _series(res.equity),
        "benchmark": _series(res.benchmark),
        "trades": [
            {"entry": t.entry_time.strftime("%Y-%m-%d"), "exit": t.exit_time.strftime("%Y-%m-%d"),
             "entry_price": round(t.entry_price, 2), "exit_price": round(t.exit_price, 2),
             "pnl": round(t.pnl, 2), "return_pct": round(t.return_pct, 2)}
            for t in res.trades[-50:]
        ],
    }


@api.post("/api/compare")
def compare(req: RunRequest) -> list[dict[str, Any]]:
    try:
        df = data.synthetic(bars=req.bars, seed=req.seed)
        bt = Backtester(_config(req))
        results = [bt.run(df, cls()) for cls in available().values()]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    results.sort(key=lambda r: r.metrics["sharpe"], reverse=True)
    return [_summary(r) for r in results]


class RouteFix:
    """Vercel rewrites /api/<name> to /api/index?route=<name>; restore the original path."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].rstrip("/") in ("/api", "/api/index"):
            route = parse_qs(scope.get("query_string", b"").decode()).get("route", [""])[0].strip("/")
            if route:
                scope = {**scope, "path": f"/api/{route}", "raw_path": f"/api/{route}".encode()}
        await self.inner(scope, receive, send)


app = RouteFix(api)  # Vercel looks for `app`
