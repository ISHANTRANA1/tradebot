import numpy as np
import pandas as pd
import pytest

from tradebot import data, metrics
from tradebot.backtest import Backtester
from tradebot.broker import PaperBroker
from tradebot.cli import main
from tradebot.config import BacktestConfig
from tradebot.models import Side
from tradebot.portfolio import Portfolio
from tradebot.strategies import available, create, parse_params


def frame(closes, opens=None):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    c = np.array(closes, float)
    o = np.array(opens, float) if opens is not None else c
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c}, index=idx)


ZERO = BacktestConfig(commission=0, slippage=0)


def test_config_validation():
    with pytest.raises(ValueError):
        BacktestConfig(initial_cash=0)
    with pytest.raises(ValueError):
        BacktestConfig(stop_loss=1.5)


def test_portfolio_round_trip_pnl():
    b = PaperBroker(BacktestConfig(commission=0.001, slippage=0))
    pf = Portfolio(cash=2000)
    ts = pd.Timestamp("2024-01-02")
    pf.apply(b.execute(ts, Side.BUY, 10, 100))
    assert pf.cash == pytest.approx(2000 - 1000 - 1.0)
    pf.apply(b.execute(ts, Side.SELL, 10, 110))
    assert pf.trades[0].pnl == pytest.approx(100 - 1.0 - 1.1)
    assert not pf.in_position


def test_portfolio_rejects_bad_orders():
    pf = Portfolio(cash=100)
    b = PaperBroker(ZERO)
    with pytest.raises(ValueError):
        pf.apply(b.execute(pd.Timestamp("2024-01-02"), Side.BUY, 10, 100))  # not enough cash
    with pytest.raises(ValueError):
        pf.apply(b.execute(pd.Timestamp("2024-01-02"), Side.SELL, 1, 100))  # nothing to sell


def test_buy_hold_matches_benchmark_without_costs():
    df = data.synthetic(300, seed=1)
    res = Backtester(ZERO).run(df, create("buy_hold"))
    # signal is shifted one bar so entry is at bar-1 open
    assert res.metrics["total_return"] == pytest.approx(res.benchmark.iloc[-1] / res.benchmark.iloc[0] - 1, abs=0.02)
    assert len(res.trades) == 1


def test_no_lookahead_signal_executes_next_open():
    closes = [10, 10, 10, 20, 20, 20]
    opens = [10, 10, 10, 10, 20, 20]
    df = frame(closes, opens)

    class Flip(type(create("buy_hold"))):
        def generate(self, d):
            return (d["close"] > 15).astype(int)

    res = Backtester(ZERO).run(df, Flip())
    buy = res.fills[0]
    assert buy.timestamp == df.index[4]  # signal on bar 3 -> fill on bar 4
    assert buy.price == 20


def test_stop_loss_triggers_and_blocks_reentry():
    closes = [100, 100, 100, 80, 80, 80]
    df = frame(closes, [100, 100, 100, 100, 80, 80])
    cfg = BacktestConfig(commission=0, slippage=0, stop_loss=0.1)
    res = Backtester(cfg).run(df, create("buy_hold"))
    reasons = [f.reason for f in res.fills]
    assert "stop_loss" in reasons
    assert reasons.count("signal") == 1  # no re-entry while signal stays on


def test_equity_never_negative_and_indexed():
    df = data.synthetic(500, seed=3)
    for name in available():
        res = Backtester().run(df, create(name))
        assert (res.equity > 0).all()
        assert res.equity.index.equals(df.index)


def test_sma_validation_and_params():
    with pytest.raises(ValueError):
        create("sma_cross", fast=30, slow=10)
    assert parse_params(["fast=5", "mode=abc"]) == {"fast": 5, "mode": "abc"}
    with pytest.raises(KeyError):
        create("nope")


def test_metrics_drawdown():
    eq = pd.Series([100, 120, 90, 110.0])
    assert metrics.max_drawdown(eq) == pytest.approx(-0.25)


def test_csv_roundtrip(tmp_path):
    df = data.synthetic(50, seed=5)
    p = tmp_path / "x.csv"
    df.rename_axis("Date").to_csv(p)
    loaded = data.load_csv(p)
    assert len(loaded) == 50 and {"open", "close"} <= set(loaded.columns)


def test_validate_rejects_bad_data():
    with pytest.raises(ValueError):
        data.validate(pd.DataFrame({"close": [1, 2]}, index=pd.bdate_range("2024-01-01", periods=2)))


def test_cli_smoke(capsys, tmp_path):
    assert main(["list"]) == 0
    assert main(["backtest", "sma_cross", "-p", "fast=5", "-p", "slow=20", "--bars", "300"]) == 0
    assert main(["compare", "--bars", "300"]) == 0
    assert main(["optimize", "sma_cross", "-g", "fast=5,10", "-g", "slow=20,40", "--bars", "300"]) == 0
    png = tmp_path / "c.png"
    assert main(["backtest", "rsi", "--bars", "300", "--plot", str(png)]) == 0
    assert png.exists()
    assert main(["backtest", "sma_cross", "-p", "fast=50", "-p", "slow=10"]) == 2


def test_web_api():
    import importlib.util, pathlib
    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location("api_index", pathlib.Path(__file__).parent.parent / "api" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c = TestClient(mod.app)
    assert c.get("/api/health").json() == {"status": "ok"}
    assert len(c.get("/api/strategies").json()) == 5
    r = c.post("/api/backtest", json={"strategy": "sma_cross", "params": {"fast": 5, "slow": 20}, "bars": 300})
    assert r.status_code == 200 and len(r.json()["equity"]["v"]) > 10
    assert c.post("/api/backtest", json={"strategy": "sma_cross", "params": {"fast": 50, "slow": 5}}).status_code == 400
    assert c.post("/api/backtest", json={"strategy": "nope"}).status_code == 400
    assert c.post("/api/backtest", json={"strategy": "rsi", "params": {"bogus": 1}}).status_code == 400
    assert c.post("/api/backtest", json={"bars": 999999}).status_code == 422
    assert len(c.post("/api/compare", json={"bars": 300}).json()) == 5
