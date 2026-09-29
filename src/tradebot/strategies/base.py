from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

_REGISTRY: dict[str, type["Strategy"]] = {}


class Strategy(ABC):
    """A strategy maps price data to a target position series: 1 = long, 0 = flat.

    Signals at bar *t* use only data up to and including bar *t*; the backtester
    executes them at the open of bar *t+1*, so there is no look-ahead bias.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series: ...

    def params(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def __str__(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.name}({inner})"


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator that adds a strategy to the registry."""
    if not getattr(cls, "name", None):
        raise TypeError(f"{cls.__name__} must define a `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"Duplicate strategy name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> dict[str, type[Strategy]]:
    return dict(_REGISTRY)


def create(name: str, **params: Any) -> Strategy:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown strategy '{name}'. Available: {', '.join(sorted(_REGISTRY))}") from None
    return cls(**params)


def parse_params(pairs: list[str]) -> dict[str, Any]:
    """Turn ['fast=10', 'mode=x'] into {'fast': 10, 'mode': 'x'}."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"Bad parameter '{pair}', expected key=value")
        try:
            out[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key.strip()] = raw
    return out
