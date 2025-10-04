"""Feature engineering utilities for technical indicators."""
from __future__ import annotations

import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=period, adjust=False).mean()
    roll_down = down.ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    close = features["close"]

    features["ema_20"] = close.ewm(span=20, adjust=False).mean()
    features["ema_50"] = close.ewm(span=50, adjust=False).mean()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema_12 - ema_26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()

    features["rsi"] = compute_rsi(close, period=14)
    features["return_change"] = close.pct_change().fillna(0)

    features = features.dropna()
    return features


__all__ = ["add_technical_features"]
