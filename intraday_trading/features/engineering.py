from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    """Create technical indicators required for the model."""

    def __init__(self, lookback: int) -> None:
        self.lookback = lookback

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

        df["macd_fast"] = df["close"].ewm(span=12, adjust=False).mean()
        df["macd_slow"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["macd_fast"] - df["macd_slow"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        df["return"] = df["close"].pct_change()
        df["return_change"] = df["return"].diff()

        df = df.drop(columns=["macd_fast", "macd_slow"])
        df = df.dropna().copy()
        return df

    @property
    def feature_columns(self) -> list[str]:
        return [
            "open",
            "high",
            "low",
            "close",
            "rsi",
            "ema_20",
            "ema_50",
            "macd",
            "macd_signal",
            "macd_hist",
            "return",
            "return_change",
        ]
