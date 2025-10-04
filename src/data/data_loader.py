"""Data loading utilities for fetching historical OHLC data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


@dataclass
class YFinanceConfig:
    ticker: str
    interval: str = "15m"
    lookback_days: int = 60

    def resolve_start_end(self) -> tuple[datetime, datetime]:
        end = datetime.utcnow()
        start = end - timedelta(days=self.lookback_days)
        return start, end


class YFinanceLoader:
    """Wrapper around ``yfinance`` for fetching OHLCV data."""

    def __init__(self, config: YFinanceConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        start, end = self.config.resolve_start_end()
        data = yf.download(
            tickers=self.config.ticker,
            interval=self.config.interval,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            raise ValueError(
                f"No data returned from yfinance for ticker {self.config.ticker}"
            )
        data = data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)
        return data


__all__ = ["YFinanceConfig", "YFinanceLoader"]
