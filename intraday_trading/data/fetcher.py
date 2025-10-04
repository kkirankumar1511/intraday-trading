from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class MarketDataFetcher:
    ticker: str
    interval: str
    history_days: int

    def fetch(self, end: Optional[datetime] = None) -> pd.DataFrame:
        end = end or datetime.utcnow()
        start = end - timedelta(days=self.history_days)
        data = yf.download(
            self.ticker,
            start=start,
            end=end,
            interval=self.interval,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            raise ValueError("No data returned from yfinance. Check the ticker or interval.")
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
        data = data.sort_index()
        return data
