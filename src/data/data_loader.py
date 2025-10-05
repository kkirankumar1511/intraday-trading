"""Data loading utilities for fetching historical OHLC data."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import yfinance as yf
from yfinance.shared import _ERRORS


YFPricesMissingError = _ERRORS.get("YFPricesMissingError", Exception)

INTRADAY_MAX_LOOKBACK = {
    "15m": 59,
}


@dataclass
class YFinanceConfig:
    ticker: str
    interval: str = "15m"
    lookback_days: int = 60

    def resolve_period(self) -> str:
        return f"{self.lookback_days}d"

    def resolved_lookback_days(self) -> int:
        max_days = INTRADAY_MAX_LOOKBACK.get(self.interval)
        if max_days is None:
            return self.lookback_days
        return min(self.lookback_days, max_days)

    def build_download_kwargs(self) -> dict:
        lookback_days = self.resolved_lookback_days()
        if lookback_days != self.lookback_days:
            period = f"{lookback_days}d"
        else:
            period = self.resolve_period()
        return {
            "tickers": self.ticker,
            "interval": self.interval,
            "period": period,
            "auto_adjust": False,
            "progress": False,
        }


class YFinanceLoader:
    """Wrapper around ``yfinance`` for fetching OHLCV data."""

    def __init__(self, config: YFinanceConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        kwargs = self.config.build_download_kwargs()
        try:
            data = yf.download(**kwargs)
        except YFPricesMissingError:
            max_days = INTRADAY_MAX_LOOKBACK.get(self.config.interval)
            if max_days is None or kwargs["period"] == f"{max_days}d":
                raise
            data = yf.download(
                **kwargs | {"period": f"{max_days}d"}
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
