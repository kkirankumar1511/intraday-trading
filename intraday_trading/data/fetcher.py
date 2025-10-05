from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from typing import Optional

import pandas as pd
import yfinance as yf


def _print(level: str, message: str, *args) -> None:
    formatted = message % args if args else message
    print(f"[{level}] {formatted}")


@dataclass
class MarketDataFetcher:
    ticker: str
    interval: str
    history_days: int
    max_retries: int = 3
    retry_delay: float = 1.5

    _INTRADAY_MAX_LOOKBACK = {
        "15m": 59,
    }

    def _resolve_period(self) -> str:
        """Return a period string that respects yfinance lookback limits."""

        max_days = self._INTRADAY_MAX_LOOKBACK.get(self.interval)
        if max_days is None:
            return f"{self.history_days}d"
        days = min(self.history_days, max_days)
        return f"{days}d"

    def _download_with_retries(self, **kwargs) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            _print(
                "DEBUG",
                "Attempt %d downloading data for ticker=%s interval=%s",
                attempt,
                self.ticker,
                self.interval,
            )
            try:
                data = yf.download(**kwargs)
            except Exception as exc:  # pragma: no cover - network layer
                last_error = exc
                _print(
                    "WARN",
                    "yfinance download attempt %d/%d failed for %s (%s): %s",
                    attempt,
                    self.max_retries,
                    self.ticker,
                    self.interval,
                    exc,
                )
            else:
                if not data.empty:
                    _print(
                        "INFO",
                        "Successfully downloaded %d rows for %s (%s) on attempt %d",
                        len(data),
                        self.ticker,
                        self.interval,
                        attempt,
                    )
                    return data
                last_error = ValueError("Empty dataframe returned from yfinance")
                _print(
                    "WARN",
                    "yfinance returned an empty dataframe on attempt %d/%d for %s (%s)",
                    attempt,
                    self.max_retries,
                    self.ticker,
                    self.interval,
                )

            if attempt < self.max_retries:
                _print(
                    "DEBUG",
                    "Sleeping for %.2fs before retrying download for %s (%s)",
                    self.retry_delay * attempt,
                    self.ticker,
                    self.interval,
                )
                time.sleep(self.retry_delay * attempt)

        assert last_error is not None  # for type checkers
        raise RuntimeError(
            "Failed to download market data after multiple attempts."
        ) from last_error

    def fetch(self, end: Optional[datetime] = None) -> pd.DataFrame:
        end = end or datetime.utcnow()
        start = end - timedelta(days=self.history_days)
        _print(
            "INFO",
            "Fetching data for %s (%s) from %s to %s with lookback %dd",
            self.ticker,
            self.interval,
            start,
            end,
            self.history_days,
        )
        download_kwargs = {
            "tickers": self.ticker,
            "interval": self.interval,
            "auto_adjust": False,
            "progress": False,
            "threads": False,
        }

        # Prefer the ``period`` argument for intraday data because yfinance may reject
        # large start/end ranges with a timeout. We keep ``start``/``end`` as a fallback
        # for daily intervals.
        if self.interval.endswith("m"):
            download_kwargs["period"] = self._resolve_period()
            _print(
                "DEBUG",
                "Using period=%s for intraday request of %s (%s)",
                download_kwargs["period"],
                self.ticker,
                self.interval,
            )
        else:
            download_kwargs.update({"start": start, "end": end})
            _print(
                "DEBUG",
                "Using explicit start/end for %s (%s)",
                self.ticker,
                self.interval,
            )

        data = self._download_with_retries(**download_kwargs)
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
        _print(
            "INFO",
            "Completed fetch for %s (%s) with %d rows",
            self.ticker,
            self.interval,
            len(data),
        )
        return data
