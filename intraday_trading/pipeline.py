from __future__ import annotations

from typing import Dict

import pandas as pd

from intraday_trading.config import TrainingConfig
from intraday_trading.data.fetcher import MarketDataFetcher
from intraday_trading.training.trainer import PricePredictionTrainer


class IntradayPipeline:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.fetcher = MarketDataFetcher(
            ticker=config.ticker, interval=config.interval, history_days=config.history_days
        )
        self.trainer = PricePredictionTrainer(config)

    def run_daily_training(self) -> Dict[str, float]:
        data = self.fetcher.fetch()
        artifacts = self.trainer.train(data)
        self.trainer.save(artifacts)
        return artifacts.metrics

    def predict_next_intervals(self, data: pd.DataFrame | None = None) -> pd.Series:
        if data is None:
            data = self.fetcher.fetch()
        model, scaler = self.trainer.load_model()
        preds = self.trainer.predict_next(model, scaler, data)
        prediction_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(self.config.interval),
            periods=self.config.horizon,
            freq=self.config.interval,
        )
        return pd.Series(preds, index=prediction_index, name="predicted_close")

    def run_realtime_prediction(self) -> pd.Series:
        return self.predict_next_intervals()
