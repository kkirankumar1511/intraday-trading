from __future__ import annotations

from intraday_trading.config import TrainingConfig
from intraday_trading.pipeline import IntradayPipeline


def main() -> None:
    config = TrainingConfig()
    pipeline = IntradayPipeline(config)
    predictions = pipeline.run_realtime_prediction()
    print(predictions)


if __name__ == "__main__":
    main()
