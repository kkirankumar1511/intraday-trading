from __future__ import annotations

from intraday_trading.config import TrainingConfig
from intraday_trading.pipeline import IntradayPipeline


def main() -> None:
    config = TrainingConfig()
    pipeline = IntradayPipeline(config)
    metrics = pipeline.run_daily_training()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
