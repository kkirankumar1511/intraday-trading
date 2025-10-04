from __future__ import annotations

import argparse

from intraday_trading.config import TrainingConfig
from intraday_trading.pipeline import IntradayPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intraday 15-minute price prediction pipeline")
    parser.add_argument("command", choices=["train", "predict"], help="Action to perform")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to use")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(ticker=args.ticker, epochs=args.epochs)
    pipeline = IntradayPipeline(config)

    if args.command == "train":
        metrics = pipeline.run_daily_training()
        print("Training complete. Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    elif args.command == "predict":
        predictions = pipeline.run_realtime_prediction()
        print("Next 10 interval predictions:")
        print(predictions)


if __name__ == "__main__":
    main()
