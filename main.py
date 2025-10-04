"""IntelliJ-friendly entry point for the LSTM intraday predictor workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.data_loader import YFinanceConfig, YFinanceLoader
from src.pipeline.dataset import SequenceConfig
from src.pipeline.predictor import LSTMPredictor
from src.pipeline.trainer import ArtifactPaths, LSTMTrainer, TrainerConfig

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "bidirectional_lstm.pth"
FEATURE_SCALER_PATH = ARTIFACT_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = ARTIFACT_DIR / "target_scaler.pkl"
TRAINING_STATE_PATH = ARTIFACT_DIR / "training_state.json"


@dataclass
class WorkflowConfig:
    """Configuration block consumed by the IntelliJ run configuration."""

    ticker: str = "AAPL"
    interval: str = "15m"
    lookback_days: int = 60
    lookback_window: int = 200
    horizon: int = 10
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    force_retrain: bool = False


class IntradayWorkflow:
    """Coordinates daily training and rolling inference for IDE usage."""

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    def run_daily_training(self) -> Optional[Dict[str, float]]:
        """Train the LSTM if the saved weights are missing or stale."""

        if not self.config.force_retrain and not self._is_training_needed():
            return None

        loader = YFinanceLoader(
            YFinanceConfig(
                ticker=self.config.ticker,
                interval=self.config.interval,
                lookback_days=self.config.lookback_days,
            )
        )
        data = loader.load()
        sequence_config = SequenceConfig(
            lookback=self.config.lookback_window, horizon=self.config.horizon
        )
        trainer_config = TrainerConfig(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )
        artifact_paths = ArtifactPaths(MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH)

        trainer = LSTMTrainer(sequence_config, trainer_config, artifact_paths)
        metrics = trainer.train(data)
        self._write_training_state()
        return metrics

    def generate_predictions(self) -> pd.DataFrame:
        """Load the latest data and produce the next horizon of close prices."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model weights not found. Run run_daily_training() before requesting predictions."
            )

        loader = YFinanceLoader(
            YFinanceConfig(
                ticker=self.config.ticker,
                interval=self.config.interval,
                lookback_days=self.config.lookback_days,
            )
        )
        data = loader.load()
        sequence_config = SequenceConfig(
            lookback=self.config.lookback_window, horizon=self.config.horizon
        )
        predictor = LSTMPredictor(
            sequence_config,
            MODEL_PATH,
            FEATURE_SCALER_PATH,
            TARGET_SCALER_PATH,
        )
        predictions = predictor.predict(data)
        prediction_index = pd.date_range(
            start=data.index[-1], periods=self.config.horizon + 1, freq=self.config.interval
        )[1:]
        return pd.DataFrame({"predicted_close": predictions}, index=prediction_index)

    def _is_training_needed(self) -> bool:
        if not MODEL_PATH.exists():
            return True
        state = self._read_training_state()
        if state is None:
            return True
        last_trained_str = state.get("last_trained")
        if not last_trained_str:
            return True
        last_trained = datetime.fromisoformat(last_trained_str)
        now = datetime.now(timezone.utc)
        return last_trained.date() < now.date()

    def _read_training_state(self) -> Optional[Dict[str, str]]:
        if not TRAINING_STATE_PATH.exists():
            return None
        try:
            return json.loads(TRAINING_STATE_PATH.read_text())
        except json.JSONDecodeError:
            return None

    def _write_training_state(self) -> None:
        state = {"last_trained": datetime.now(timezone.utc).isoformat()}
        TRAINING_STATE_PATH.write_text(json.dumps(state, indent=2))


def print_metrics(metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        if key == "directional_accuracy":
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.6f}")


def run() -> None:
    """Entry point used by IntelliJ's Run Configuration."""

    workflow = IntradayWorkflow(WorkflowConfig())
    metrics = workflow.run_daily_training()
    if metrics:
        print("Training completed. Evaluation metrics on hold-out set:")
        print_metrics(metrics)
    predictions = workflow.generate_predictions()
    print("Next interval predictions:")
    print(predictions)


if __name__ == "__main__":
    run()
