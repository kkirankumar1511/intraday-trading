"""Inference utilities for the trained LSTM model."""
from __future__ import annotations

from dataclasses import dataclass
import joblib
import numpy as np
import torch

from src.features.feature_engineering import add_technical_features
from src.models.lstm_model import BidirectionalLSTM
from src.pipeline.dataset import SequenceConfig


@dataclass
class PredictorConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMPredictor:
    def __init__(
        self,
        sequence_config: SequenceConfig,
        model_path,
        feature_scaler_path,
        target_scaler_path,
        predictor_config: PredictorConfig | None = None,
    ) -> None:
        self.sequence_config = sequence_config
        self.device = (predictor_config or PredictorConfig()).device

        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)

        input_size = self.feature_scaler.mean_.shape[0]
        self.model = BidirectionalLSTM(
            input_size=input_size,
            output_size=sequence_config.horizon,
        )
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "ema_20",
            "ema_50",
            "macd",
            "rsi",
            "return_change",
        ]

    def predict(self, raw_df) -> np.ndarray:
        df = add_technical_features(raw_df)
        if len(df) < self.sequence_config.lookback:
            raise ValueError(
                "Insufficient data for prediction. Increase history length or reduce lookback."
            )
        features = df[self.feature_columns]
        scaled_features = self.feature_scaler.transform(features.values)

        latest_features = scaled_features[-self.sequence_config.lookback :]
        input_tensor = torch.tensor(
            latest_features[np.newaxis, :, :], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            scaled_prediction = self.model(input_tensor).cpu().numpy()

        prediction = self.target_scaler.inverse_transform(scaled_prediction)
        return prediction.flatten()


__all__ = ["LSTMPredictor", "PredictorConfig"]
