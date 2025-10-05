"""Inference utilities for the trained LSTM model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, cast

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

        state_dict, metadata = self._load_checkpoint(model_path)
        trained_horizon = metadata.get("output_size", sequence_config.horizon)
        if trained_horizon != sequence_config.horizon:
            raise ValueError(
                "Loaded model was trained for horizon %d but predictor configured for horizon %d."
                % (trained_horizon, sequence_config.horizon)
            )

        input_size = metadata.get("input_size", self.feature_scaler.mean_.shape[0])
        hidden_size = metadata.get("hidden_size", 128)
        num_layers = metadata.get("num_layers", 3)
        dropout = metadata.get("dropout", 0.2)

        self.model = BidirectionalLSTM(
            input_size=input_size,
            output_size=trained_horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
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

    def _load_checkpoint(self, model_path) -> Tuple[Dict[str, torch.Tensor], Dict[str, int | float]]:
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            metadata = cast(Dict[str, int | float], checkpoint.get("metadata", {}))
            state_dict = cast(Dict[str, torch.Tensor], checkpoint["model_state_dict"])
            return state_dict, metadata

        state_dict = checkpoint
        inferred_metadata: Dict[str, int | float] = {}

        weight_ih_keys = [
            key
            for key in state_dict
            if key.startswith("lstm.weight_ih_l") and "reverse" not in key
        ]
        if weight_ih_keys:
            inferred_metadata["num_layers"] = len(weight_ih_keys)
            first_weight = state_dict[weight_ih_keys[0]]
            inferred_metadata["hidden_size"] = first_weight.shape[0] // 4
            inferred_metadata["input_size"] = first_weight.shape[1]

        fc_weight = state_dict.get("fc.3.weight")
        if fc_weight is not None:
            inferred_metadata["output_size"] = fc_weight.shape[0]

        return state_dict, inferred_metadata


__all__ = ["LSTMPredictor", "PredictorConfig"]
