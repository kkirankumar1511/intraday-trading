"""Training and evaluation pipeline for the LSTM model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.features.feature_engineering import add_technical_features
from src.models.lstm_model import BidirectionalLSTM
from src.pipeline.dataset import SequenceConfig, SequenceDataset, build_sequences


@dataclass
class TrainerConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ArtifactPaths:
    model_path: Path
    feature_scaler_path: Path
    target_scaler_path: Path


class LSTMTrainer:
    def __init__(
        self,
        sequence_config: SequenceConfig,
        trainer_config: TrainerConfig,
        artifact_paths: ArtifactPaths,
    ) -> None:
        self.sequence_config = sequence_config
        self.trainer_config = trainer_config
        self.artifact_paths = artifact_paths

    def _prepare_sequences(
        self, data: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return build_sequences(data, target, self.sequence_config)

    def _split_dataset(
        self, features: np.ndarray, targets: np.ndarray, split_ratio: float = 0.8
    ) -> Tuple[SequenceDataset, SequenceDataset]:
        split_index = int(len(features) * split_ratio)
        train_dataset = SequenceDataset(features[:split_index], targets[:split_index])
        test_dataset = SequenceDataset(features[split_index:], targets[split_index:])
        return train_dataset, test_dataset

    def _create_dataloaders(
        self, train_dataset: SequenceDataset, test_dataset: SequenceDataset
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, test_loader

    def train(self, raw_df) -> Dict[str, float]:
        df = add_technical_features(raw_df)
        feature_columns = [
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
        features = df[feature_columns]
        target = df[["close"]]

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        scaled_features = feature_scaler.fit_transform(features.values)
        scaled_target = target_scaler.fit_transform(target.values).squeeze()

        sequences, targets = self._prepare_sequences(scaled_features, scaled_target)
        train_dataset, test_dataset = self._split_dataset(sequences, targets)
        train_loader, test_loader = self._create_dataloaders(train_dataset, test_dataset)

        model = BidirectionalLSTM(
            input_size=sequences.shape[-1],
            output_size=self.sequence_config.horizon,
        ).to(self.trainer_config.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.trainer_config.learning_rate)

        for epoch in range(self.trainer_config.epochs):
            model.train()
            running_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.trainer_config.device)
                batch_targets = batch_targets.to(self.trainer_config.device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_features.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{self.trainer_config.epochs} - Loss: {epoch_loss:.4f}")

        metrics = self.evaluate(model, test_loader, target_scaler)

        # Persist artifacts
        self.artifact_paths.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.artifact_paths.model_path)
        joblib.dump(feature_scaler, self.artifact_paths.feature_scaler_path)
        joblib.dump(target_scaler, self.artifact_paths.target_scaler_path)

        return metrics

    def evaluate(
        self, model: BidirectionalLSTM, data_loader: DataLoader, target_scaler: StandardScaler
    ) -> Dict[str, float]:
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.trainer_config.device)
                outputs = model(features)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.numpy())

        pred_array = np.concatenate(predictions, axis=0)
        actual_array = np.concatenate(actuals, axis=0)

        pred_rescaled = target_scaler.inverse_transform(pred_array)
        actual_rescaled = target_scaler.inverse_transform(actual_array)

        mse = mean_squared_error(actual_rescaled, pred_rescaled)
        mae = mean_absolute_error(actual_rescaled, pred_rescaled)
        mape = (
            np.mean(np.abs((actual_rescaled - pred_rescaled) / (actual_rescaled + 1e-9)))
            * 100
        )

        actual_diff = np.diff(actual_rescaled, axis=1)
        pred_diff = np.diff(pred_rescaled, axis=1)
        if actual_diff.size == 0:
            directional_accuracy = float("nan")
        else:
            directional_accuracy = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100

        return {
            "mse": float(mse),
            "mae": float(mae),
            "mape": float(mape),
            "directional_accuracy": float(directional_accuracy),
        }


__all__ = ["LSTMTrainer", "TrainerConfig", "ArtifactPaths"]
