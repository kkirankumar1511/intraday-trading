"""Training and evaluation pipeline for the LSTM model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.features.feature_engineering import add_technical_features
from src.models.lstm_model import BidirectionalLSTM
from src.pipeline.dataset import SequenceConfig, SequenceDataset, build_sequences


def _print(level: str, message: str, *args) -> None:
    formatted = message % args if args else message
    print(f"[{level}] {formatted}", flush=True)


@dataclass
class TrainerConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "auto"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    max_batches_per_epoch: Optional[int] = None
    early_stopping_patience: Optional[int] = 5
    improvement_threshold: float = 1e-4

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.max_batches_per_epoch is not None and self.max_batches_per_epoch <= 0:
            raise ValueError("max_batches_per_epoch must be positive when provided")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive when provided")
        if self.improvement_threshold < 0:
            raise ValueError("improvement_threshold must be non-negative")


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
        self.device = self._resolve_device(trainer_config.device)

    def _resolve_device(self, configured_device: str) -> str:
        """Return a valid device string, defaulting to CPU when necessary."""

        if configured_device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
            _print("INFO", "Auto-selected device: %s", resolved)
            return resolved

        try:
            device = torch.device(configured_device)
        except (RuntimeError, ValueError):
            _print(
                "WARN",
                "Requested device '%s' is not available. Falling back to CPU.",
                configured_device,
            )
            return "cpu"

        if device.type == "cuda" and not torch.cuda.is_available():
            _print(
                "WARN",
                "CUDA requested via '%s' but no GPU is available. Falling back to CPU.",
                configured_device,
            )
            return "cpu"

        return configured_device

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
        _print("INFO", "Starting training run")
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
        _print(
            "DEBUG",
            "Prepared sequences with shape %s and targets shape %s",
            sequences.shape,
            targets.shape,
        )
        train_dataset, test_dataset = self._split_dataset(sequences, targets)
        if len(train_dataset) == 0:
            raise ValueError(
                "Training dataset is empty after sequence generation. "
                "Provide more historical data or decrease the lookback window."
            )
        _print(
            "INFO",
            "Split dataset into %d training samples and %d evaluation samples",
            len(train_dataset),
            len(test_dataset),
        )
        train_loader, test_loader = self._create_dataloaders(train_dataset, test_dataset)

        model = BidirectionalLSTM(
            input_size=sequences.shape[-1],
            output_size=self.sequence_config.horizon,
            hidden_size=self.trainer_config.hidden_size,
            num_layers=self.trainer_config.num_layers,
            dropout=self.trainer_config.dropout,
        )

        try:
            model = model.to(self.device)
        except RuntimeError as exc:
            _print(
                "ERROR",
                "Unable to move model to requested device '%s'. Falling back to CPU. Error: %s",
                self.device,
                exc,
            )
            self.device = "cpu"
            model = model.to(self.device)
        else:
            _print("INFO", "Model initialized on device %s", self.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.trainer_config.learning_rate)

        total_batches = len(train_loader)
        configured_batch_cap = self.trainer_config.max_batches_per_epoch
        effective_batches = (
            total_batches
            if configured_batch_cap is None
            else min(total_batches, configured_batch_cap)
        )
        if configured_batch_cap is not None and configured_batch_cap < total_batches:
            _print(
                "INFO",
                "Capping each epoch at %d batches (of %d total)",
                effective_batches,
                total_batches,
            )
        log_interval = max(1, effective_batches // 5) if effective_batches else 1

        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.trainer_config.epochs):
            model.train()
            running_loss = 0.0
            processed_samples = 0
            processed_batches = 0
            _print(
                "DEBUG",
                "Epoch %d/%d - starting (%d batches)",
                epoch + 1,
                self.trainer_config.epochs,
                effective_batches,
            )
            for batch_idx, (batch_features, batch_targets) in enumerate(
                train_loader, start=1
            ):
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_features.size(0)
                processed_samples += batch_features.size(0)
                processed_batches += 1

                if processed_batches % log_interval == 0 or processed_batches == effective_batches:
                    _print(
                        "DEBUG",
                        "Epoch %d/%d - processed %d/%d batches",
                        epoch + 1,
                        self.trainer_config.epochs,
                        processed_batches,
                        effective_batches,
                    )

                if (
                    configured_batch_cap is not None
                    and processed_batches >= effective_batches
                ):
                    if total_batches > effective_batches:
                        _print(
                            "DEBUG",
                            "Epoch %d/%d - reached configured batch limit (%d/%d)",
                            epoch + 1,
                            self.trainer_config.epochs,
                            processed_batches,
                            total_batches,
                        )
                    break

            if processed_samples == 0:
                _print(
                    "WARN",
                    "Epoch %d/%d - no batches processed; stopping training loop",
                    epoch + 1,
                    self.trainer_config.epochs,
                )
                break

            epoch_loss = running_loss / processed_samples
            _print(
                "INFO",
                "Epoch %d/%d - Loss: %.4f",
                epoch + 1,
                self.trainer_config.epochs,
                epoch_loss,
            )

            if epoch_loss + self.trainer_config.improvement_threshold < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    self.trainer_config.early_stopping_patience is not None
                    and epochs_without_improvement
                    >= self.trainer_config.early_stopping_patience
                ):
                    _print(
                        "INFO",
                        "Early stopping triggered after %d epochs without improvement",
                        epochs_without_improvement,
                    )
                    break

        metrics = self.evaluate(model, test_loader, target_scaler)
        _print("INFO", "Evaluation metrics: %s", metrics)

        # Persist artifacts
        self.artifact_paths.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.artifact_paths.model_path)
        joblib.dump(feature_scaler, self.artifact_paths.feature_scaler_path)
        joblib.dump(target_scaler, self.artifact_paths.target_scaler_path)
        _print("INFO", "Saved model and scalers to disk")

        return metrics

    def evaluate(
        self, model: BidirectionalLSTM, data_loader: DataLoader, target_scaler: StandardScaler
    ) -> Dict[str, float]:
        if len(data_loader.dataset) == 0:
            _print("WARN", "Test dataset is empty; returning NaN metrics")
            return {
                "mse": float("nan"),
                "mae": float("nan"),
                "mape": float("nan"),
                "directional_accuracy": float("nan"),
            }

        model.eval()
        predictions: list[np.ndarray] = []
        actuals: list[np.ndarray] = []
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(data_loader, start=1):
                features = features.to(self.device)
                outputs = model(features)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.numpy())
                _print(
                    "DEBUG",
                    "Processed evaluation batch %d/%d",
                    batch_idx,
                    len(data_loader),
                )

        pred_array = np.concatenate(predictions, axis=0)
        actual_array = np.concatenate(actuals, axis=0)

        # The target scaler is fit on a single-column array, so we need to reshape
        # the multi-step predictions and actuals before applying the inverse
        # transformation.
        pred_rescaled = target_scaler.inverse_transform(pred_array.reshape(-1, 1)).reshape(
            pred_array.shape
        )
        actual_rescaled = target_scaler.inverse_transform(
            actual_array.reshape(-1, 1)
        ).reshape(actual_array.shape)

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
