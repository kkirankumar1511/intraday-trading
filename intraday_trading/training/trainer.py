from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from intraday_trading.features.engineering import FeatureEngineer
from intraday_trading.models.lstm_model import BiLSTMPricePredictor


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


@dataclass
class TrainingArtifacts:
    model: BiLSTMPricePredictor
    scaler: StandardScaler
    metrics: Dict[str, float]


class PricePredictionTrainer:
    def __init__(
        self,
        config,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_engineer = FeatureEngineer(lookback=config.lookback)

    def _prepare_sequences(self, df: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return df, targets

    def _build_sequences(self, df) -> Tuple[np.ndarray, np.ndarray]:
        feature_cols = self.feature_engineer.feature_columns
        values = df[feature_cols].values
        closes = df["close"].values
        lookback = self.config.lookback
        horizon = self.config.horizon

        sequences = []
        target_sequences = []
        for end_idx in range(lookback, len(df) - horizon + 1):
            start_idx = end_idx - lookback
            x = values[start_idx:end_idx]
            y = closes[end_idx:end_idx + horizon]
            if len(x) == lookback and len(y) == horizon:
                sequences.append(x)
                target_sequences.append(y)
        sequences = np.stack(sequences)
        target_sequences = np.stack(target_sequences)
        return sequences, target_sequences

    def _scale_sequences(self, sequences: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        n_samples, seq_len, n_features = sequences.shape
        scaler = StandardScaler()
        reshaped = sequences.reshape(-1, n_features)
        scaled = scaler.fit_transform(reshaped)
        sequences = scaled.reshape(n_samples, seq_len, n_features)
        return sequences, scaler

    def train(self, df) -> TrainingArtifacts:
        df = self.feature_engineer.transform(df)
        sequences, targets = self._build_sequences(df)
        if len(sequences) == 0:
            raise ValueError(
                "Not enough samples to build training sequences. Increase history_days or reduce lookback."
            )
        sequences, scaler = self._scale_sequences(sequences)

        X_train, X_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=self.config.test_size, shuffle=False
        )

        train_dataset = SequenceDataset(X_train, y_train)
        test_dataset = SequenceDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

        model = BiLSTMPricePredictor(
            input_size=sequences.shape[-1],
            hidden_size=128,
            num_layers=3,
            horizon=self.config.horizon,
            dropout=0.2,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            for features, target in train_loader:
                features = features.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = model(features)
                loss = criterion(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item() * len(features)
            epoch_loss /= len(train_loader.dataset)

        metrics = self.evaluate(model, test_loader)

        return TrainingArtifacts(model=model, scaler=scaler, metrics=metrics)

    @torch.no_grad()
    def evaluate(self, model: BiLSTMPricePredictor, data_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        all_preds = []
        all_targets = []
        for features, target in data_loader:
            features = features.to(self.device)
            preds = model(features).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(target.numpy())

        if not all_preds:
            return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "directional_accuracy": float("nan")}

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        mse = mean_squared_error(targets, preds)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(targets, preds)
        mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
        direction_accuracy = self._directional_accuracy(targets, preds)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "directional_accuracy": direction_accuracy,
        }

    @staticmethod
    def _directional_accuracy(targets: np.ndarray, preds: np.ndarray) -> float:
        target_returns = np.diff(targets, axis=1)
        pred_returns = np.diff(preds, axis=1)
        target_sign = np.sign(target_returns)
        pred_sign = np.sign(pred_returns)
        matches = target_sign == pred_sign
        return float(matches.mean()) * 100

    def save(self, artifacts: TrainingArtifacts) -> None:
        self.config.ensure_directories()
        torch.save(artifacts.model.state_dict(), self.config.model_path)
        with open(self.config.scaler_path, "wb") as f:
            import pickle

            pickle.dump(artifacts.scaler, f)

    def load_model(self) -> Tuple[BiLSTMPricePredictor, StandardScaler]:
        scaler: StandardScaler
        if not self.config.scaler_path.exists() or not self.config.model_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found. Please run daily training before requesting predictions."
            )

        with open(self.config.scaler_path, "rb") as f:
            import pickle

            scaler = pickle.load(f)

        sample_input_size = len(self.feature_engineer.feature_columns)
        model = BiLSTMPricePredictor(
            input_size=sample_input_size,
            hidden_size=128,
            num_layers=3,
            horizon=self.config.horizon,
            dropout=0.2,
        )
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model, scaler

    @torch.no_grad()
    def predict_next(self, model: BiLSTMPricePredictor, scaler: StandardScaler, df) -> np.ndarray:
        df = self.feature_engineer.transform(df)
        feature_cols = self.feature_engineer.feature_columns
        values = df[feature_cols].values
        if len(values) < self.config.lookback:
            raise ValueError("Insufficient data for prediction")
        latest = values[-self.config.lookback :]
        scaled = scaler.transform(latest)
        tensor = torch.from_numpy(scaled).float().unsqueeze(0).to(self.device)
        preds = model(tensor).cpu().numpy().reshape(-1)
        return preds
