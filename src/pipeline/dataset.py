"""Utilities for preparing time-series sequences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceConfig:
    lookback: int = 200
    horizon: int = 10


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if len(features) != len(targets):
            raise ValueError("Features and targets must have the same length")
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.features[idx], self.targets[idx]


def build_sequences(
    values: np.ndarray,
    target: np.ndarray,
    config: SequenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    lookback = config.lookback
    horizon = config.horizon
    features = []
    targets = []
    for end_idx in range(lookback, len(values) - horizon + 1):
        start_idx = end_idx - lookback
        feature_window = values[start_idx:end_idx]
        target_window = target[end_idx : end_idx + horizon]
        features.append(feature_window)
        targets.append(target_window)
    return np.array(features), np.array(targets)


__all__ = ["SequenceDataset", "SequenceConfig", "build_sequences"]
