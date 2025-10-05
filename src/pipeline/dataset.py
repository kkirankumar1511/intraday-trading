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
    stride: int = 1

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        if self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if len(features) != len(targets):
            raise ValueError("Features and targets must have the same length")
        features_tensor = torch.from_numpy(
            np.asarray(features, dtype=np.float32)
        ).contiguous()
        targets_tensor = torch.from_numpy(
            np.asarray(targets, dtype=np.float32)
        ).contiguous()
        self.features = features_tensor
        self.targets = targets_tensor

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
    stride = config.stride

    if values.ndim != 2:
        raise ValueError("values must be a 2D array of shape (timesteps, features)")
    if target.ndim != 1:
        raise ValueError("target must be a 1D array with the target series")
    if len(values) != len(target):
        raise ValueError("values and target must have the same number of timesteps")
    if len(values) < lookback + horizon:
        return np.empty((0, lookback, values.shape[1]), dtype=np.float32), np.empty(
            (0, horizon), dtype=np.float32
        )

    # Attempt to use the optimised numpy sliding window implementation. Fallback to
    # a Python loop if the current numpy version does not expose the helper.
    feature_windows: np.ndarray
    target_windows: np.ndarray
    try:
        from numpy.lib.stride_tricks import sliding_window_view

        feature_windows = sliding_window_view(values, (lookback, values.shape[1]))
        feature_windows = feature_windows[:, 0, :, :]
        target_windows = sliding_window_view(target, horizon)[lookback:]
    except Exception:  # pragma: no cover - fallback path for old numpy versions
        features_list = []
        targets_list = []
        for end_idx in range(lookback, len(values) - horizon + 1):
            start_idx = end_idx - lookback
            features_list.append(values[start_idx:end_idx])
            targets_list.append(target[end_idx : end_idx + horizon])
        feature_windows = np.stack(features_list) if features_list else np.empty(
            (0, lookback, values.shape[1])
        )
        target_windows = np.stack(targets_list) if targets_list else np.empty(
            (0, horizon)
        )

    total_windows = min(len(feature_windows), len(target_windows))
    feature_windows = feature_windows[:total_windows]
    target_windows = target_windows[:total_windows]

    if stride > 1 and total_windows:
        feature_windows = feature_windows[::stride]
        target_windows = target_windows[::stride]

    return (
        np.ascontiguousarray(feature_windows, dtype=np.float32),
        np.ascontiguousarray(target_windows, dtype=np.float32),
    )


__all__ = ["SequenceDataset", "SequenceConfig", "build_sequences"]
