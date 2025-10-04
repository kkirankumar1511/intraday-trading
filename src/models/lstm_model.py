"""Neural network models for time-series forecasting."""
from __future__ import annotations

import torch
from torch import nn


class BidirectionalLSTM(nn.Module):
    """Three-layer bidirectional LSTM for multi-step forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_size: int = 10,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        lstm_out, _ = self.lstm(x)
        # Use last time step from both directions
        last_timestep = lstm_out[:, -1, :]
        out = self.fc(last_timestep)
        return out


__all__ = ["BidirectionalLSTM"]
