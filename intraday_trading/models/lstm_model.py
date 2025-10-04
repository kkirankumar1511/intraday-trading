from __future__ import annotations

import torch
from torch import nn


class BiLSTMPricePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, horizon: int = 10, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size * 2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.output(out)
        return out
