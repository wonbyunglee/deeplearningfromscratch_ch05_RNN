
"""
models/bilstm.py (PyTorch)
- BiLSTM 회귀 모델
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # bidirectional이면 hidden이 2배가 된다
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.bilstm(x)
        last = out[:, -1, :]  # (B, hidden*2)
        return self.head(last)
