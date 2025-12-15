
"""
models/lstm.py (PyTorch)
- 단일 LSTM 회귀 모델
""" 
from __future__ import annotations
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        # num_layers=1이면 PyTorch LSTM dropout은 적용되지 않으므로, 뒤쪽 FC에 dropout을 둔다.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        out, (h_n, c_n) = self.lstm(x)
        # 마지막 timestep의 hidden을 사용 (B, hidden)
        last = out[:, -1, :]
        y = self.head(last)
        return y
