"""
models/bilstm.py (PyTorch)
- BiLSTM 회귀 모델 (many-to-one)
- 입력: 과거 L년치 feature 시퀀스 (B, L, F)
- 출력: 다음 시즌 ERA 예측 (B, 1)

BiLSTM 개념:
- 양방향 RNN(Schuster & Paliwal, 1997) + LSTM 셀(Hochreiter & Schmidhuber, 1997)
- 정방향(과거 → 현재)과 역방향(미래 → 현재) 정보를 모두 활용
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,          # F: 한 시점(한 시즌)에 들어가는 feature 개수
        hidden_size: int = 128,    # H: 단방향 LSTM 은닉 상태 차원
        num_layers: int = 1,       # LSTM 층 수(스택 LSTM)
        dropout: float = 0.3       # FC(head)에서 사용할 드롭아웃 비율
    ):
        super().__init__()

        # ✅ BiLSTM 본체
        # - 입력: (B, L, F)
        # - 출력(out): (B, L, 2H)
        #   → forward hidden(H) + backward hidden(H)를 concat
        # - 출력(h_n): (num_layers*2, B, H)
        #   → 각 layer의 forward / backward 마지막 timestep hidden
        # - 출력(c_n): (num_layers*2, B, H)
        #
        # bidirectional=True 설정으로:
        # - 정방향 LSTM: t = 1 → L
        # - 역방향 LSTM: t = L → 1
        # 두 개가 동시에 시퀀스를 처리한다.
        self.bilstm = nn.LSTM(
            input_size=input_size,     # 입력 feature 차원 F
            hidden_size=hidden_size,   # 단방향 은닉 차원 H
            num_layers=num_layers,     # LSTM 층 수
            batch_first=True,          # 입력을 (B, L, F) 형태로 받음
            bidirectional=True,        # 🔑 양방향 LSTM
        )

        # ✅ 회귀(Regression) Head
        # - BiLSTM의 출력 hidden 차원은 2H이므로
        #   Linear 입력 차원도 hidden_size * 2
        #
        # 구성 의도:
        # - Dropout: 과적합 방지
        # - Linear(2H -> 64) + ReLU: 비선형 변환
        # - Linear(64 -> 1): 최종 ERA 예측
        self.head = nn.Sequential(
            nn.Dropout(dropout),                # (B, 2H) 드롭아웃
            nn.Linear(hidden_size * 2, 64),     # (B, 2H) -> (B, 64)
            nn.ReLU(),                          # 비선형 활성화
            nn.Dropout(dropout),                # (B, 64) 드롭아웃
            nn.Linear(64, 1),                   # (B, 64) -> (B, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        # - B: batch size
        # - L: sequence length (과거 몇 년치 기록; 예: 2/3/4)
        # - F: feature dimension (한 시즌당 독립변수 개수)
        out, (h_n, c_n) = self.bilstm(x)

        # out: (B, L, 2H)
        # - 각 timestep마다
        #   [정방향 hidden | 역방향 hidden] 이 concat된 결과
        #
        # h_n: (num_layers*2, B, H)
        # - 마지막 layer 기준:
        #   h_n[-2]: 정방향 LSTM의 마지막 timestep hidden
        #   h_n[-1]: 역방향 LSTM의 마지막 timestep hidden
        #
        # ⚠️ 여기서는 many-to-one 회귀를 위해
        #     out[:, -1, :]을 사용한다.
        #
        # out[:, -1, :]은:
        # - 정방향: 시퀀스 끝(L)까지 읽은 요약
        # - 역방향: 시퀀스 끝 위치에서의 backward hidden
        #
        # (실전에서 자주 쓰이지만, 해석은 h_n 기반 방식보다 덜 직관적일 수 있음)

        # last: (B, 2H)
        last = out[:, -1, :]

        # 회귀 head를 통과시켜 최종 ERA 예측
        # y: (B, 1)
        y = self.head(last)
        return y

