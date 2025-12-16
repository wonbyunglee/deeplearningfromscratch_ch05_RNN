
"""
models/lstm.py (PyTorch)
- 단일 LSTM 회귀 모델 (many-to-one)
- 입력: 과거 L년치 feature 시퀀스 (B, L, F)
- 출력: 다음 시즌 ERA 예측 (B, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,          # F: 한 시점(한 시즌)에 들어가는 feature 개수
        hidden_size: int = 128,    # H: LSTM 은닉 상태 차원(기억 용량)
        num_layers: int = 1,       # LSTM 층 수(스택 LSTM)
        dropout: float = 0.3       # FC(head)에서 사용할 드롭아웃 비율
    ):
        super().__init__()

        # ✅ LSTM 본체
        # - 입력: (B, L, F)
        # - 출력(out): (B, L, H)  -> 각 timestep마다 생성된 hidden 시퀀스
        # - 출력(h_n): (num_layers, B, H) -> 각 layer의 "마지막 timestep" hidden
        # - 출력(c_n): (num_layers, B, H) -> 각 layer의 "마지막 timestep" cell state
        #
        # 참고: PyTorch LSTM의 dropout 옵션은 num_layers>1일 때만 layer 사이에 적용된다.
        # (num_layers=1이면 LSTM 내부 dropout은 사실상 꺼져 있음)
        self.lstm = nn.LSTM(
            input_size=input_size,     # 입력 feature 차원 F
            hidden_size=hidden_size,   # 은닉 차원 H
            num_layers=num_layers,     # LSTM 층 수
            batch_first=True,          # 입력을 (B, L, F) 형태로 받도록 설정
        )

        # ✅ 회귀(Regression) Head
        # - LSTM이 만든 마지막 timestep hidden 벡터 (B, H)를 받아서
        #   최종 ERA 스칼라 (B, 1)를 출력한다.
        #
        # 구성 의도:
        # - Dropout: 과적합 완화
        # - Linear(H -> 64) + ReLU: 비선형 변환으로 표현력 증가
        # - Linear(64 -> 1): 최종 스칼라 예측
        self.head = nn.Sequential(
            nn.Dropout(dropout),            # (B, H) 중 일부를 랜덤으로 0 처리
            nn.Linear(hidden_size, 64),     # (B, H) -> (B, 64)
            nn.ReLU(),                      # 비선형 활성화
            nn.Dropout(dropout),            # (B, 64) 드롭아웃
            nn.Linear(64, 1),               # (B, 64) -> (B, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        # - B: batch size
        # - L: sequence length (과거 몇 년치 기록을 보는지; 예: 2/3/4)
        # - F: feature dimension (한 시즌당 독립변수 개수)
        out, (h_n, c_n) = self.lstm(x)

        # out: (B, L, H)
        # - 각 timestep(각 연도)의 정보를 반영한 hidden 벡터들이 L개 쌓여 있음

        # ✅ many-to-one 회귀에서 흔히 쓰는 방식:
        # - 마지막 timestep의 hidden(out[:, -1, :])만 뽑아서
        #   "지금까지(L년) 정보를 다 읽고 난 요약 벡터"로 사용한다.
        #
        # last: (B, H)
        last = out[:, -1, :]

        # head를 통과시키면 최종 예측값(ERA) 스칼라가 나온다.
        # y: (B, 1)
        y = self.head(last)
        return y