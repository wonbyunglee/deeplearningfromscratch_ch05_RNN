"""
models/gru.py (PyTorch)
- 단일 GRU 회귀 모델 (many-to-one)
- 입력: 과거 L년치 feature 시퀀스 (B, L, F)
- 출력: 다음 시즌 ERA 예측 (B, 1)

GRU 원논문(Cho et al., 2014)의 핵심:
- reset gate, update gate를 사용해 hidden state 하나만으로 시계열 정보를 유지
- LSTM보다 구조가 단순하며, cell state(c_t)를 따로 두지 않는다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_size: int,          # F: 한 시점(한 시즌)에 들어가는 feature 개수
        hidden_size: int = 128,    # H: GRU 은닉 상태 차원
        num_layers: int = 1,       # GRU 층 수(스택 GRU)
        dropout: float = 0.3       # FC(head)에서 사용할 드롭아웃 비율
    ):
        super().__init__()

        # ✅ GRU 본체
        # - 입력: (B, L, F)
        # - 출력(out): (B, L, H)  -> 각 timestep마다의 hidden state
        # - 출력(h_n): (num_layers, B, H) -> 각 layer의 마지막 timestep hidden
        #
        # GRU는 LSTM과 달리:
        # - cell state(c_t)를 사용하지 않음
        # - reset gate / update gate만으로 hidden state를 갱신
        #
        # 참고: PyTorch GRU의 dropout 옵션은 num_layers>1일 때만
        # layer 사이에 적용된다. (num_layers=1이면 내부 dropout 없음)
        self.gru = nn.GRU(
            input_size=input_size,     # 입력 feature 차원 F
            hidden_size=hidden_size,   # 은닉 차원 H
            num_layers=num_layers,     # GRU 층 수
            batch_first=True,          # 입력을 (B, L, F) 형태로 받도록 설정
        )

        # ✅ 회귀(Regression) Head
        # - GRU가 만든 마지막 timestep hidden 벡터 (B, H)를 받아
        #   최종 ERA 스칼라 값 (B, 1)을 출력
        #
        # 구성 의도:
        # - Dropout: 과적합 방지
        # - Linear(H -> 64) + ReLU: 비선형 변환으로 표현력 증가
        # - Linear(64 -> 1): 최종 회귀 출력
        self.head = nn.Sequential(
            nn.Dropout(dropout),            # (B, H) 일부를 랜덤으로 0 처리
            nn.Linear(hidden_size, 64),     # (B, H) -> (B, 64)
            nn.ReLU(),                      # 비선형 활성화
            nn.Dropout(dropout),            # (B, 64) 드롭아웃
            nn.Linear(64, 1),               # (B, 64) -> (B, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        # - B: batch size
        # - L: sequence length (과거 몇 년치 기록; 예: 2/3/4)
        # - F: feature dimension (한 시즌당 독립변수 개수)
        out, h_n = self.gru(x)

        # out: (B, L, H)
        # - 각 timestep(각 연도)의 정보를 반영한 hidden state 시퀀스
        #
        # h_n: (num_layers, B, H)
        # - 각 layer의 마지막 timestep hidden state
        # - 단방향 GRU + 고정 길이 시퀀스라면
        #   out[:, -1, :]과 h_n[-1]은 거의 동일한 의미를 가짐

        # ✅ many-to-one 회귀 방식
        # - 마지막 timestep의 hidden을 사용해
        #   "과거 L년 기록을 모두 반영한 요약 벡터"로 간주
        #
        # last: (B, H)
        last = out[:, -1, :]

        # 회귀 head를 통과시켜 ERA 예측값 출력
        # y: (B, 1)
        y = self.head(last)
        return y
