# coding: utf-8
"""
models/simple_rnn_book.py

책(SimpleRnnlm)의 구조를 최대한 유지한 "회귀용" Simple RNN 모델 (PyTorch 버전)

- 책 코드: TimeEmbedding -> TimeRNN -> TimeAffine -> TimeSoftmaxWithLoss (분류/언어모델)
- 여기:     (실수 feature 입력) -> RNN -> Linear -> (회귀 loss는 train_eval에서 MSE로)

입력/출력
- 입력 X: (B, L, F)  # B=batch, L=sequence_length, F=num_features
- 출력 y_hat: (B, 1) # ERA 예측

실습 포인트
- 마지막 시점의 hidden state(h_last)를 뽑아서 Linear로 1차원 회귀
- stateful=True이면 batch 간 hidden state를 유지 (책의 stateful 느낌 재현)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRNNBookRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.0,
        stateful: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stateful = stateful

        # 책의 "RNN(Wx, Wh, b)" 역할
        # batch_first=True => (B, L, F) 입력 그대로 사용
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            batch_first=True,
        )

        # 책의 "TimeAffine" 역할 (hidden -> output)
        self.fc = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # stateful 모드에서 hidden state를 저장해두기 위한 변수
        self._h: torch.Tensor | None = None

        # (선택) 책 느낌의 초기화: 너무 큰 값 방지
        self._init_weights_like_book()

    def _init_weights_like_book(self):
        """
        책 코드처럼 너무 큰 값으로 시작하지 않게끔 간단 초기화.
        - 완전히 동일하진 않지만 '작게 시작'한다는 의도는 동일.
        """
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)

    def reset_state(self):
        """책 코드의 reset_state() 대응"""
        self._h = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, F)
        return: (B, 1)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B, L, F), got {tuple(x.shape)}")

        # stateful이면 이전 hidden을 다음 batch에 이어서 사용
        if self.stateful and self._h is not None:
            out, h_n = self.rnn(x, self._h)
        else:
            out, h_n = self.rnn(x)  # out: (B, L, H), h_n: (1, B, H)

        # 마지막 시점 hidden 사용 (책에서 마지막 단어/시점 예측하는 느낌)
        h_last = out[:, -1, :]      # (B, H)
        h_last = self.dropout(h_last)

        y_hat = self.fc(h_last)     # (B, 1)

        # 다음 batch를 위해 hidden 저장 (stateful일 때만)
        if self.stateful:
            # h_n shape: (1, B, H) -> detach해서 그래프 끊기
            self._h = h_n.detach()

        return y_hat
