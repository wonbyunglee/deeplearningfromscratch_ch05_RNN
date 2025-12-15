
"""
data_preprocess.py
- basic.csv 로드
- period(=예측 타깃 연도)와 sequence_length(L)에 맞춰 (과거 L년 -> 타깃 연도 ERA) 형태의 시계열 샘플 생성
- 스케일링(입력/타깃 각각 MinMax)
- train/val 분할용 유틸 포함

실습 포인트
- "선수별로" 시계열을 만들기 때문에, 서로 다른 선수의 기록이 섞여 들어가는 문제를 방지한다.
- period=2019이면 2015~2019, period=2023이면 2019~2023 구간에서 샘플을 만든다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


@dataclass
class DatasetBundle:
    """학습/평가에 필요한 데이터 묶음"""
    X: np.ndarray  # (N, L, F)
    y: np.ndarray  # (N, 1)
    player_ids: np.ndarray  # (N,)
    feature_cols: List[str]
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler


def load_basic_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 논문 코드에서 drop 했던 컬럼 (이름 컬럼은 모델 입력에 불필요)
    if 'last_name, first_name' in df.columns:
        df = df.drop(columns=['last_name, first_name'])

    # 결측치는 0으로 채움
    df = df.fillna(0)

    # 정렬: 선수별/연도별 순서를 확실히 맞춰준다
    df = df.sort_values(['player_id', 'year']).reset_index(drop=True)
    return df


def infer_feature_cols(df: pd.DataFrame, target_col: str = "p_era") -> List[str]:
    # player_id/year/target 제외
    return [c for c in df.columns if c not in ["player_id", "year", target_col]]


def _players_with_all_years(df: pd.DataFrame, years: List[int]) -> np.ndarray:
    """years 목록의 모든 연도를 다 가진 선수만 남긴다."""
    players = None
    for y in years:
        ids = set(df.loc[df["year"] == y, "player_id"].unique())
        players = ids if players is None else (players & ids)
    return np.array(sorted(players), dtype=df["player_id"].dtype)


def build_playerwise_dataset(
    df: pd.DataFrame,
    target_year: int,
    seq_length: int,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "p_era",
) -> DatasetBundle:
    """
    (과거 L년 feature 시퀀스) -> (target_year의 ERA) 샘플 생성.

    - 입력 시퀀스 연도: target_year-seq_length ... target_year-1 (총 L개)
    - 타깃 연도: target_year
    - 선수는 (입력 연도들 + 타깃 연도) 모두 존재해야 샘플 생성 가능
    """
    if feature_cols is None:
        feature_cols = infer_feature_cols(df, target_col=target_col)

    hist_years = list(range(target_year - seq_length, target_year))
    required_years = hist_years + [target_year]

    # 필요한 연도가 전부 있는 선수만 추림 (실습에서는 이게 제일 깔끔함)
    common_players = _players_with_all_years(df, required_years)

    # 선수별로 시퀀스 생성
    X_list, y_list, pid_list = [], [], []
    for pid in common_players:
        sub = df[df["player_id"] == pid]

        # 입력: 과거 L년
        x_seq = []
        for y in hist_years:
            row = sub[sub["year"] == y]
            # 안전장치: 혹시라도 중복/누락이 있으면 스킵
            if len(row) != 1:
                x_seq = None
                break
            x_seq.append(row[feature_cols].to_numpy(dtype=np.float32)[0])
        if x_seq is None:
            continue

        # 타깃: target_year ERA
        row_t = sub[sub["year"] == target_year]
        if len(row_t) != 1:
            continue
        y_val = float(row_t[target_col].values[0])

        X_list.append(np.stack(x_seq, axis=0))  # (L, F)
        y_list.append([y_val])                  # (1,)
        pid_list.append(pid)

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, L, F)
    y = np.array(y_list, dtype=np.float32)           # (N, 1)
    player_ids = np.array(pid_list)

    # ===== 스케일링 =====
    # 입력은 (N*L, F)로 펴서 스케일링 후 다시 reshape
    x_scaler = MinMaxScaler()
    X2 = X.reshape(-1, X.shape[-1])
    X_scaled = x_scaler.fit_transform(X2).reshape(X.shape).astype(np.float32)

    # 타깃 스케일러는 y만 기준으로 fit
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y).astype(np.float32)

    return DatasetBundle(
        X=X_scaled,
        y=y_scaled,
        player_ids=player_ids,
        feature_cols=feature_cols,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


def split_train_val( 
    bundle: DatasetBundle,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    선수 단위 샘플을 랜덤 분할.
    - 실습에서는 간단하게 hold-out으로 충분
    """
    X_train, X_val, y_train, y_val = train_test_split(
        bundle.X, bundle.y, test_size=val_ratio, random_state=seed, shuffle=True
    )
    return X_train, X_val, y_train, y_val
