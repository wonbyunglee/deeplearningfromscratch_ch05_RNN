"""
inspect_parsing.py
- period(target_year), seq_length만 바꿔서
  (과거 L년 -> target_year ERA) 파싱이 제대로 되는지 눈으로 확인하는 스크립트

하드코딩:
- csv_path = "basic.csv"
- max_samples = 20 (train 샘플 기준)

출력:
- train 데이터 일부를 long-format으로 펼쳐서 head(20) 출력
  (sample_idx, player_id, year, step, target_year, target_era_raw, feature들...)
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocess import load_basic_csv, build_playerwise_dataset


CSV_PATH = "basic.csv"
MAX_SAMPLES = 20


def split_keep_ids(X, y, player_ids, val_ratio=0.2, seed=42):
    """train/val split 하면서 player_id도 같이 유지"""
    idx = np.arange(len(X))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_ratio, random_state=seed, shuffle=True
    )
    return X[tr_idx], y[tr_idx], player_ids[tr_idx], X[va_idx], y[va_idx], player_ids[va_idx]


def make_long_df(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    player_ids: np.ndarray,
    feature_cols: list[str],
    x_scaler,
    y_scaler,
    target_year: int,
    seq_length: int,
    max_samples: int = 20,
) -> pd.DataFrame:
    """
    (N, L, F) -> 사람이 보기 좋게 long-format으로 변환
    - inverse_transform으로 원본 스케일 feature/타깃 값을 같이 보여줌
    """
    N, L, F = X_scaled.shape
    n_show = min(max_samples, N)

    hist_years = list(range(target_year - seq_length, target_year))

    # 스케일 복원 (원래 단위로 보여주기)
    X_raw = x_scaler.inverse_transform(X_scaled[:n_show].reshape(-1, F)).reshape(n_show, L, F)
    y_raw = y_scaler.inverse_transform(y_scaled[:n_show])

    rows = []
    for i in range(n_show):
        pid = player_ids[i]
        target_era_raw = float(y_raw[i, 0])

        for t in range(L):
            year = hist_years[t]
            row = {
                "sample_idx": i,
                "player_id": pid,
                "target_year": target_year,
                "target_era_raw": target_era_raw,
                "step": t,     # 0..L-1
                "year": year,  # 실제 입력 연도
            }

            # feature는 raw(원래 스케일)로 출력
            for j, col in enumerate(feature_cols):
                row[col] = float(X_raw[i, t, j])

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, choices=[2019, 2023], required=True)
    parser.add_argument("--seq_length", type=int, choices=[2, 3, 4], required=True)
    args = parser.parse_args()

    df = load_basic_csv(CSV_PATH)

    # period/seq_length에 맞춰 (과거 L년 -> target_year ERA) 샘플 생성
    bundle = build_playerwise_dataset(
        df=df,
        target_year=args.period,
        seq_length=args.seq_length,
        target_col="p_era",
    )

    # train/val split (inspect는 train만 보면 충분)
    X_tr, y_tr, pid_tr, X_va, y_va, pid_va = split_keep_ids(
        bundle.X, bundle.y, bundle.player_ids, val_ratio=0.2, seed=42
    )

    print("\n" + "=" * 80)
    print("[파싱 요약]")
    print(f"- period(target_year): {args.period}")
    print(f"- seq_length(L): {args.seq_length}")
    print(f"- hist_years: {list(range(args.period - args.seq_length, args.period))}")
    print(f"- 전체 샘플 N: {len(bundle.X)}")
    print(f"- train N: {len(X_tr)} | val N: {len(X_va)}")
    print(f"- feature 개수 F: {len(bundle.feature_cols)}")
    print("=" * 80)

    train_long = make_long_df(
        X_scaled=X_tr,
        y_scaled=y_tr,
        player_ids=pid_tr,
        feature_cols=bundle.feature_cols,
        x_scaler=bundle.x_scaler,
        y_scaler=bundle.y_scaler,
        target_year=args.period,
        seq_length=args.seq_length,
        max_samples=MAX_SAMPLES,
    )

    SHOW_COLS = [
        "sample_idx",
        "player_id",
        "target_year",
        "target_era_raw",
        "year",
        "player_age",
    ]

    print("<Sample>")
    print(train_long[SHOW_COLS].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
