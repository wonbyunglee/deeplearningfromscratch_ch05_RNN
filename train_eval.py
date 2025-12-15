"""
train_eval.py (PyTorch)
- argparse로 period(타깃 연도), sequence_length, model 등을 선택해서 학습/평가
- val 예측 결과 출력 시 player_id도 함께 출력(실습용)

예시)
python train_eval.py --csv_path basic.csv --period 2019 --seq_length 4 --model lstm
python train_eval.py --csv_path basic.csv --period 2023 --seq_length 2 --model gru
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_preprocess import load_basic_csv, build_playerwise_dataset


def _build_model(model_name: str, input_size: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    m = model_name.lower()
    if m == "lstm":
        from models.lstm import LSTMRegressor
        return LSTMRegressor(input_size=input_size, hidden_size=hidden, num_layers=layers, dropout=dropout)
    if m == "bilstm":
        from models.bilstm import BiLSTMRegressor
        return BiLSTMRegressor(input_size=input_size, hidden_size=hidden, num_layers=layers, dropout=dropout)
    if m == "gru":
        from models.gru import GRURegressor
        return GRURegressor(input_size=input_size, hidden_size=hidden, num_layers=layers, dropout=dropout)
    raise ValueError(f"지원하지 않는 모델: {model_name} (lstm|bilstm|gru)")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        ys.append(yb.detach().cpu().numpy())
        preds.append(out.detach().cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="basic.csv", help="basic.csv 경로")
    parser.add_argument("--period", type=int, choices=[2019, 2023], default=2019, help="예측 타깃 연도")
    parser.add_argument("--seq_length", type=int, choices=[2, 3, 4], default=4, help="입력 시퀀스 길이(L)")
    parser.add_argument("--model", type=str, choices=["lstm", "bilstm", "gru"], default="lstm", help="사용 모델")

    # 학습 하이퍼파라미터(실습용)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)

    args = parser.parse_args()

    # 재현성(실습용)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 1) 데이터 로드
    df = load_basic_csv(args.csv_path)

    # 2) (선수별) 시퀀스 데이터 생성
    bundle = build_playerwise_dataset(
        df=df,
        target_year=args.period,
        seq_length=args.seq_length,
        feature_cols=None,
        target_col="p_era",
    )

    print(f"[데이터] period={args.period}, seq_length={args.seq_length}")
    print(f"  - 샘플 수 N = {bundle.X.shape[0]}")
    print(f"  - 입력 shape = {bundle.X.shape} (N, L, F)")
    print(f"  - 타깃 shape = {bundle.y.shape} (N, 1)")

    # 3) train/val split (player_id도 함께 유지되도록 index로 split)
    idx = np.arange(len(bundle.X))
    train_idx, val_idx = train_test_split(
        idx, test_size=args.val_ratio, random_state=args.seed, shuffle=True
    )

    X_train, y_train, pid_train = bundle.X[train_idx], bundle.y[train_idx], bundle.player_ids[train_idx]
    X_val, y_val, pid_val = bundle.X[val_idx], bundle.y[val_idx], bundle.player_ids[val_idx]

    # torch tensor 변환
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False)

    # 4) 모델 생성
    model = _build_model(
        args.model,
        input_size=X_train.shape[2],
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout
    )
    model.to(device)
    print(model)

    # 5) 학습 루프
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()

            tr_losses.append(loss.item())

        # val loss
        yv_true, yv_pred = evaluate(model, val_loader, device)
        val_loss = float(np.mean((yv_true - yv_pred) ** 2))

        print(f"Epoch {epoch:03d} | train_mse={np.mean(tr_losses):.6f} | val_mse={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[EarlyStopping] patience={args.patience} 도달 → 학습 종료")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) 평가 (스케일 복원 후 지표 계산)
    y_val_true_scaled, y_val_pred_scaled = evaluate(model, val_loader, device)

    y_val_true = bundle.y_scaler.inverse_transform(y_val_true_scaled)
    y_val_pred = bundle.y_scaler.inverse_transform(y_val_pred_scaled)

    _rmse = rmse(y_val_true, y_val_pred)
    _mae = float(np.mean(np.abs(y_val_true - y_val_pred)))
    _mape = mape(y_val_true, y_val_pred)

    print("\n[평가: Validation]")
    print(f"  RMSE = {_rmse:.4f}")
    print(f"  MAE  = {_mae:.4f}")
    print(f"  MAPE = {_mape:.4f}")

    # 상위 몇 개 예측 결과 확인 (player_id 포함)
    N_SHOW = min(20, len(y_val_true))

    rows = []
    for i in range(N_SHOW):
        rows.append({
            "sample": i,
            "player_id": int(pid_val[i]),
            "true": float(y_val_true[i, 0]),
            "pred": round(float(y_val_pred[i, 0]), 2),
        })

    df_pred = pd.DataFrame(rows)

    print("\n[Validation Predictions]")
    print(df_pred.to_string(index=False))



if __name__ == "__main__":
    main()
