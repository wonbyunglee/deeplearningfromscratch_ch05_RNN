
# Deep Learning from Scratch 2 Ch05 RNN (Baseball ERA Prediction with RNNs)

![alt text](image.png)
* Illustrated by DALL·E 4o

## 파일 구조 
- `data_preprocess.py` : 데이터 로드/전처리 + (선수별) 시퀀스 샘플 생성
- `models/`
  - `lstm.py` 
  - `bilstm.py` 
  - `gru.py` 
- `sample.py` : 파싱된 데이터 확인용 코드
- `train_eval.py` : 실험 세팅 옵션 및 모델 파라미터 선택 후 학습/평가

## 실습 순서
## 1. 데이터 셋팅 (sample.py)
### A. 입력 변수 선택
- load_basic_csv 함수 -> df.drop(columns=) 에서 학습에서 제거하고자 하는 변수명 입력
```bash
def load_basic_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ##############################################
    #### 원하는 feature 추가/제거 해보기! ####
    # df = df.drop(columns=['여기 변수명 입력'])
    ##############################################
```
<변수 설명 테이블>
| 변수명                  | 의미                          |
| -------------------- | ------------------------------ |
| `p_game`             | 시즌 동안 등판한 경기 수                 |
| `k_percent`          | 삼진 비율 (%)                      |
| `bb_percent`         | 볼넷 비율 (%)                      |
| `slg_percent`        | 장타율 (Slugging Percentage)      |
| `on_base_percent`    | 출루율 (On-Base Percentage)       |
| `p_save`             | 세이브(Save) 개수                   |
| `p_win`              | 승리(Wins) 수                     |
| `p_loss`             | 패배(Losses) 수                   |
| `p_quality_start`    | 퀄리티 스타트(Quality Start) 횟수      |
| `p_hold`             | 홀드(Hold) 개수                    |
| `in_zone_percent`    | 스트라이크 존 안에 투구된 비율 (%)          |
| `edge_percent`       | 스트라이크 존 가장자리(edge)에 투구된 비율 (%) |
| `swing_percent`      | 타자가 스윙한 비율 (%)                 |
| `f_strike_percent`   | 초구 스트라이크 비율 (%)                |
| `flyballs_percent`   | 플라이볼 비율 (%)                    |
| `linedrives_percent` | 라인드라이브 비율 (%)                  |
| `popups_percent`     | 팝업 타구 비율 (%)                   |
| `fastball_avg_speed` | 패스트볼 평균 구속 (mph)               |
| `fastball_avg_spin`  | 패스트볼 평균 회전수 (rpm)              |
| `fastball_avg_break` | 패스트볼 평균 무브먼트(브레이크)             |
| `breaking_avg_speed` | 브레이킹볼 평균 구속 (mph)              |
| `breaking_avg_break` | 브레이킹볼 평균 무브먼트                  |
| `offspeed_avg_speed` | 오프스피드 평균 구속 (mph)              |
| `offspeed_avg_spin`  | 오프스피드 평균 회전수 (rpm)             |
| `offspeed_avg_break` | 오프스피드 평균 무브먼트                  |
### B. 타겟 연도, 시퀀스 길이 설정
- 타겟 연도 (period) : 2019 / 2023

      2019 예측 <- 2015~2018 데이터 학습
      2023 예측 <- 2019~2022 데이터 학습
- 시퀀스 길이 (seq_length) : 2 / 3 / 4
      
      2019 예측 (seq_length = 2) <- 2017,2018 데이터 학습
      2019 예측 (seq_length = 3) <- 2016,2017,2018 데이터 학습
      2019 예측 (seq_length = 4) <- 2015,2016,2017,2018 데이터 학습
      
      2023 예측 (seq_length = 2) <- 2021,2022 데이터 학습
      2023 예측 (seq_length = 3) <- 2020,2021,2022 데이터 학습
      2023 예측 (seq_length = 4) <- 2019,2020,2021,2022 데이터 학습

- terminal 창에 아래와 같이 period, seq_length 설정해서 실행
```bash
python sample.py --period (2019 / 2023) --seq_length (2 / 3 / 4)
```
### C. 데이터 파싱 확인
- **[선택된 변수들]** 에서 변수들이 알맞게 선택되었는지 확인
- **[파싱 요약]** 에서 선택한 period, seq_length와 입력 년도, 전체 샘플 수, feature 개수 확인
- **[샘플 데이터]** 에서 최종 데이터가 알맞은 period, seq_length 대로 파싱되었는지 확인

## 2. 모델 학습 및 평가 (train_eval.py)
### A. 모델 확인 (models/simple_rnn,lstm,gru,bilstm.py)

- **Simple_RNN** (밑바닥2) : 기본적인 순환 신경망으로 과거 시계열 정보를 순차적으로 누적해 다음 시즌 ERA를 예측하는 가장 단순한 베이스라인 모델.

- **LSTM** (Hochreiter & Schmidhuber, 1997) : 게이트 구조를 통해 장기 의존성을 효과적으로 학습하며, 과거 L년치 투수 기록을 요약해 ERA를 회귀 예측하는 모델.

- **GRU** (Chung et al., 2014) : LSTM을 단순화한 구조로 reset·update gate만을 사용해 효율적으로 시계열 패턴을 학습하는 경량화된 RNN 모델.

- **BiLSTM** (Schuster & Paliwal, 1997): 시계열을 정방향과 역방향으로 동시에 처리해 과거와 미래 문맥 정보를 함께 활용하여 ERA를 예측하는 양방향 LSTM 모델.

### B. 모델 파라미터 설정
- train_eval.py의 def main()에서 파라미터 설정 가능

  *원본 코드의 default 값이 통상적으로 쓰이는 값
```
def main():

    ##### 원하는 파라미터 값 입력 #####
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
```

| 파라미터           | 설명                        | 값을 늘리면                  | 값을 줄이면                  |
| -------------- | ------------------------- | ----------------------- | ----------------------- |
| `--epochs`     | 전체 학습 데이터를 반복 학습하는 횟수     | 더 충분한 학습 가능하나 과적합 위험 증가 | 학습 부족(underfitting) 가능  |
| `--batch_size` | 한 번의 업데이트에 사용하는 샘플 수      | 학습이 안정적이나 메모리 사용량 증가    | 학습이 불안정해질 수 있으나 일반화에 도움 |
| `--val_ratio`  | 검증 데이터로 분리할 비율            | 검증 신뢰도↑, 학습 데이터 감소      | 학습 데이터↑, 검증 성능 신뢰도↓     |
| `--seed`       | 난수 시드(재현성 제어)             | 결과 자체는 변하지 않음           | 결과 자체는 변하지 않음           |
| `--hidden`     | 은닉 상태 차원(모델 용량)           | 표현력↑, 과적합·연산량 증가        | 단순해져 복잡한 패턴 학습 어려움      |
| `--layers`     | 순환 신경망 층 수                | 복잡한 시계열 패턴 학습 가능        | 학습은 안정적이나 표현력 제한        |
| `--dropout`    | 드롭아웃 비율(정규화 강도)           | 과적합 감소, 학습 느려질 수 있음     | 과적합 위험 증가               |
| `--lr`         | 학습률(업데이트 크기)              | 학습 불안정·발산 위험            | 학습 속도 매우 느려짐            |
| `--patience`   | early stopping 대기 epoch 수 | 더 오래 학습, 과적합 가능         | 학습이 너무 빨리 종료될 수 있음      |

### C. 학습 및 평가 실행
- terminal에서 train_eval 파일 실행
- 실행창에서도 파라미터 설정 가능
- ex) period : 2019, seq_length : 2, model : lstm, epochs : 300, batch_size : 128
```
python train_eval.py --period 2019 --seq_length 2 --model lstm --epochs 300 --batch_size 128
```

### D. 학습 결과 확인
- **[데이터]** 에서 설정한 period, seq_length 및 모델 shape 확인, 선택한 모델 프레임워크 및 파라미터 확인
```
[데이터] period=2019, seq_length=2
  - 샘플 수 N = 224
  - 입력 shape = (224, 2, 26) (N, L, F)
  - 타깃 shape = (224, 1) (N, 1)
LSTMRegressor(
  (lstm): LSTM(26, 128, batch_first=True)
  (head): Sequential(
    (0): Dropout(p=0.3, inplace=False)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=64, out_features=1, bias=True)
  )
)
```
- 모델 학습 과정 체크
```
Epoch 001 | train_mse=0.100043 | val_mse=0.085891
Epoch 002 | train_mse=0.063051 | val_mse=0.061223
Epoch 003 | train_mse=0.048184 | val_mse=0.041967
Epoch 004 | train_mse=0.031839 | val_mse=0.031284
Epoch 005 | train_mse=0.029095 | val_mse=0.033472
.
.
[EarlyStopping] patience=20 도달 → 학습 종료
```
- validation 평가 지표 확인
```
[평가: Validation]
  RMSE = 1.3794
  MAE  = 1.0634
  MAPE = 0.2408
  ```
- 모델이 예측한 output 샘플 20개 확인
```
[Validation Predictions]
 sample  player_id  true  pred
      0     518774  7.09  4.85
      1     456701  4.57  5.87
      2     502043  4.84  5.00
      3     543037  2.50  3.72
      4     502188  3.52  4.99
      5     543243  2.87  4.35
      6     501625  3.36  4.19
      7     458708  3.74  5.18
      8     502624  4.21  4.05
      9     429719  9.58  4.92
     10     593372  3.17  4.27
     11     607074  5.19  4.56
     12     592717  5.89  3.95
     13     545333  4.48  3.51
     14     543208  4.61  5.58
     15     453562  4.64  4.42
     16     594835  3.99  4.39
     17     519242  4.40  2.92
     18     542882  4.71  4.85
     19     593576  2.93  4.39
  ```

## 옵션 설명
- `--period` : 예측하려는 타깃 연도(2019 또는 2023)
- `--seq_length` : 타깃 연도 직전 L년치 기록을 입력으로 사용(2/3/4)
- `--model` : lstm | bilstm | gru


