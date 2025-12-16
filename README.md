
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

## 옵션 설명
- `--period` : 예측하려는 타깃 연도(2019 또는 2023)
- `--seq_length` : 타깃 연도 직전 L년치 기록을 입력으로 사용(2/3/4)
- `--model` : lstm | bilstm | gru


