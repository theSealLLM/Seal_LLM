# 분류 모델 비교 · inquiries

- 생성 시각: 2026-10-13T11:27:54+09:00
- 평가 데이터: validation 120건
- 평가 방식: 이 스크립트에서 모든 실행을 같은 조건으로 다시 평가
- 지연 시간: 한 문장씩 넣었을 때 (워밍업 후), 처리량: validation 전체를 한 번에 넣었을 때
- 현재 최종 모델 (config.py FINAL_MODEL): 없음

## 참고 추천: `baseline/20261013-1057`

macro_f1 최고값 1.0000와 0.01 이내인 실행(bert-base/20261013-1055, baseline/20261013-1057, koelectra-small-v3-discriminator/20261013-1056, koelectra-small-v3-discriminator/20261013-1122, roberta-base/20261013-1029) 중 지연 시간이 가장 짧음 (0.23ms, macro_f1 0.9917)  
(규칙: macro_f1 최고값과 0.01 이내 → 지연 중앙값이 짧은 순 → 크기가 작은 순)

이 실행으로 정하려면 `src/theseal/classifier/config.py`의 `FINAL_MODEL`에 아래처럼 적는다.

```python
"inquiries": {"model": "baseline", "run": "20261013-1057"},
```

## 성능·시간·자원

| 모델 | 종류 | Macro-F1 | Accuracy | 학습 시간(s) | 지연 중앙값(ms) | 지연 p95(ms) | 처리량(문장/s) | 크기(MB) | 파라미터 수 | GPU 피크 학습/추론(MB) | 장치 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-base/20261013-1055 | bert | 1.0000 | 1.0000 | 36.44 | 8.522 | 8.805 | 339.3 | 422.96 | 110,619,651 | - / - | mps |
| baseline/20261013-1057 | baseline | 0.9917 | 0.9917 | 0.065 | 0.23 | 0.246 | 19364.2 | 0.43 | 30,177 | - / - | cpu |
| koelectra-small-v3-discriminator/20261013-1056 | bert | 0.9917 | 0.9917 | 14.39 | 9.157 | 10.726 | 1467.9 | 54.93 | 14,122,755 | - / - | mps |
| koelectra-small-v3-discriminator/20261013-1122 | bert | 0.9917 | 0.9917 | 14.29 | 7.385 | 10.405 | 1460.9 | 54.93 | 14,122,755 | - / - | mps |
| roberta-base/20261013-1029 | bert | 0.9917 | 0.9917 | 47.74 | 8.591 | 9.191 | 351.9 | 422.96 | 110,620,419 | - / - | mps |

## 실행별 설정

| 모델/실행 | 주요 설정 |
|---|---|
| bert-base/20261013-1055 | lr=3e-05, batch=16, epochs=10 (best 1), max_len=128, seed=42 |
| baseline/20261013-1057 | char_wb 1~3gram, C=10.0, min_df=1 |
| koelectra-small-v3-discriminator/20261013-1056 | lr=3e-05, batch=16, epochs=10 (best 5), max_len=128, seed=42 |
| koelectra-small-v3-discriminator/20261013-1122 | lr=3e-05, batch=16, epochs=10 (best 6), max_len=128, seed=42 |
| roberta-base/20261013-1029 | lr=3e-05, batch=16, epochs=10 (best 2), max_len=128, seed=42 |

## 클래스별 F1

| 모델 | shipping F1 | refund F1 | account F1 |
|---|---|---|---|
| bert-base/20261013-1055 | 1.0000 | 1.0000 | 1.0000 |
| baseline/20261013-1057 | 0.9877 | 0.9873 | 1.0000 |
| koelectra-small-v3-discriminator/20261013-1056 | 1.0000 | 0.9873 | 0.9877 |
| koelectra-small-v3-discriminator/20261013-1122 | 1.0000 | 0.9873 | 0.9877 |
| roberta-base/20261013-1029 | 0.9873 | 0.9877 | 1.0000 |

## 혼동행렬 · bert-base/20261013-1055

| 정답 \ 예측 | shipping | refund | account |
|---|---|---|---|
| shipping | 40 | 0 | 0 |
| refund | 0 | 40 | 0 |
| account | 0 | 0 | 40 |

## 혼동행렬 · baseline/20261013-1057

| 정답 \ 예측 | shipping | refund | account |
|---|---|---|---|
| shipping | 40 | 0 | 0 |
| refund | 1 | 39 | 0 |
| account | 0 | 0 | 40 |

## 혼동행렬 · koelectra-small-v3-discriminator/20261013-1056

| 정답 \ 예측 | shipping | refund | account |
|---|---|---|---|
| shipping | 40 | 0 | 0 |
| refund | 0 | 39 | 1 |
| account | 0 | 0 | 40 |

## 혼동행렬 · koelectra-small-v3-discriminator/20261013-1122

| 정답 \ 예측 | shipping | refund | account |
|---|---|---|---|
| shipping | 40 | 0 | 0 |
| refund | 0 | 39 | 1 |
| account | 0 | 0 | 40 |

## 혼동행렬 · roberta-base/20261013-1029

| 정답 \ 예측 | shipping | refund | account |
|---|---|---|---|
| shipping | 39 | 1 | 0 |
| refund | 0 | 40 | 0 |
| account | 0 | 0 | 40 |
