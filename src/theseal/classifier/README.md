# 분류기 사용하기

학습해 둔 분류 모델을 불러와 문장을 분류합니다.

## 세 줄 요약

```python
from theseal.classifier.classifier import Classifier

clf = Classifier.from_final()                 # config.py의 FINAL_MODEL이 가리키는 모델
result = clf.predict("사이즈가 맞지 않아 반품하려는데 환불은 언제 되나요?")
result.label_name                             # 'refund'
```

위 코드가 돌아가려면 학습이 끝난 모델이 있어야 합니다. 아직 학습한 모델을
공유하지 않고 있어서 각자 직접 학습해야 합니다. 순서는 아래
[모델 학습하기](#모델-학습하기)에 정리해 두었습니다.

## 모델 불러오기

```python
clf = Classifier.from_final()                 # 최종 모델 (기본 task: inquiries, 추후 documents도 추가 예정)
clf = Classifier("runs/classifier/inquiries/mmBERT-small/20261013-1458/model")  # 폴더 직접 지정
clf = Classifier.from_final(device="cpu")     # auto(기본) | cuda | cuda:0 | mps | cpu
```

최종 모델은 `config.py`의 `FINAL_MODEL`에 `{"model": ..., "run": ...}`로 적어 둡니다.
비어 있으면 `FinalModelNotSetError`가 납니다. `comparison.md`를 보고 정하세요.

가중치는 git에 없습니다(`runs/**/model/`은 gitignore). 학습한 컴퓨터에서 쓰거나
아래로 받아 옵니다.

```bash
uv run python scripts/classifier/share_model.py pull --final --repo-id <계정>/<저장소>
```

## 반환값: `ClassificationResult`

`predict`는 튜플이 아니라 객체를 돌려줍니다.

```python
result = clf.predict("사이즈가 맞지 않아 반품하려는데 환불은 언제 되나요?")

result.label_num    # 1                              int
result.label_name   # 'refund'                       str
result.scores       # (0.00105, 0.99812, 0.00083)    tuple[float, ...]
result.confidence   # 0.99812                        float
```

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `label_num` | `int` | 라벨 번호. `scores`에서 확률이 가장 큰 칸의 위치 |
| `label_name` | `str` | 라벨 이름. `clf.labels[label_num]`과 같다 |
| `scores` | `tuple[float, ...]` | 라벨별 확률. 라벨 순서대로이고 합은 1 |
| `confidence` | `float` | 고른 라벨의 확률 (`scores[label_num]`). 프로퍼티 |

번호와 `scores`의 순서는 `labels.json`의 `labels` 순서입니다.
inquiries는 `0=shipping, 1=refund, 2=account`이고, `clf.labels`로 확인할 수 있습니다.

```python
clf.labels          # ('shipping', 'refund', 'account')
```

분류 결과는 고쳐 쓸 값이 아니라 `frozen`입니다. 바꾸려 하면 바로 막힙니다.

```python
result.label_num = 2
# dataclasses.FrozenInstanceError: cannot assign to field 'label_num'
```

## 문장 여러 개

`predict_batch`는 같은 객체의 리스트를 넣은 문장 순서대로 돌려줍니다.

```python
for result in clf.predict_batch(["택배가 사흘째 멈춰 있어요.", "로그인 인증 문자를 못 받아요."]):
    print(result.label_num, result.label_name, round(result.confidence, 5))
# 0 shipping 1.0
# 2 account 1.0
```

## 모델 정보

라벨 판정과는 무관하지만, 어떤 모델을 쓰고 있는지 확인할 때 씁니다.

```python
clf.model_dir        # Path   불러온 모델 폴더
clf.meta             # dict   classifier_meta.json 내용
clf.meta["kind"]     # str    'baseline'(TF-IDF) 또는 'bert'(파인튜닝 모델)
```

## 모델 학습하기

스크립트 번호 순서대로 돌리면 됩니다. 모두 `--task`를 주지 않으면 `inquiries`로
돌아가고, 다른 주제를 쓰려면 `--task documents`를 붙입니다.(아직 documents 학습 기능은 추가되지 않았습니다.)

### 0. 사전학습 모델 받기

```bash
uv run scripts/classifier/fetch_models.py
```

`config.py`의 `MODEL_CANDIDATES`에 적힌 후보를 `models/` 아래로 내려받습니다. 베이스라인만 쓸 거면
건너뛰어도 됩니다.

### 1. 베이스라인 학습

```bash
uv run python scripts/classifier/train_baseline.py
```

TF-IDF + 로지스틱 회귀입니다. 파인튜닝 모델이 이것보다
나은지 재는 기준선이 됩니다.

### 2. 사전학습 모델 파인튜닝

```bash
uv run python scripts/classifier/train_classifier_model.py
```

`--model`을 주지 않으면 후보 목록이 번호로 나옵니다. 다운받은 모델이 없으면 다운받고 진행합니다.

결과는 `runs/classifier/<task>/<모델 이름>/<실행 ID>/` 아래에 `model/`(가중치)과
`val_report.json`(validation 결과 + 그때 쓴 설정)으로 남습니다. 실행 ID는 학습을
시작한 시각(초 단위)이라, 같은 모델을 다시 학습해도 이전 결과를 덮어쓰지 않습니다.
그 실행에서 무엇을 바꿨는지는 `val_report.json`의 `config`를 보면 됩니다.

### 3. 비교표 만들기

```bash
uv run python scripts/classifier/compare.py
```

`val_report.json`이 있는 실행을 전부 같은 조건에서 다시 평가해
`runs/classifier/<task>/comparison.md`에 표로 남깁니다. 모델을 복사하거나
확정하지는 않습니다.

### 4. 최종 모델 정하기

`comparison.md`를 보고 `config.py`의 `FINAL_MODEL`에 직접 적습니다.

```python
FINAL_MODEL = {
    "inquiries": {"model": "mmBERT-small", "run": "20261013-1458"},
    "documents": None,
}
```


### 5. test 평가

```bash
uv run python scripts/classifier/evaluate_test.py
```

모델과 조건을 모두 정한 뒤 **딱 한 번만** 돌립니다. 이미 `test_report.json`이
있으면 `--force` 없이는 다시 실행되지 않습니다.

## 직접 돌려 보기

```bash
uv run src/theseal/classifier/demo.py                      # 문장을 입력하며 대화형으로
```
