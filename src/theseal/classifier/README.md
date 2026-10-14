# 분류기 사용하기

학습해 둔 분류 모델을 불러와 문장을 분류합니다. 모델 종류(TF-IDF 베이스라인,
파인튜닝 모델)와 상관없이 쓰는 방법은 같습니다.

## 세 줄 요약

```python
from theseal.classifier.classifier import Classifier

clf = Classifier.from_final()                 # config.py의 FINAL_MODEL이 가리키는 모델
result = clf.predict("사이즈가 맞지 않아 반품하려는데 환불은 언제 되나요?")
result.label_name                             # 'refund'
```

`src`가 import 경로에 있어야 합니다.

## 모델 불러오기

```python
clf = Classifier.from_final()                 # 최종 모델 (기본 task: inquiries)
clf = Classifier.from_final("documents")      # task 지정
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

## 직접 돌려 보기

```bash
uv run src/theseal/classifier/demo.py                      # 문장을 입력하며 대화형으로
uv run python scripts/classifier/classifier_example.py     # 예제 문장으로 한 번에
```
