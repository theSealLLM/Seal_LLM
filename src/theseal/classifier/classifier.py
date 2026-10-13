"""저장된 분류 모델을 종류와 상관없이 같은 방식으로 불러와 쓴다 (나중에 FastAPI에서 사용)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .config import DEFAULT_TASK, META_FILENAME, task_paths
from .data_loader import LabelSet
from .metrics import load_json


class ClassifierProtocol(Protocol):
    kind: str
    label_set: LabelSet

    @property
    def device(self) -> str: ...

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    def predict(self, texts: list[str]) -> list[str]: ...

    def describe(self) -> dict: ...


def read_meta(model_dir: Path) -> dict:
    path = Path(model_dir) / META_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"모델 메타 파일이 없습니다: {path}")
    return load_json(path)


def load_classifier(model_dir: Path, device: str | None = None) -> ClassifierProtocol:
    meta = read_meta(model_dir)
    if meta["kind"] == "baseline":
        from .tfidf_logreg import TfidfLogRegClassifier

        return TfidfLogRegClassifier.load(model_dir)
    if meta["kind"] == "bert":
        from .classifier_models import TransformerClassifier

        return TransformerClassifier.load(model_dir, device=device)
    raise ValueError(f"알 수 없는 모델 종류: {meta['kind']!r}")


class Classifier:
    def __init__(self, model_dir: Path, device: str | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.meta = read_meta(self.model_dir)
        self.model = load_classifier(self.model_dir, device=device)

    @classmethod
    def from_final(cls, task: str = DEFAULT_TASK, device: str | None = None) -> Classifier:
        """config.py의 FINAL_MODEL[task]가 가리키는 모델을 불러온다 (기본: inquiries).

        FINAL_MODEL이 비어 있으면 FinalModelNotSetError가 난다.
        """
        return cls(task_paths(task).final_model_dir, device=device)

    @property
    def labels(self) -> tuple[str, ...]:
        """라벨 이름. labels[번호]로 predict가 돌려준 번호를 이름으로 바꾼다 (labels.json 순서)."""
        return self.model.label_set.labels

    def predict(self, text: str) -> tuple[int, list[float]]:
        """문장 하나 → (라벨 번호, 라벨별 확률 배열).

        - 라벨 번호와 배열 순서는 labels.json의 labels 순서다 (inquiries: 0=shipping, 1=refund, 2=account).
        - 라벨 번호는 확률이 가장 큰 칸의 위치이고, 확률 배열의 합은 1이다.

        예) label, scores = clf.predict("환불은 언제 되나요?")  →  1, [0.00105, 0.99812, 0.00083]
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[tuple[int, list[float]]]:
        """문장 여러 개 → [(라벨 번호, 라벨별 확률 배열), ...]. 형식은 predict와 같다."""
        probs = self.model.predict_proba(texts)
        return [(int(row.argmax()), [float(p) for p in row]) for row in probs]
