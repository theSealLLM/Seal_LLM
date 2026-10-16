"""저장된 분류 모델을 종류와 상관없이 같은 방식으로 불러와 쓴다 (나중에 FastAPI에서 사용)."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ClassificationResult:
    """분류 결과 하나. Classifier.predict와 predict_batch의 반환값.

        result = clf.predict("사이즈가 맞지 않아 반품하려는데 환불은 언제 되나요?")

        result.text        # '사이즈가 맞지 않아 반품하려는데 환불은 언제 되나요?'
        result.label_num   # 1
        result.label_name  # 'refund'
        result.scores      # (0.00105, 0.99812, 0.00083)
        result.confidence  # 0.99812

    label_num과 scores의 순서는 labels.json의 labels 순서다 (inquiries: 0=shipping, 1=refund, 2=account).
    고쳐 쓸 일이 없는 값이라 frozen으로 두고 scores도 튜플로 둔다.
    """

    text: str
    """분류한 원문. predict에 넣은 문장 그대로다."""

    label_num: int
    """라벨 번호. scores에서 확률이 가장 큰 칸의 위치다."""

    label_name: str
    """라벨 이름. clf.labels[label_num]과 같다."""

    scores: tuple[float, ...]
    """라벨별 확률. 라벨 순서대로이고 합은 1이다."""

    @property
    def confidence(self) -> float:
        """고른 라벨의 확률 (scores[label_num])."""
        return self.scores[self.label_num]


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

    def predict(self, text: str) -> ClassificationResult:
        """문장 하나 → ClassificationResult.

        예) result = clf.predict("환불은 언제 되나요?")
            result.label_num, result.label_name, result.confidence  →  1, 'refund', 0.99812
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """문장 여러 개 → ClassificationResult 리스트. 순서는 넣은 문장 순서와 같다.

        예) for result in clf.predict_batch(texts):
                print(result.label_name, result.text)
        """
        labels = self.labels
        results = []
        for text, row in zip(texts, self.model.predict_proba(texts), strict=True):
            label_num = int(row.argmax())
            results.append(ClassificationResult(
                text=text,
                label_num=label_num,
                label_name=labels[label_num],
                scores=tuple(float(p) for p in row),
            ))
        return results
