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
        """compare.py가 확정한 runs/classifier/<task>/final_model 모델을 불러온다 (기본: inquiries)."""
        return cls(task_paths(task).final_dir, device=device)

    @property
    def labels(self) -> tuple[str, ...]:
        return self.model.label_set.labels

    def predict_batch(self, texts: list[str]) -> list[dict]:
        probs = self.model.predict_proba(texts)
        results = []
        for row in probs:
            best = int(row.argmax())
            results.append({
                "label": self.labels[best],
                "confidence": float(row[best]),
                "scores": {label: float(p) for label, p in zip(self.labels, row)},
            })
        return results

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]
