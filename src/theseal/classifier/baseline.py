"""베이스라인: TF-IDF + 로지스틱 회귀."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import META_FILENAME, BaselineConfig
from .data_loader import LabelSet
from .metrics import load_json, now_iso, save_json

KIND = "baseline"
MODEL_FILENAME = "model.joblib"


class TfidfLogRegClassifier:
    kind = KIND
    device = "cpu"

    def __init__(self, label_set: LabelSet, config: BaselineConfig | None = None) -> None:
        self.label_set = label_set
        self.config = config or BaselineConfig()
        c = self.config
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer=c.analyzer,
                ngram_range=(c.ngram_min, c.ngram_max),
                min_df=c.min_df,
                sublinear_tf=c.sublinear_tf,
            )),
            ("clf", LogisticRegression(C=c.C, max_iter=c.max_iter, random_state=c.seed)),
        ])

    def fit(self, texts: list[str], labels: list[str]) -> TfidfLogRegClassifier:
        # 숫자 라벨로 학습해야 classes_ 순서가 labels.json 순서(0, 1, 2 ...)와 같아진다
        self.pipeline.fit(texts, self.label_set.encode(labels))
        classes = self.pipeline.named_steps["clf"].classes_.tolist()
        if classes != list(range(len(self.label_set))):
            raise ValueError(f"학습 데이터에 없는 라벨이 있습니다 (classes_={classes}).")
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: list[str]) -> list[str]:
        return self.label_set.decode(self.predict_proba(texts).argmax(axis=1))

    def describe(self) -> dict:
        clf = self.pipeline.named_steps["clf"]
        return {
            "num_parameters": int(clf.coef_.size + clf.intercept_.size),
            "vocab_size": len(self.pipeline.named_steps["tfidf"].vocabulary_),
        }

    def save(self, model_dir: Path, *, name: str, task: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_dir / MODEL_FILENAME)
        save_json({
            "kind": KIND,
            "name": name,
            "task": task,
            "labels": list(self.label_set.labels),
            "config": asdict(self.config),
            "saved_at": now_iso(),
        }, model_dir / META_FILENAME)

    @classmethod
    def load(cls, model_dir: Path) -> TfidfLogRegClassifier:
        model_dir = Path(model_dir)
        meta = load_json(model_dir / META_FILENAME)
        if meta["kind"] != KIND:
            raise ValueError(f"{model_dir}는 베이스라인 모델이 아닙니다 (kind={meta['kind']}).")
        obj = cls(LabelSet(tuple(meta["labels"])), BaselineConfig(**meta["config"]))
        obj.pipeline = joblib.load(model_dir / MODEL_FILENAME)
        return obj
