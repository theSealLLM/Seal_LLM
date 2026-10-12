"""① 베이스라인(TF-IDF + 로지스틱 회귀) 학습 → validation 평가.

uv run python scripts/classifier/train_baseline.py

--task를 주지 않으면 inquiries로 돌아간다. documents를 쓰려면 --task documents.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import _common

from theseal.classifier.tfidf_logreg import KIND, TfidfLogRegClassifier
from theseal.classifier.config import BaselineConfig, display_path
from theseal.classifier.data_loader import load_task
from theseal.classifier.metrics import LATENCY_SAMPLES, evaluate_classifier, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = BaselineConfig()
    p = argparse.ArgumentParser(description="TF-IDF + 로지스틱 회귀 베이스라인 학습")
    _common.add_common_args(p)
    p.add_argument("--name", default="baseline", help="결과 폴더 이름 (기본: baseline)")
    p.add_argument("--analyzer", default=d.analyzer, choices=["char_wb", "char", "word"])
    p.add_argument("--ngram-min", type=int, default=d.ngram_min)
    p.add_argument("--ngram-max", type=int, default=d.ngram_max)
    p.add_argument("--min-df", type=int, default=d.min_df)
    p.add_argument("--C", type=float, default=d.C, help="로지스틱 회귀 규제 강도의 역수")
    p.add_argument("--max-iter", type=int, default=d.max_iter)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--latency-samples", type=int, default=LATENCY_SAMPLES)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    _common.check_name(args.name)
    paths = _common.paths_from_args(args)
    config = BaselineConfig(
        analyzer=args.analyzer,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        C=args.C,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    label_set, splits = load_task(paths.data_dir, ("train", "validation"))
    train, val = splits["train"], splits["validation"]
    print(f"[data] {args.task} train={len(train)} validation={len(val)} labels={label_set.labels}")

    clf = TfidfLogRegClassifier(label_set, config)
    start = time.perf_counter()
    clf.fit(train.texts, train.labels)
    train_time = round(time.perf_counter() - start, 3)

    model_dir = paths.model_dir(args.name)
    clf.save(model_dir, name=args.name, task=args.task)

    result = evaluate_classifier(clf, val, model_dir=model_dir, latency_samples=args.latency_samples)
    report = _common.build_report(
        name=args.name,
        kind=KIND,
        task=args.task,
        model_dir=model_dir,
        result=result,
        train_time_sec=train_time,
        config=asdict(config),
    )
    save_json(report, paths.val_report(args.name))

    _common.print_summary(report)
    print(f"\n모델: {display_path(model_dir)}\n리포트: {display_path(paths.val_report(args.name))}")
    return report


if __name__ == "__main__":
    main()
