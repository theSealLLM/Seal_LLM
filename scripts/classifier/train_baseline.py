"""① 베이스라인(TF-IDF + 로지스틱 회귀) 학습 → validation 평가.

uv run python scripts/classifier/train_baseline.py
uv run python scripts/classifier/train_baseline.py --C 1 --tag c1

결과: runs/classifier/<task>/baseline/<실행 ID>/ (model/ + val_report.json)
실행 ID는 학습을 시작한 시각(분 단위)이라 다시 학습해도 덮어쓰지 않는다. --tag는 ID 뒤에 붙는 메모.
--task를 주지 않으면 inquiries로 돌아간다. documents를 쓰려면 --task documents.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import _common

from theseal.classifier.config import BaselineConfig, display_path, new_run_id
from theseal.classifier.data_loader import load_task
from theseal.classifier.metrics import LATENCY_SAMPLES, evaluate_classifier, save_json
from theseal.classifier.tfidf_logreg import KIND, TfidfLogRegClassifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = BaselineConfig()
    p = argparse.ArgumentParser(description="TF-IDF + 로지스틱 회귀 베이스라인 학습")
    _common.add_common_args(p)
    p.add_argument("--name", default="baseline", help="모델 폴더 이름 (기본: baseline)")
    p.add_argument("--tag", default=None, help="실행 ID 뒤에 붙일 메모 (예: c1 → 20261013-1411_c1)")
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
    run_id = new_run_id(args.tag)  # 학습 시작 시각
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

    run, _ = _common.unique_run_dir(paths, args.name, run_id)
    model_dir = paths.model_dir(args.name, run)
    clf.save(model_dir, name=args.name, task=args.task, run=run)

    result = evaluate_classifier(clf, val, model_dir=model_dir, latency_samples=args.latency_samples)
    report = _common.build_report(
        model=args.name,
        run=run,
        kind=KIND,
        task=args.task,
        model_dir=model_dir,
        result=result,
        train_time_sec=train_time,
        config=asdict(config),
    )
    save_json(report, paths.val_report(args.name, run))

    _common.print_summary(report)
    print(f"\n모델: {display_path(model_dir)}\n"
          f"리포트: {display_path(paths.val_report(args.name, run))}")
    return report


if __name__ == "__main__":
    main()
