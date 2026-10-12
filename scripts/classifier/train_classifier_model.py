"""② 사전학습 분류 모델 파인튜닝 → validation 평가 (GPU 권장).

uv run python scripts/classifier/train_bert.py --task inquiries --model klue/roberta-base

--model에는 HF Hub 이름이나 미리 받아 둔 로컬 폴더 경로를 넣는다.
처음 실행할 때 Hub 이름을 주면 HF 캐시로 내려받는다. 이미 받아 둔 모델만 쓰려면 --local-files-only.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict
from pathlib import Path

import _common
import numpy as np

from theseal.classifier.classifier_models import KIND, TransformerClassifier, train_classifier
from theseal.classifier.config import BertConfig, display_path
from theseal.classifier.data_loader import load_task
from theseal.classifier.metrics import LATENCY_SAMPLES, evaluate_classifier, save_json


def default_name(model: str) -> str:
    base = Path(model.rstrip("/\\")).name or "model"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", base)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = BertConfig()
    p = argparse.ArgumentParser(description="사전학습 분류 모델 파인튜닝")
    _common.add_common_args(p)
    p.add_argument("--model", default=d.model_name,
                   help=f"HF Hub 이름 또는 로컬 경로 (기본: {d.model_name})")
    p.add_argument("--name", default=None, help="결과 폴더 이름 (기본: 모델 폴더/저장소 이름)")
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--eval-batch-size", type=int, default=d.eval_batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-ratio", type=float, default=d.warmup_ratio)
    p.add_argument("--max-grad-norm", type=float, default=d.max_grad_norm)
    p.add_argument("--patience", type=int, default=d.patience,
                   help="validation macro-F1이 이 에폭 수만큼 안 오르면 중단 (0: 끄기)")
    p.add_argument("--precision", default=d.precision, choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | mps | cpu")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--local-files-only", action="store_true",
                   help="HF Hub에서 내려받지 않고 캐시/로컬 경로만 사용")
    p.add_argument("--latency-samples", type=int, default=LATENCY_SAMPLES)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    name = _common.check_name(args.name or default_name(args.model))
    paths = _common.paths_from_args(args)
    config = BertConfig(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        precision=args.precision,
        local_files_only=args.local_files_only,
        seed=args.seed,
    )

    label_set, splits = load_task(paths.data_dir, ("train", "validation"))
    train, val = splits["train"], splits["validation"]
    print(f"[data] {args.task} train={len(train)} validation={len(val)} labels={label_set.labels}")

    clf = TransformerClassifier.from_pretrained(
        config.model_name,
        label_set,
        max_length=config.max_length,
        device=args.device,
        local_files_only=config.local_files_only,
        eval_batch_size=config.eval_batch_size,
    )
    print(f"[model] {config.model_name} → device={clf.device} "
          f"params={clf.describe()['num_parameters']:,}")

    lengths = np.array(clf.token_lengths(train.texts + val.texts))
    token_stats = {
        "max": int(lengths.max()),
        "p95": int(np.percentile(lengths, 95)),
        "truncated": int((lengths > config.max_length).sum()),
    }
    print(f"[tokens] max={token_stats['max']} p95={token_stats['p95']} "
          f"max_length={config.max_length} 잘린 문장={token_stats['truncated']}")
    if token_stats["truncated"]:
        print(f"[경고] max_length보다 긴 문장이 {token_stats['truncated']}개 있습니다. "
              "--max-length를 늘리는 것을 고려하세요.")

    result_train = train_classifier(clf, train, val, config)

    model_dir = paths.model_dir(name)
    clf.save(model_dir, name=name, task=args.task, config=config)

    result = evaluate_classifier(clf, val, model_dir=model_dir, latency_samples=args.latency_samples)
    report = _common.build_report(
        name=name,
        kind=KIND,
        task=args.task,
        model_dir=model_dir,
        result=result,
        train_time_sec=result_train.train_time_sec,
        peak_gpu_memory_mb_train=result_train.peak_gpu_memory_mb,
        config=asdict(config),
        training={
            "best_epoch": result_train.best_epoch,
            "best_val_macro_f1": result_train.best_val_macro_f1,
            "stopped_early": result_train.stopped_early,
            "precision": result_train.precision,
            "token_lengths": token_stats,
            "history": result_train.history,
        },
    )
    save_json(report, paths.val_report(name))

    _common.print_summary(report)
    print(f"\n모델: {display_path(model_dir)}\n리포트: {display_path(paths.val_report(name))}")
    return report


if __name__ == "__main__":
    main()
