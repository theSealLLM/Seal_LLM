"""② 사전학습 분류 모델 파인튜닝 → validation 평가 (GPU 권장).

uv run python scripts/classifier/train_classifier_model.py

--model을 생략하면 후보 목록을 번호로 고른다. 번호로 고르면 models/ 아래 받아 둔
폴더가 있을 때 그걸 쓰므로 다시 내려받지 않는다.
--model을 직접 주면 메뉴 없이 바로 학습한다 (HF Hub 이름 또는 로컬 폴더 경로).
처음 실행할 때 Hub 이름을 주면 HF 캐시로 내려받는다. 이미 받아 둔 모델만 쓰려면 --local-files-only.
--task를 주지 않으면 inquiries로 돌아간다. documents를 쓰려면 --task documents.

결과: runs/classifier/<task>/<모델 이름>/<실행 ID>/ (model/ + val_report.json)
실행 ID는 학습을 시작한 시각(분 단위)이라 다시 학습해도 덮어쓰지 않는다. --tag는 ID 뒤에 붙는 메모.
에폭별 가중치는 남기지 않고, validation macro-F1이 가장 좋았던 에폭의 가중치 하나만 저장한다.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import _common
import numpy as np

from theseal.classifier.classifier_models import KIND, TransformerClassifier, train_classifier
from theseal.classifier.config import (
    MODEL_CANDIDATES,
    BertConfig,
    TaskPaths,
    display_path,
    local_model_dir,
    new_run_id,
    safe_name,
)
from theseal.classifier.data_loader import load_task
from theseal.classifier.metrics import LATENCY_SAMPLES, evaluate_classifier, save_json


def default_name(model: str) -> str:
    return safe_name(Path(model.rstrip("/\\")).name)


def choose_model(paths: TaskPaths) -> tuple[str, str | None]:
    """--model을 생략했을 때 후보를 번호로 고르게 한다.

    (모델 경로 또는 Hub 이름, 결과 폴더 이름)을 돌려준다. 번호로 고른 경우 이름은
    저장소 이름에서 뽑아, 로컬 폴더로 돌리든 Hub 이름으로 돌리든 같은 폴더에 쌓이게 한다.
    직접 입력한 경우에는 이름을 정하지 않고 None을 주어 호출한 쪽이 경로에서 뽑게 한다.
    """
    print("\n학습할 모델을 고르세요.")
    for i, repo_id in enumerate(MODEL_CANDIDATES, start=1):
        local = local_model_dir(repo_id)
        where = f"로컬 {display_path(local)}" if local.is_dir() else "HF Hub에서 내려받음"
        root = paths.model_root(default_name(repo_id))
        n_runs = sum(1 for d in root.iterdir() if d.is_dir()) if root.is_dir() else 0
        done = f" · 학습 기록 {n_runs}개" if n_runs else ""
        print(f"  {i}) {repo_id:<45} {where}{done}")
    print("  0) 직접 입력 (HF Hub 이름 또는 폴더 경로)")

    while True:
        try:
            raw = input("번호 [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n[중단] 모델을 고르지 않았습니다.") from None
        if raw == "0":
            try:
                typed = input("모델 이름 또는 경로: ").strip()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit("\n[중단] 모델을 고르지 않았습니다.") from None
            if typed:
                return typed, None
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(MODEL_CANDIDATES):
            repo_id = MODEL_CANDIDATES[int(raw) - 1]
            local = local_model_dir(repo_id)
            # 받아 둔 폴더가 있으면 그걸 쓴다. 없으면 Hub 이름 그대로 넘겨 내려받게 한다.
            return (str(local) if local.is_dir() else repo_id), default_name(repo_id)
        print(f"  1~{len(MODEL_CANDIDATES)} 또는 0을 입력하세요.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = BertConfig()
    p = argparse.ArgumentParser(description="사전학습 분류 모델 파인튜닝")
    _common.add_common_args(p)
    p.add_argument("--model", default=None,
                   help="HF Hub 이름 또는 로컬 경로 (생략하면 번호로 고른다)")
    p.add_argument("--name", default=None, help="모델 폴더 이름 (기본: 모델 폴더/저장소 이름)")
    p.add_argument("--tag", default=None, help="실행 ID 뒤에 붙일 메모 (예: lr2e5 → 20261013-1411_lr2e5)")
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
    paths = _common.paths_from_args(args)

    model, chosen_name = args.model, None
    if model is None:
        if sys.stdin.isatty():
            model, chosen_name = choose_model(paths)
        else:
            # 입력을 받을 수 없는 환경(파이프, 백그라운드 실행)에서는 기본 모델로 간다
            model = BertConfig.model_name
            print(f"[안내] --model이 없고 입력을 받을 수 없어 기본값 {model}로 학습합니다.")

    name = _common.check_name(args.name or chosen_name or default_name(model))
    run_id = new_run_id(args.tag)  # 학습 시작 시각
    config = BertConfig(
        model_name=model,
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

    run, _ = _common.unique_run_dir(paths, name, run_id)
    model_dir = paths.model_dir(name, run)
    clf.save(model_dir, name=name, task=args.task, run=run, config=config)

    result = evaluate_classifier(clf, val, model_dir=model_dir, latency_samples=args.latency_samples)
    report = _common.build_report(
        model=name,
        run=run,
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
    save_json(report, paths.val_report(name, run))

    _common.print_summary(report)
    print(f"\n모델: {display_path(model_dir)}\n리포트: {display_path(paths.val_report(name, run))}")
    return report


if __name__ == "__main__":
    main()
