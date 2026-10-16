"""④ 확정한 모델로 test를 평가한다. 모델과 조건을 정한 뒤 딱 한 번만 실행한다.

uv run python scripts/classifier/evaluate_test.py

--task를 주지 않으면 inquiries로 돌아간다. documents를 쓰려면 --task documents.

- config.py의 FINAL_MODEL[task]에 최종 모델(모델 이름 + 실행 ID)이 적혀 있어야 실행된다.
- 이미 test_report.json이 있으면 다시 실행하지 않는다 (--force로만 덮어쓰기).
"""

from __future__ import annotations

import argparse

import _common

from theseal.classifier.classifier import load_classifier, read_meta
from theseal.classifier.config import FinalModelNotSetError, display_path, final_choice
from theseal.classifier.data_loader import load_task
from theseal.classifier.metrics import (
    LATENCY_SAMPLES,
    confusion_md,
    evaluate_classifier,
    load_json,
    now_iso,
    per_class_table_md,
    save_json,
    summary_table_md,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="확정 모델의 test 최종 평가")
    _common.add_common_args(p)
    p.add_argument("--device", default="auto", help="사전학습 모델 평가 장치")
    p.add_argument("--latency-samples", type=int, default=LATENCY_SAMPLES)
    p.add_argument("--force", action="store_true", help="이미 있는 test 결과를 덮어쓴다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    paths = _common.paths_from_args(args)

    try:
        model, run = final_choice(args.task)
    except FinalModelNotSetError as e:
        raise SystemExit(f"[중단] {e} test는 최종 모델을 정한 뒤에만 봅니다.") from None
    model_dir = paths.model_dir(model, run)
    if not model_dir.exists():
        raise SystemExit(f"[중단] 최종 모델 가중치가 없습니다: {display_path(model_dir)}. "
                         "가중치는 git에 올라가지 않으니 학습한 컴퓨터에서 실행하거나 "
                         "share_model.py pull --final 로 받아 오세요.")
    final_meta = read_meta(model_dir)
    val_path = paths.val_report(model, run)
    val_report = load_json(val_path) if val_path.exists() else None
    if paths.test_json.exists() and not args.force:
        raise SystemExit(f"[중단] 이미 test 결과가 있습니다: {display_path(paths.test_json)}. "
                         "test는 한 번만 평가합니다. 정말 다시 하려면 --force.")

    label_set, splits = load_task(paths.data_dir, ("train", "validation", "test"))
    test = splits["test"]

    clf = load_classifier(model_dir, device=args.device)
    if clf.label_set.labels != label_set.labels:
        raise SystemExit(f"[중단] 모델 라벨 {clf.label_set.labels}와 "
                         f"labels.json {label_set.labels}의 순서가 다릅니다.")

    result = evaluate_classifier(clf, test, model_dir=model_dir,
                                 latency_samples=args.latency_samples)
    val_eff = (val_report or {}).get("efficiency", {})
    report = _common.build_report(
        model=model,
        run=run,
        kind=final_meta["kind"],
        task=args.task,
        model_dir=model_dir,
        result=result,
        train_time_sec=val_eff.get("train_time_sec"),
        peak_gpu_memory_mb_train=val_eff.get("peak_gpu_memory_mb_train"),
        config=final_meta.get("config"),
        training={"final_model": {"model": model, "run": run, "source": "config.FINAL_MODEL"}},
    )
    save_json(report, paths.test_json)

    val_row = (
        f"| validation | {val_report['metrics']['macro_f1']:.4f} | "
        f"{val_report['metrics']['accuracy']:.4f} |"
        if val_report else "| validation | - | - |"
    )
    lines = [
        f"# 최종 분류 평가 · {args.task}",
        "",
        f"- 평가 시각: {now_iso()}",
        f"- 모델: `{model}/{run}` ({final_meta['kind']}) — config.py FINAL_MODEL",
        f"- 평가 데이터: test {len(test)}건",
        "",
        "| 분할 | Macro-F1 | Accuracy |",
        "|---|---|---|",
        val_row,
        (f"| **test** | **{report['metrics']['macro_f1']:.4f}** | "
        f"**{report['metrics']['accuracy']:.4f}** |"),
        "",
        "## 성능·시간·자원 (test)",
        "",
        summary_table_md([report]),
        "## 클래스별 F1 (test)",
        "",
        per_class_table_md([report]),
        "## 혼동행렬 (test)",
        "",
        confusion_md(report["metrics"]),
        f"오답 {len(report['errors'])}건은 `{paths.test_json.name}`의 errors에 있습니다.",
        "",
    ]
    paths.test_md.write_text("\n".join(lines), encoding="utf-8")

    _common.print_summary(report)
    print(f"\n리포트: {display_path(paths.test_json)}, {display_path(paths.test_md)}")
    return report


if __name__ == "__main__":
    main()
