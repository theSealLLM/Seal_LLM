"""④ 확정한 모델로 test를 평가한다. 모델과 조건을 정한 뒤 딱 한 번만 실행한다.

uv run python scripts/classifier/evaluate_test.py --task inquiries

- compare.py가 만든 runs/classifier/<task>/selection.json과 final_model/이 있어야 실행된다.
- 이미 test_report.json이 있으면 다시 실행하지 않는다 (--force로만 덮어쓰기).
"""

from __future__ import annotations

import argparse

import _common

from theseal.classifier.config import display_path
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
from theseal.classifier.predictor import load_classifier, read_meta


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

    if not paths.selection_json.exists():
        raise SystemExit(f"[중단] {display_path(paths.selection_json)}가 없습니다. "
                         "compare.py로 모델을 먼저 확정하세요. test는 확정 전에 보지 않습니다.")
    selection = load_json(paths.selection_json)
    if not paths.final_dir.exists():
        raise SystemExit(f"[중단] 확정 모델 폴더가 없습니다: {display_path(paths.final_dir)}. "
                         "모델은 git에 올라가지 않으니 이 컴퓨터에서 compare.py를 다시 실행하세요.")
    final_meta = read_meta(paths.final_dir)
    if final_meta["name"] != selection["selected"]:
        raise SystemExit(f"[중단] final 모델({final_meta['name']})과 selection.json의 선택"
                         f"({selection['selected']})이 다릅니다. compare.py를 다시 실행하세요.")
    if paths.test_json.exists() and not args.force:
        raise SystemExit(f"[중단] 이미 test 결과가 있습니다: {display_path(paths.test_json)}. "
                         "test는 한 번만 평가합니다. 정말 다시 하려면 --force.")

    label_set, splits = load_task(paths.data_dir, ("train", "validation", "test"))
    test = splits["test"]

    clf = load_classifier(paths.final_dir, device=args.device)
    if clf.label_set.labels != label_set.labels:
        raise SystemExit(f"[중단] 모델 라벨 {clf.label_set.labels}와 "
                         f"labels.json {label_set.labels}의 순서가 다릅니다.")

    result = evaluate_classifier(clf, test, model_dir=paths.final_dir,
                                 latency_samples=args.latency_samples)
    val_eff = selection["validation"]["efficiency"]
    report = _common.build_report(
        name=selection["selected"],
        kind=selection["kind"],
        task=args.task,
        model_dir=paths.final_dir,
        result=result,
        train_time_sec=val_eff.get("train_time_sec"),
        peak_gpu_memory_mb_train=val_eff.get("peak_gpu_memory_mb_train"),
        config=final_meta.get("config"),
        training={"selection": {k: selection[k] for k in ("selected", "reason", "rule",
                                                           "selected_at")}},
    )
    save_json(report, paths.test_json)

    val_metrics = selection["validation"]["metrics"]
    lines = [
        f"# 최종 분류 평가 · {args.task}",
        "",
        f"- 평가 시각: {now_iso()}",
        f"- 모델: `{selection['selected']}` ({selection['kind']})",
        f"- 선택 이유: {selection['reason']}",
        f"- 평가 데이터: test {len(test)}건",
        "",
        "| 분할 | Macro-F1 | Accuracy |",
        "|---|---|---|",
        f"| validation | {val_metrics['macro_f1']:.4f} | {val_metrics['accuracy']:.4f} |",
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
