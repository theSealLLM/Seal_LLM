"""③ 후보 모델을 같은 기준으로 비교하고 최종 모델을 확정·저장한다.

uv run python scripts/classifier/compare.py --task inquiries

- 후보: runs/classifier/<task>/<이름>/val_report.json 이 있는 폴더 전부
- 모든 후보를 이 스크립트 안에서 validation으로 다시 평가한다 (같은 장치, 같은 지연 측정 방식).
  학습 시간과 학습 GPU 피크 메모리는 학습 때 남긴 리포트 값을 쓴다.
- 선택 규칙: macro-F1 최고값과의 차이가 --tolerance 이내인 후보 중
  지연 시간(중앙값)이 가장 짧은 모델 → 같으면 크기가 작은 모델.
- 결과: runs/classifier/<task>/ 아래 comparison.md, selection.json, final_model/ (선택 모델 복사본)
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import _common

from theseal.classifier.config import META_FILENAME, VAL_REPORT_FILENAME, display_path
from theseal.classifier.data_loader import load_task
from theseal.classifier.metrics import (
    LATENCY_SAMPLES,
    PRIMARY_METRIC,
    confusion_md,
    environment_info,
    evaluate_classifier,
    load_json,
    now_iso,
    per_class_table_md,
    save_json,
    summary_table_md,
)
from theseal.classifier.predictor import load_classifier, read_meta

DEFAULT_TOLERANCE = 0.01


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="후보 모델 비교 및 최종 모델 확정")
    _common.add_common_args(p)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help=f"macro-F1 최고값과 이 차이 이내면 같은 성능으로 보고 "
                        f"더 빠른 모델을 고른다 (기본: {DEFAULT_TOLERANCE})")
    p.add_argument("--select", default=None, help="규칙 대신 이 이름의 모델을 직접 확정")
    p.add_argument("--device", default="auto", help="사전학습 모델 평가 장치")
    p.add_argument("--latency-samples", type=int, default=LATENCY_SAMPLES)
    p.add_argument("--no-reeval", action="store_true",
                   help="다시 평가하지 않고 학습 때 리포트 값을 그대로 비교")
    return p.parse_args(argv)


def collect_candidates(paths, args, val_split) -> list[dict]:
    reports = []
    for report_path in sorted(paths.runs_dir.glob(f"*/{VAL_REPORT_FILENAME}")):
        saved = load_json(report_path)
        name = report_path.parent.name
        model_dir = paths.model_dir(name)
        if not (model_dir / META_FILENAME).exists():
            print(f"[건너뜀] {name}: 모델 폴더가 없습니다 ({display_path(model_dir)})")
            continue
        if args.no_reeval:
            reports.append(saved)
            continue

        print(f"[평가] {name} ...")
        clf = load_classifier(model_dir, device=args.device)
        result = evaluate_classifier(clf, val_split, model_dir=model_dir,
                                     latency_samples=args.latency_samples)
        reports.append(_common.build_report(
            name=name,
            kind=saved["kind"],
            task=saved["task"],
            model_dir=model_dir,
            result=result,
            train_time_sec=saved["efficiency"].get("train_time_sec"),
            peak_gpu_memory_mb_train=saved["efficiency"].get("peak_gpu_memory_mb_train"),
            config=saved.get("config"),
            training=saved.get("training"),
        ))
    return reports


def select(reports: list[dict], tolerance: float, forced: str | None = None) -> tuple[dict, str]:
    if forced:
        for rep in reports:
            if rep["name"] == forced:
                return rep, f"--select로 직접 지정 ({forced})"
        raise SystemExit(f"[중단] --select {forced!r} 후보가 없습니다. "
                         f"가능: {[r['name'] for r in reports]}")

    def score(rep):
        return rep["metrics"][PRIMARY_METRIC]

    best = max(score(r) for r in reports)
    near = [r for r in reports if best - score(r) <= tolerance + 1e-9]
    chosen = min(near, key=lambda r: (
        r["efficiency"]["latency_ms_median"],
        r["efficiency"].get("model_size_mb") or math.inf,
        -score(r),
    ))
    if len(near) == 1:
        reason = f"validation {PRIMARY_METRIC}가 가장 높음 ({best:.4f})"
    else:
        names = ", ".join(r["name"] for r in near)
        reason = (f"{PRIMARY_METRIC} 최고값 {best:.4f}와 {tolerance} 이내인 후보({names}) 중 "
                  f"지연 시간이 가장 짧음 ({chosen['efficiency']['latency_ms_median']}ms, "
                  f"{PRIMARY_METRIC} {score(chosen):.4f})")
    return chosen, reason


def write_comparison_md(path: Path, task: str, reports: list[dict], chosen: dict, reason: str,
                        tolerance: float, reevaluated: bool) -> None:
    lines = [
        f"# 분류 모델 비교 · {task}",
        "",
        f"- 생성 시각: {now_iso()}",
        f"- 평가 데이터: validation {reports[0]['num_samples']}건",
        f"- 평가 방식: {'이 스크립트에서 모든 후보를 같은 조건으로 다시 평가' if reevaluated else '학습 때 리포트 값'}",
        f"- 선택 규칙: {PRIMARY_METRIC} 최고값과 {tolerance} 이내 → 지연 중앙값이 짧은 순 → 크기가 작은 순",
        "- 지연 시간: 한 문장씩 넣었을 때 (워밍업 후), 처리량: validation 전체를 한 번에 넣었을 때",
        "",
        f"## 선택: `{chosen['name']}`",
        "",
        f"{reason}",
        "",
        "## 성능·시간·자원",
        "",
        summary_table_md(reports),
        "## 클래스별 F1",
        "",
        per_class_table_md(reports),
    ]
    for rep in reports:
        lines += [f"## 혼동행렬 · {rep['name']}", "", confusion_md(rep["metrics"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    paths = _common.paths_from_args(args)

    _, splits = load_task(paths.data_dir, ("train", "validation"))
    reports = collect_candidates(paths, args, splits["validation"])
    if not reports:
        raise SystemExit(f"[중단] 비교할 후보가 없습니다. 먼저 train_*.py를 실행하세요 "
                         f"({display_path(paths.runs_dir)}).")

    reports.sort(key=lambda r: -r["metrics"][PRIMARY_METRIC])
    chosen, reason = select(reports, args.tolerance, args.select)

    source_dir = paths.model_dir(chosen["name"])
    final_dir = paths.final_dir
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(source_dir, final_dir)

    write_comparison_md(paths.comparison_md, args.task, reports, chosen, reason,
                        args.tolerance, reevaluated=not args.no_reeval)
    selection = {
        "task": args.task,
        "selected": chosen["name"],
        "kind": chosen["kind"],
        "reason": reason,
        "rule": {
            "primary_metric": PRIMARY_METRIC,
            "tolerance": args.tolerance,
            "tie_break": ["latency_ms_median", "model_size_mb"],
            "forced": args.select,
        },
        "source_model_dir": display_path(source_dir),
        "final_model_dir": display_path(final_dir),
        "final_model_meta": read_meta(final_dir),
        "validation": {
            "metrics": chosen["metrics"],
            "efficiency": chosen["efficiency"],
        },
        "candidates": [
            {
                "name": r["name"],
                "kind": r["kind"],
                "macro_f1": r["metrics"]["macro_f1"],
                "accuracy": r["metrics"]["accuracy"],
                "latency_ms_median": r["efficiency"]["latency_ms_median"],
                "model_size_mb": r["efficiency"].get("model_size_mb"),
                "train_time_sec": r["efficiency"].get("train_time_sec"),
            }
            for r in reports
        ],
        "selected_at": now_iso(),
        "environment": environment_info(),
    }
    save_json(selection, paths.selection_json)

    print("\n" + summary_table_md(reports))
    print(f"선택: {chosen['name']} — {reason}")
    print(f"확정 모델: {display_path(final_dir)}")
    print(f"리포트: {display_path(paths.comparison_md)}, {display_path(paths.selection_json)}")
    return selection


if __name__ == "__main__":
    main()
