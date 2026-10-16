"""③ 학습해 둔 모든 (모델, 실행)을 같은 기준으로 비교해 표로 만든다.

uv run python scripts/classifier/compare.py

--task를 주지 않으면 inquiries로 돌아간다. documents를 쓰려면 --task documents.

- 후보: runs/classifier/<task>/<모델 이름>/<실행 ID>/val_report.json 이 있는 실행 전부
- 모든 후보를 이 스크립트 안에서 validation으로 다시 평가한다 (같은 장치, 같은 지연 측정 방식).
  학습 시간과 학습 GPU 피크 메모리는 학습 때 남긴 리포트 값을 쓴다.
- 결과: runs/classifier/<task>/comparison.md (비교표). 모델을 복사하거나 확정하지 않는다.
- 최종 모델은 이 표를 보고 config.py의 FINAL_MODEL에 모델 이름과 실행 ID를 직접 적는다.
  표에는 참고용 추천(macro-F1 최고값과 --tolerance 이내 중 지연 시간이 가장 짧은 실행)을 함께 보여 준다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import _common

from theseal.classifier.classifier import load_classifier
from theseal.classifier.config import (
    META_FILENAME,
    VAL_REPORT_FILENAME,
    FinalModelNotSetError,
    display_path,
    final_choice,
)
from theseal.classifier.data_loader import load_task
from theseal.classifier.metrics import (
    LATENCY_SAMPLES,
    PRIMARY_METRIC,
    confusion_md,
    evaluate_classifier,
    load_json,
    now_iso,
    per_class_table_md,
    summary_table_md,
)

DEFAULT_TOLERANCE = 0.01


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="학습한 모델/실행 비교표 만들기")
    _common.add_common_args(p)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help=f"추천 기준: macro-F1 최고값과 이 차이 이내면 같은 성능으로 보고 "
                        f"더 빠른 실행을 추천한다 (기본: {DEFAULT_TOLERANCE})")
    p.add_argument("--device", default="auto", help="사전학습 모델 평가 장치")
    p.add_argument("--latency-samples", type=int, default=LATENCY_SAMPLES)
    p.add_argument("--no-reeval", action="store_true",
                   help="다시 평가하지 않고 학습 때 리포트 값을 그대로 비교")
    return p.parse_args(argv)


def collect_candidates(paths, args, val_split) -> list[dict]:
    reports = []
    for report_path in sorted(paths.runs_dir.glob(f"*/*/{VAL_REPORT_FILENAME}")):
        model, run = report_path.parent.parent.name, report_path.parent.name
        saved = load_json(report_path)
        model_dir = paths.model_dir(model, run)
        if not (model_dir / META_FILENAME).exists():
            print(f"[건너뜀] {model}/{run}: 가중치가 없습니다 ({display_path(model_dir)})")
            continue
        if args.no_reeval:
            reports.append({**saved, "name": f"{model}/{run}", "model": model, "run": run})
            continue

        print(f"[평가] {model}/{run} ...")
        clf = load_classifier(model_dir, device=args.device)
        result = evaluate_classifier(clf, val_split, model_dir=model_dir,
                                     latency_samples=args.latency_samples)
        reports.append(_common.build_report(
            model=model,
            run=run,
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


def recommend(reports: list[dict], tolerance: float) -> tuple[dict, str]:
    """참고용 추천. 최종 결정은 사람이 config.py에 적는다."""
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
        reason = (f"{PRIMARY_METRIC} 최고값 {best:.4f}와 {tolerance} 이내인 실행({names}) 중 "
                  f"지연 시간이 가장 짧음 ({chosen['efficiency']['latency_ms_median']}ms, "
                  f"{PRIMARY_METRIC} {score(chosen):.4f})")
    return chosen, reason


def settings_summary(rep: dict) -> str:
    """실행마다 무엇을 바꿨는지 한눈에 보이도록 주요 설정만 뽑는다."""
    c = rep.get("config") or {}
    t = rep.get("training") or {}
    if rep["kind"] == "baseline":
        return (f"{c.get('analyzer')} {c.get('ngram_min')}~{c.get('ngram_max')}gram, "
                f"C={c.get('C')}, min_df={c.get('min_df')}")
    return (f"lr={c.get('lr')}, batch={c.get('batch_size')}, epochs={c.get('epochs')} "
            f"(best {t.get('best_epoch')}), max_len={c.get('max_length')}, seed={c.get('seed')}")


def config_snippet(task: str, rep: dict) -> str:
    return f'"{task}": {{"model": "{rep["model"]}", "run": "{rep["run"]}"}},'


def write_comparison_md(path: Path, task: str, reports: list[dict], recommended: dict,
                        reason: str, current: tuple[str, str] | None, tolerance: float,
                        reevaluated: bool) -> None:
    current_name = f"{current[0]}/{current[1]}" if current else None
    settings = "| 모델/실행 | 주요 설정 |\n|---|---|\n" + "\n".join(
        f"| {r['name']}{' ★' if r['name'] == current_name else ''} | {settings_summary(r)} |"
        for r in reports
    ) + "\n"
    lines = [
        f"# 분류 모델 비교 · {task}",
        "",
        f"- 생성 시각: {now_iso()}",
        f"- 평가 데이터: validation {reports[0]['num_samples']}건",
        f"- 평가 방식: {'이 스크립트에서 모든 실행을 같은 조건으로 다시 평가' if reevaluated else '학습 때 리포트 값'}",
        "- 지연 시간: 한 문장씩 넣었을 때 (워밍업 후), 처리량: validation 전체를 한 번에 넣었을 때",
        f"- 현재 최종 모델 (config.py FINAL_MODEL): {f'`{current_name}` ★' if current_name else '없음'}",
        "",
        f"## 참고 추천: `{recommended['name']}`",
        "",
        f"{reason}  ",
        f"(규칙: {PRIMARY_METRIC} 최고값과 {tolerance} 이내 → 지연 중앙값이 짧은 순 → 크기가 작은 순)",
        "",
        "이 실행으로 정하려면 `src/theseal/classifier/config.py`의 `FINAL_MODEL`에 아래처럼 적는다.",
        "",
        "```python",
        config_snippet(task, recommended),
        "```",
        "",
        "## 성능·시간·자원",
        "",
        summary_table_md(reports),
        "## 실행별 설정",
        "",
        settings,
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
        raise SystemExit(f"[중단] 비교할 실행이 없습니다. 먼저 train_*.py를 실행하세요 "
                         f"({display_path(paths.runs_dir)}).")

    reports.sort(key=lambda r: -r["metrics"][PRIMARY_METRIC])
    recommended, reason = recommend(reports, args.tolerance)
    try:
        current = final_choice(args.task)
    except FinalModelNotSetError:
        current = None

    write_comparison_md(paths.comparison_md, args.task, reports, recommended, reason, current,
                        args.tolerance, reevaluated=not args.no_reeval)

    print("\n" + summary_table_md(reports))
    print(f"참고 추천: {recommended['name']} — {reason}")
    print(f"현재 최종 모델: {f'{current[0]}/{current[1]}' if current else '없음'}")
    print("최종 모델로 정하려면 config.py의 FINAL_MODEL에:")
    print(f"    {config_snippet(args.task, recommended)}")
    print(f"리포트: {display_path(paths.comparison_md)}")
    return {"recommended": recommended["name"], "reason": reason,
            "candidates": [r["name"] for r in reports]}


if __name__ == "__main__":
    main()
