"""⑥ 학습한 분류 모델마다 경로 규칙 보정값(T · tau · tau_low · delta)을 정해 저장한다 (재학습 없음).

uv run python scripts/classifier/calibrate_route_rule.py --all
uv run python scripts/classifier/calibrate_route_rule.py --model roberta-base --run 20261013-1029
uv run python scripts/classifier/calibrate_route_rule.py --model roberta-base      # 가장 최근 실행
uv run python scripts/classifier/calibrate_route_rule.py --final                   # config.py FINAL_MODEL
uv run python scripts/classifier/calibrate_route_rule.py --model-dir <모델 폴더>    # 위 구조 밖의 폴더

- 대상: runs/classifier/<task>/<모델>/<실행 ID>/model 이 있는 모델 (가중치는 git에 없으므로 학습했거나 share_model.py pull로 받은 것)
- 결과: src/theseal/prompt/routing_config.py의 ROUTE_CALIBRATIONS에 넣을 줄을 출력한다 (직접 붙여 넣는다. 서비스는 이 값만 읽는다).
  키는 모델 폴더 이름(<모델>/<실행 ID>, 예전 구조는 <모델>). 보정값은 가중치마다 다르므로 모델마다 따로 둔다.
  모델 폴더 옆 route_rule.json에는 평가 수치까지 담은 기록을 남긴다 (서비스는 읽지 않는다)
- 값은 실제 문의(validation)로만 정한다. 범위 밖 글(--ood-task의 validation)은 평가에만 쓴다
  T            보정 후 실제 문의 p1의 (1-keep) 분위수가 --target-tau-low가 되는 값
  tau_low      보정 후 p1의 (1-keep) 분위수 (= target-tau-low)
  tau          보정 후 p1의 10% 분위수
  delta        보정 후 margin의 (1-keep) 분위수
- 평가는 group_id 기준 2-fold (한쪽으로 값을 정하고 다른 쪽으로 평가, 서로 바꿔 평균). test.csv는 쓰지 않는다
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import _common
import numpy as np

from theseal.classifier.classifier import Classifier
from theseal.classifier.config import SEED, TASKS, FinalModelNotSetError, display_path, task_paths
from theseal.classifier.data_loader import load_task
from theseal.classifier.metrics import environment_info, now_iso, save_json
from theseal.prompt.routing_config import CALIBRATION_FILENAME, calibration_key, calibration_path

KEEP = 0.975  # 실제 문의 중 unknown으로 잘못 버려도 되는 비율 2.5%
TARGET_TAU_LOW = 0.45
TAU_QUANTILE = 0.10
EXAMPLE = {"temperature": 1.0, "tau": 0.60, "tau_low": 0.45, "delta": 0.15}
MIN_PROB = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="경로 규칙 보정값 계산 · 저장")
    _common.add_common_args(p)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="runs/classifier/<task>/<모델> 하나 (--run이 없으면 가장 최근 실행)")
    who.add_argument("--final", action="store_true", help="config.py FINAL_MODEL의 최종 모델")
    who.add_argument("--all", action="store_true", help="model/ 폴더가 있는 실행 전부")
    who.add_argument("--model-dir", type=Path, help="classifier_meta.json이 있는 모델 폴더를 직접 지정")
    p.add_argument("--run", default=None, help="--model과 함께: <실행 ID> (예: 20261013-1029)")
    p.add_argument("--ood-task", default="documents", choices=TASKS,
                   help="평가에만 쓰는 범위 밖 글 (기본: documents validation)")
    p.add_argument("--keep", type=float, default=KEEP, help=f"실제 문의 통과 비율 (기본: {KEEP})")
    p.add_argument("--target-tau-low", type=float, default=TARGET_TAU_LOW,
                   help=f"보정 후 tau_low가 될 값 (기본: {TARGET_TAU_LOW})")
    p.add_argument("--device", default=None, help="auto | cuda | cpu (기본: 자동)")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 계산 (배치)

def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def p1_margin(z: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.sort(softmax(z, t), axis=1)
    return p[:, -1], p[:, -1] - p[:, -2]


def fit_temperature(z: np.ndarray, target: float, q: float) -> float:
    """보정 후 p1의 q 분위수가 target이 되는 T (p1은 T가 커질수록 작아진다)."""
    lo, hi = np.log(1e-3), np.log(1e4)
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.quantile(p1_margin(z, np.exp(mid))[0], q) > target:
            lo = mid
        else:
            hi = mid
    return float(np.exp((lo + hi) / 2))


def fit(z: np.ndarray, keep: float, target: float) -> dict:
    t = fit_temperature(z, target, 1 - keep)
    p1, margin = p1_margin(z, t)
    return {
        "temperature": t,
        "tau": float(np.quantile(p1, TAU_QUANTILE)),
        "tau_low": float(np.quantile(p1, 1 - keep)),
        "delta": float(np.quantile(margin, 1 - keep)),
    }


def routes(z: np.ndarray, prm: dict) -> np.ndarray:
    """theseal.routing.route_rule.decide와 같은 판정 순서 (배치)"""
    p1, margin = p1_margin(z, prm["temperature"])
    return np.where(p1 < prm["tau_low"], "unknown",
                    np.where(margin < prm["delta"], "ambiguous",
                             np.where(p1 >= prm["tau"], "confident", "low_confidence")))


def group_folds(group_ids: list[str]) -> list[np.ndarray]:
    groups = sorted(set(group_ids))
    random.Random(SEED).shuffle(groups)
    half = set(groups[: len(groups) // 2])
    a = np.array([g in half for g in group_ids])
    return [np.where(a)[0], np.where(~a)[0]]


def evaluate(z_in: np.ndarray, y_in: np.ndarray, z_ood: np.ndarray, folds, keep, target) -> dict:
    """2-fold로 세 방식을 비교한다: 예시 값(T=1) · 단일 기준값(보정 없음) · 보정값(이 스크립트)."""
    out = {}
    for method in ("example", "single_threshold", "calibrated"):
        agg = {k: [] for k in ("in_unknown", "in_uncertain", "in_confident_acc", "ood_unknown",
                               "ood_confident")}
        for k in range(2):
            fit_idx, ev_idx = folds[k], folds[1 - k]
            if method == "example":
                prm = EXAMPLE
            elif method == "single_threshold":
                p1, _ = p1_margin(z_in[fit_idx], 1.0)
                prm = {"temperature": 1.0, "tau": -1.0, "tau_low": float(np.quantile(p1, 1 - keep)),
                       "delta": -1.0}
            else:
                prm = fit(z_in[fit_idx], keep, target)
            r_in, r_ood = routes(z_in[ev_idx], prm), routes(z_ood, prm)
            conf = r_in == "confident"
            pred = z_in[ev_idx].argmax(axis=1)
            agg["in_unknown"].append((r_in == "unknown").mean())
            agg["in_uncertain"].append(np.isin(r_in, ["ambiguous", "low_confidence"]).mean())
            agg["in_confident_acc"].append(
                (pred[conf] == y_in[ev_idx][conf]).mean() if conf.any() else np.nan)
            agg["ood_unknown"].append((r_ood == "unknown").mean())
            agg["ood_confident"].append((r_ood == "confident").mean())
        out[method] = {k: round(float(np.nanmean(v)), 4) for k, v in agg.items()}
    return out


def config_values(report: dict) -> dict:
    """prompt/routing_config.py ROUTE_CALIBRATIONS에 넣을 값."""
    return {"temperature": report["temperature"], **report["thresholds"]}


def config_line(report: dict) -> str:
    return f'    {json.dumps(report["name"], ensure_ascii=False)}: {json.dumps(config_values(report))},'


# ---------------------------------------------------------------- 모델 하나

def calibrate_one(model_dir: Path, args: argparse.Namespace, data: dict) -> dict:
    clf = Classifier(model_dir, device=args.device)
    path = calibration_path(model_dir)
    name = calibration_key(model_dir, args.task, args.runs_dir)
    labels = list(clf.labels)

    def logits(texts: list[str]) -> np.ndarray:
        return np.log(np.clip(np.asarray(clf.model.predict_proba(texts), dtype=np.float64), MIN_PROB, None))

    z_in, z_ood = logits(data["in"].texts), logits(data["ood"].texts)
    y_in = np.array([labels.index(label) for label in data["in"].labels])
    folds = group_folds(data["in"].group_ids)

    prm = fit(z_in, args.keep, args.target_tau_low)  # 서비스용: 실제 문의 전체로 다시 정한다
    evaluation = evaluate(z_in, y_in, z_ood, folds, args.keep, args.target_tau_low)
    report = {
        "name": name,
        "kind": clf.meta["kind"],
        "task": args.task,
        "model_dir": display_path(model_dir),
        "created_at": now_iso(),
        "temperature": round(prm["temperature"], 6),
        "thresholds": {k: round(prm[k], 6) for k in ("tau", "tau_low", "delta")},
        "method": {
            "temperature": f"보정 후 실제 문의 p1의 {1 - args.keep:.3g} 분위수 = {args.target_tau_low}",
            "tau": f"보정 후 p1의 {TAU_QUANTILE} 분위수",
            "tau_low": f"보정 후 p1의 {1 - args.keep:.3g} 분위수",
            "delta": f"보정 후 margin의 {1 - args.keep:.3g} 분위수",
            "fit_data": f"{args.task} validation {len(z_in)}건 (실제 문의만)",
            "ood_data": f"{args.ood_task} validation {len(z_ood)}건 (평가에만 사용)",
        },
        "evaluation": evaluation,
        "environment": environment_info(),
    }
    save_json(report, path)

    e = evaluation
    print(f"\n=== {name} ({report['kind']}) · T={report['temperature']:.4g} "
          f"tau={report['thresholds']['tau']:.4f} tau_low={report['thresholds']['tau_low']:.4f} "
          f"delta={report['thresholds']['delta']:.4f}")
    print(f"  {'방식':18} 실제 문의 버림  확신 아님  | 범위 밖 unknown  범위 밖 확신 통과")
    for method, title in (("example", "예시 값 (T=1)"), ("single_threshold", "단일 기준값"),
                          ("calibrated", "보정값")):
        m = e[method]
        print(f"  {title:18} {m['in_unknown']:>12.1%} {m['in_uncertain']:>10.1%}  | "
              f"{m['ood_unknown']:>15.1%} {m['ood_confident']:>16.1%}")
    print(f"  기록 → {display_path(path)}")
    print(f"  config.py 항목 → {config_line(report).strip()}")
    return report


def main(argv: list[str] | None = None) -> list[dict]:
    args = parse_args(argv)
    paths = _common.paths_from_args(args)
    if args.run and not args.model:
        raise SystemExit("[중단] --run은 --model과 함께 씁니다.")
    if args.all:
        model_dirs = sorted(d for d in paths.runs_dir.glob("*/*/model") if d.is_dir())
    elif args.final:
        try:
            model_dirs = [paths.final_model_dir]
        except FinalModelNotSetError as e:
            raise SystemExit(f"[중단] {e}") from None
    elif args.model_dir:
        model_dirs = [args.model_dir]
    else:
        model = _common.check_name(args.model)
        if args.run:
            model_dirs = [paths.model_dir(model, args.run)]
        else:  # 실행 ID는 시각이라 이름순 마지막이 가장 최근
            runs = sorted(d for d in paths.model_root(model).glob("*/model") if d.is_dir())
            model_dirs = runs[-1:] or [paths.model_root(model) / "<실행 ID>" / "model"]
    missing = [d for d in model_dirs if not d.exists()]
    if not model_dirs or missing:
        raise SystemExit(f"[중단] 학습된 모델이 없습니다: {', '.join(map(display_path, missing)) or paths.runs_dir}"
                         " (train_*.py로 학습하거나 share_model.py pull로 받으세요)")

    _, splits = load_task(paths.data_dir, ("validation",))
    _, ood = load_task(task_paths(args.ood_task, data_dir=args.data_dir).data_dir, ("validation",))
    data = {"in": splits["validation"], "ood": ood["validation"]}
    print(f"[data] {args.task} validation={len(data['in'])} · 범위 밖 {args.ood_task} "
          f"validation={len(data['ood'])} · 보정 파일 이름 {CALIBRATION_FILENAME}")
    reports = [calibrate_one(d, args, data) for d in model_dirs]
    print("\n서비스에 쓰려면 아래 줄을 src/theseal/prompt/routing_config.py의 ROUTE_CALIBRATIONS에 넣으세요 (같은 키는 바꾸기):")
    for r in reports:
        print(config_line(r))
    return reports


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main()
