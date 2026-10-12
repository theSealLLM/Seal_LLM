"""scripts/classifier 공통: src 경로 등록, 공통 인자, 리포트 형식."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# pyproject가 package = false라서 theseal 패키지를 설치하지 않는다. src를 직접 경로에 넣는다.
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from theseal.classifier.config import (
    DATA_DIR,
    FINAL_NAME,
    RUNS_DIR,
    TASKS,
    TaskPaths,
    display_path,
    task_paths,
)
from theseal.classifier.metrics import environment_info, now_iso


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default="inquiries", choices=TASKS,
                        help="data/ 아래 주제 폴더 (기본: inquiries)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"데이터 루트 (기본: {display_path(DATA_DIR)})")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help=f"모델/리포트 저장 루트 (기본: {display_path(RUNS_DIR)})")


def paths_from_args(args: argparse.Namespace) -> TaskPaths:
    return task_paths(args.task, data_dir=args.data_dir, runs_dir=args.runs_dir)


def check_name(name: str) -> str:
    if name == FINAL_NAME:
        raise SystemExit(f"[중단] '{FINAL_NAME}'은 확정 모델용 이름이라 쓸 수 없습니다.")
    return name


def build_report(
    *,
    name: str,
    kind: str,
    task: str,
    model_dir: Path,
    result: dict,
    train_time_sec: float | None = None,
    peak_gpu_memory_mb_train: float | None = None,
    config: dict | None = None,
    training: dict | None = None,
) -> dict:
    """train_*.py, compare.py, evaluate_test.py가 모두 같은 형식의 리포트를 남긴다."""
    return {
        "name": name,
        "kind": kind,
        "task": task,
        "split": result["split"],
        "num_samples": result["num_samples"],
        "model_dir": display_path(model_dir),
        "created_at": now_iso(),
        "metrics": result["metrics"],
        "efficiency": {
            **result["efficiency"],
            "train_time_sec": train_time_sec,
            "peak_gpu_memory_mb_train": peak_gpu_memory_mb_train,
        },
        "config": config,
        "training": training,
        "environment": environment_info(),
        "errors": result["errors"],
    }


def print_summary(report: dict) -> None:
    m, e = report["metrics"], report["efficiency"]
    print(f"\n=== {report['name']} ({report['kind']}) · {report['task']} · {report['split']} "
          f"{report['num_samples']}건 ===")
    print(f"macro-F1 {m['macro_f1']:.4f} | accuracy {m['accuracy']:.4f} | 오답 {len(report['errors'])}건")
    for label, v in m["per_class"].items():
        print(f"  {label:<14} P {v['precision']:.4f}  R {v['recall']:.4f}  F1 {v['f1']:.4f}  "
              f"(n={v['support']})")
    print(f"지연 중앙값 {e['latency_ms_median']}ms (p95 {e['latency_ms_p95']}ms) | "
          f"처리량 {e['throughput_per_sec']}문장/s | 크기 {e['model_size_mb']}MB | "
          f"장치 {e['device']}")
    if e.get("train_time_sec") is not None:
        gpu = e.get("peak_gpu_memory_mb_train")
        print(f"학습 시간 {e['train_time_sec']}s" + (f" | GPU 피크(학습) {gpu}MB" if gpu else ""))
