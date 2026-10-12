"""두 모델을 '같은 기준'으로 평가하는 공통 함수.

베이스라인과 사전학습 모델 모두 evaluate_classifier() 하나로만 평가한다.
분류기는 아래 속성/메서드만 있으면 된다.

- label_set: LabelSet
- device: str            ("cpu", "cuda", "cuda:0", "mps" ...)
- predict_proba(texts) -> np.ndarray (문장 수, 라벨 수), 열 순서는 label_set.labels
- describe() -> dict     (num_parameters 등 모델 정보)
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data_loader import LabelSet, Split

LATENCY_SAMPLES = 100
LATENCY_WARMUP = 5
PRIMARY_METRIC = "macro_f1"


def compute_metrics(y_true: list[int], y_pred: list[int], label_set: LabelSet) -> dict:
    label_ids = list(range(len(label_set)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    return {
        "accuracy": _r(accuracy_score(y_true, y_pred)),
        "macro_f1": _r(float(np.mean(f1))),
        "macro_precision": _r(float(np.mean(precision))),
        "macro_recall": _r(float(np.mean(recall))),
        "per_class": {
            label: {
                "precision": _r(precision[i]),
                "recall": _r(recall[i]),
                "f1": _r(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(label_set.labels)
        },
        # 행 = 정답, 열 = 예측
        "confusion_matrix": {"labels": list(label_set.labels), "matrix": matrix.tolist()},
    }


def measure_latency(clf, texts: list[str], *, warmup: int = LATENCY_WARMUP) -> dict:
    """한 문장씩 넣었을 때의 지연 시간(ms). 서비스에서 문의 하나를 분류하는 상황을 가정한다."""
    if not texts:
        raise ValueError("지연 시간을 잴 문장이 없습니다.")
    for text in texts[:warmup]:
        clf.predict_proba([text])
    durations = []
    for text in texts:
        start = time.perf_counter()
        clf.predict_proba([text])
        durations.append((time.perf_counter() - start) * 1000)
    durations.sort()
    p95_index = min(len(durations) - 1, round(0.95 * (len(durations) - 1)))
    return {
        "latency_ms_median": _r(statistics.median(durations), 3),
        "latency_ms_p95": _r(durations[p95_index], 3),
        "latency_samples": len(durations),
    }


def evaluate_classifier(
    clf,
    split: Split,
    *,
    model_dir: Path | None = None,
    latency_samples: int = LATENCY_SAMPLES,
) -> dict:
    """정확도 지표 + 추론 비용 + 오답 목록을 한 번에 계산한다."""
    label_set: LabelSet = clf.label_set
    latency = measure_latency(clf, split.texts[:latency_samples])

    peak = _GpuPeak(clf.device)
    with peak:
        start = time.perf_counter()
        probs = np.asarray(clf.predict_proba(split.texts))
        batch_sec = time.perf_counter() - start

    pred_ids = probs.argmax(axis=1).tolist()
    true_ids = label_set.encode(split.labels)
    metrics = compute_metrics(true_ids, pred_ids, label_set)

    errors = [
        {
            "id": split.ids[i],
            "text": split.texts[i],
            "label": split.labels[i],
            "pred": label_set.labels[p],
            "confidence": _r(float(probs[i, p])),
        }
        for i, p in enumerate(pred_ids)
        if p != true_ids[i]
    ]

    efficiency = {
        "device": str(clf.device),
        **latency,
        "throughput_per_sec": _r(len(split) / batch_sec, 1) if batch_sec > 0 else None,
        "peak_gpu_memory_mb_infer": peak.mb,
        "model_size_mb": _r(dir_size_mb(model_dir), 2) if model_dir else None,
        **clf.describe(),
    }
    return {"split": split.name, "num_samples": len(split), "metrics": metrics,
            "efficiency": efficiency, "errors": errors}


def dir_size_mb(path: Path) -> float:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1024**2
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**2


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("numpy", "scikit-learn", "torch", "transformers"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


# ---------- 마크다운 표 ----------

def summary_table_md(reports: list[dict]) -> str:
    header = (
        "| 모델 | 종류 | Macro-F1 | Accuracy | 학습 시간(s) | 지연 중앙값(ms) | 지연 p95(ms) "
        "| 처리량(문장/s) | 크기(MB) | 파라미터 수 | GPU 피크 학습/추론(MB) | 장치 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for rep in reports:
        m, e = rep["metrics"], rep["efficiency"]
        gpu = f"{_fmt(e.get('peak_gpu_memory_mb_train'))} / {_fmt(e.get('peak_gpu_memory_mb_infer'))}"
        rows.append(
            f"| {rep['name']} | {rep['kind']} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} "
            f"| {_fmt(e.get('train_time_sec'))} | {_fmt(e.get('latency_ms_median'))} "
            f"| {_fmt(e.get('latency_ms_p95'))} | {_fmt(e.get('throughput_per_sec'))} "
            f"| {_fmt(e.get('model_size_mb'))} | {_fmt_int(e.get('num_parameters'))} "
            f"| {gpu} | {e.get('device')} |"
        )
    return header + "\n".join(rows) + "\n"


def per_class_table_md(reports: list[dict]) -> str:
    labels = reports[0]["metrics"]["confusion_matrix"]["labels"]
    header = "| 모델 | " + " | ".join(f"{label} F1" for label in labels) + " |\n"
    header += "|---|" + "---|" * len(labels) + "\n"
    rows = []
    for rep in reports:
        per_class = rep["metrics"]["per_class"]
        cells = " | ".join(f"{per_class[label]['f1']:.4f}" for label in labels)
        rows.append(f"| {rep['name']} | {cells} |")
    return header + "\n".join(rows) + "\n"


def confusion_md(metrics: dict) -> str:
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    out = "| 정답 \\ 예측 | " + " | ".join(labels) + " |\n"
    out += "|---|" + "---|" * len(labels) + "\n"
    for label, row in zip(labels, matrix):
        out += f"| {label} | " + " | ".join(str(v) for v in row) + " |\n"
    return out


class _GpuPeak:
    """CUDA 장치일 때 구간 안의 최대 GPU 메모리(MB)를 잰다. 그 외 장치는 None."""

    def __init__(self, device) -> None:
        self.device = str(device)
        self.mb: float | None = None

    def __enter__(self):
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
            self.mb = _r(torch.cuda.max_memory_allocated() / 1024**2, 1)


def _r(value, digits: int = 4) -> float:
    return round(float(value), digits)


def _fmt(value) -> str:
    return "-" if value is None else f"{value:g}"


def _fmt_int(value) -> str:
    return "-" if value is None else f"{int(value):,}"
