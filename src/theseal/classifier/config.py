"""분류 파이프라인의 경로, 시드, 하이퍼파라미터 기본값."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "classifier"
REPORTS_DIR = PROJECT_ROOT / "reports" / "classifier"

# data/ 아래 주제 폴더 이름 (README 참고: 둘 중 하나를 골라 사용)
TASKS = ("inquiries", "documents")
SPLITS = ("train", "validation", "test")

SEED = 42

# 저장된 모델 폴더마다 들어가는 메타 파일 (predictor가 모델 종류를 판단할 때 사용)
META_FILENAME = "classifier_meta.json"
# compare.py가 확정한 모델을 복사해 두는 폴더 이름
FINAL_NAME = "final"


@dataclass
class BaselineConfig:
    """TF-IDF + 로지스틱 회귀 설정."""

    analyzer: str = "char_wb"  # 형태소 분석기 없이 한국어를 다루기 위해 문자 n-gram 사용
    ngram_min: int = 1
    ngram_max: int = 3
    min_df: int = 1
    sublinear_tf: bool = True
    C: float = 10.0
    max_iter: int = 2000
    seed: int = SEED


@dataclass
class BertConfig:
    """사전학습 모델 파인튜닝 설정."""

    model_name: str = "klue/roberta-base"
    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 10
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 3  # validation macro-F1이 이 에폭 수만큼 안 오르면 중단 (0이면 끄기)
    precision: str = "auto"  # auto | fp32 | fp16 | bf16 (GPU에서만 적용)
    local_files_only: bool = False  # True면 HF Hub에서 내려받지 않고 캐시/로컬 경로만 사용
    seed: int = SEED


@dataclass(frozen=True)
class TaskPaths:
    data_dir: Path
    models_dir: Path
    reports_dir: Path

    def model_dir(self, name: str) -> Path:
        return self.models_dir / name

    @property
    def final_dir(self) -> Path:
        return self.models_dir / FINAL_NAME

    def val_report(self, name: str) -> Path:
        return self.reports_dir / f"val_{name}.json"

    @property
    def comparison_md(self) -> Path:
        return self.reports_dir / "comparison.md"

    @property
    def selection_json(self) -> Path:
        return self.reports_dir / "selection.json"

    @property
    def test_json(self) -> Path:
        return self.reports_dir / "test_final.json"

    @property
    def test_md(self) -> Path:
        return self.reports_dir / "test_final.md"


def task_paths(
    task: str,
    *,
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> TaskPaths:
    if task not in TASKS:
        raise ValueError(f"알 수 없는 task: {task!r} (가능: {', '.join(TASKS)})")
    return TaskPaths(
        data_dir=Path(data_dir or DATA_DIR) / task,
        models_dir=Path(models_dir or MODELS_DIR) / task,
        reports_dir=Path(reports_dir or REPORTS_DIR) / task,
    )


def display_path(path: Path) -> str:
    """리포트에 남길 경로. 프로젝트 안이면 상대 경로로 적어 OS와 무관하게 읽히게 한다."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
