"""분류 파이프라인의 경로, 시드, 하이퍼파라미터 기본값."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
# 학습/평가 결과: runs/classifier/<task>/<모델 이름>/<실행 ID>/ 아래에 모델(model/)과 리포트를 함께 둔다
RUNS_DIR = PROJECT_ROOT / "runs" / "classifier"
# fetch_models.py가 사전학습 모델을 받아 두는 곳
MODELS_DIR = PROJECT_ROOT / "models"

# 파인튜닝 후보 (train_classifier_model.py의 번호 메뉴 순서이자 fetch_models.py의 목록)
MODEL_CANDIDATES = (
    "klue/roberta-base",
    "klue/bert-base",
    "monologg/koelectra-small-v3-discriminator",
)

# data/ 아래 주제 폴더 이름 (README 참고: 둘 중 하나를 골라 사용)
TASKS = ("inquiries", "documents")
# task를 따로 주지 않으면 항상 이 주제를 쓴다
DEFAULT_TASK = "inquiries"
SPLITS = ("train", "validation", "test")

SEED = 42

# 저장된 모델 폴더마다 들어가는 메타 파일 (classifier가 모델 종류를 판단할 때 사용)
META_FILENAME = "classifier_meta.json"
# 실행 폴더 안에서 모델 파일을 두는 하위 폴더 이름
MODEL_SUBDIR = "model"
VAL_REPORT_FILENAME = "val_report.json"
# 실행 ID = 학습을 시작한 시각 (분 단위). Windows 폴더 이름에 못 쓰는 ':'는 쓰지 않는다
RUN_ID_FORMAT = "%Y%m%d-%H%M"

# 최종 모델: comparison.md를 보고 task별로 모델 이름과 실행 ID를 직접 적는다.
# 예) "inquiries": {"model": "roberta-base", "run": "20261013-1411"}
#     → runs/classifier/inquiries/roberta-base/20261013-1411/model/ 을 그대로 사용 (복사하지 않음)
FINAL_MODEL: dict[str, dict[str, str] | None] = {
    "inquiries": {"model": "baseline", "run": "20261013-1057"},
    "documents": None,
}


class FinalModelNotSetError(LookupError):
    """config.py의 FINAL_MODEL에 해당 task의 최종 모델이 정해지지 않았을 때."""


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
    """runs/classifier/<task>/ 아래 구조.

    <모델 이름>/<실행 ID>/model/            학습된 가중치 (gitignore)
    <모델 이름>/<실행 ID>/val_report.json   validation 결과 + 그때 쓴 설정
    comparison.md, test_report.json, test_report.md
    """

    task: str
    data_dir: Path
    runs_dir: Path

    def model_root(self, model: str) -> Path:
        return self.runs_dir / model

    def run_dir(self, model: str, run: str) -> Path:
        return self.model_root(model) / run

    def model_dir(self, model: str, run: str) -> Path:
        return self.run_dir(model, run) / MODEL_SUBDIR

    def val_report(self, model: str, run: str) -> Path:
        return self.run_dir(model, run) / VAL_REPORT_FILENAME

    @property
    def final_model_dir(self) -> Path:
        """config.py의 FINAL_MODEL이 가리키는 모델 폴더."""
        return self.model_dir(*final_choice(self.task))

    @property
    def comparison_md(self) -> Path:
        return self.runs_dir / "comparison.md"

    @property
    def test_json(self) -> Path:
        return self.runs_dir / "test_report.json"

    @property
    def test_md(self) -> Path:
        return self.runs_dir / "test_report.md"


def task_paths(
    task: str = DEFAULT_TASK,
    *,
    data_dir: Path | None = None,
    runs_dir: Path | None = None,
) -> TaskPaths:
    if task not in TASKS:
        raise ValueError(f"알 수 없는 task: {task!r} (가능: {', '.join(TASKS)})")
    return TaskPaths(
        task=task,
        data_dir=Path(data_dir or DATA_DIR) / task,
        runs_dir=Path(runs_dir or RUNS_DIR) / task,
    )


def final_choice(task: str = DEFAULT_TASK) -> tuple[str, str]:
    """FINAL_MODEL에 적힌 (모델 이름, 실행 ID)."""
    choice = FINAL_MODEL.get(task)
    if not choice or not choice.get("model") or not choice.get("run"):
        raise FinalModelNotSetError(
            f"config.py의 FINAL_MODEL[{task!r}]에 최종 모델이 없습니다. "
            "comparison.md를 보고 {\"model\": ..., \"run\": ...}을 적어 주세요."
        )
    return choice["model"], choice["run"]


def safe_name(text: str) -> str:
    """폴더 이름으로 쓸 수 있게 영문, 숫자, '.', '_', '-' 외의 글자를 '-'로 바꾼다."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "model"


def new_run_id(tag: str | None = None, now: datetime | None = None) -> str:
    """학습 시작 시각으로 실행 ID를 만든다. 예) 20261013-1411, 태그가 있으면 20261013-1411_lr2e5."""
    run = (now or datetime.now()).strftime(RUN_ID_FORMAT)
    return f"{run}_{safe_name(tag)}" if tag else run


def local_model_dir(repo_id: str) -> Path:
    """fetch_models.py가 내려받아 두는 위치: models/<저장소 이름의 /를 __로 바꾼 것>."""
    return MODELS_DIR / repo_id.replace("/", "__")


def display_path(path: Path) -> str:
    """리포트에 남길 경로. 프로젝트 안이면 상대 경로로 적어 OS와 무관하게 읽히게 한다."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
