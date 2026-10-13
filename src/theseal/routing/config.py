"""분류 후처리 설정. 값은 .env에서 읽는다 (theseal.llm.config와 같은 방식).

경로 규칙 값의 우선순위: .env에 적은 값 > 모델별 보정 파일(route_rule.json) > 예시 값
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from theseal.classifier.config import DEFAULT_TASK, task_paths
from theseal.llm.config import ENV_FILE, config_from_env

CALIBRATION_FILENAME = "route_rule.json"
EXAMPLE_PARAMS = {"temperature": 1.0, "tau": 0.60, "tau_low": 0.45, "delta": 0.15}
VERIFY_MODES = ("off", "uncertain", "all")
UNCERTAIN_ACTIONS = ("top1", "unknown")


@dataclass
class RoutingConfig:
    # ② 경로 규칙
    route_rule_enabled: bool = True
    # 보정 파일을 고르는 방법: auto면 쓰는 분류 모델 폴더(<모델>/<실행 ID>/model) 옆의 route_rule.json,
    # "<모델>/<실행 ID>"를 적으면 그 실행의 보정 파일
    route_calibration: str = "auto"
    route_task: str = DEFAULT_TASK
    route_temperature: float | None = None  # 아래 네 값은 비우면 보정 파일 → 예시 값
    route_tau: float | None = None
    route_tau_low: float | None = None
    route_delta: float | None = None
    route_uncertain_action: str = "top1"  # 확신 부족 · 접전일 때: top1(1등 라벨) | unknown

    # ③ LLM 검증
    verify_mode: str = "uncertain"  # off | uncertain(경로 규칙이 확신이 아닌 것만) | all
    verify_provider: str = "local"
    verify_timeout_sec: float = 10  # 넘으면 검증 없이 분류 라벨을 그대로 쓴다

    def __post_init__(self) -> None:
        if self.verify_mode not in VERIFY_MODES:
            raise ValueError(f"VERIFY_MODE는 {VERIFY_MODES} 중 하나여야 합니다: {self.verify_mode!r}")
        if self.route_uncertain_action not in UNCERTAIN_ACTIONS:
            raise ValueError(
                f"ROUTE_UNCERTAIN_ACTION은 {UNCERTAIN_ACTIONS} 중 하나여야 합니다: "
                f"{self.route_uncertain_action!r}")

    @classmethod
    def from_env(cls, env_file: Path | None = ENV_FILE) -> RoutingConfig:
        return config_from_env(cls, env_file)


def calibration_path(model_dir: Path) -> Path:
    """분류 모델 폴더(runs/classifier/<task>/<모델>/<실행 ID>/model) 옆의 route_rule.json.
    보정값은 학습한 가중치마다 다르므로 실행 폴더에 둔다 (scripts/classifier/calibrate_route_rule.py가 만든다)."""
    return Path(model_dir).parent / CALIBRATION_FILENAME


def named_calibration_path(name: str, task: str = DEFAULT_TASK, runs_dir: Path | None = None) -> Path:
    """name = "<모델>/<실행 ID>" → runs/classifier/<task>/<모델>/<실행 ID>/route_rule.json"""
    model, sep, run = name.partition("/")
    if not sep or not model or not run:
        raise ValueError(f"ROUTE_CALIBRATION은 auto 또는 '<모델>/<실행 ID>' 형식이어야 합니다: {name!r}")
    return task_paths(task, runs_dir=runs_dir).run_dir(model, run) / CALIBRATION_FILENAME


def load_calibration(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def calibration_name(path: Path) -> str:
    """보정 파일이 어느 모델 실행 것인지: <모델>/<실행 ID>"""
    return f"{path.parent.parent.name}/{path.parent.name}"


def resolve_params(config: RoutingConfig, model_dir: Path | None = None) -> dict:
    """경로 규칙에 넣을 값: .env > 보정 파일 > 예시 값. 어디서 왔는지(source)도 함께 돌려준다.
    model_dir: 쓰는 분류 모델 폴더 (Classifier.model_dir). ROUTE_CALIBRATION=auto일 때 보정 파일을 찾는 데 쓴다"""
    if config.route_calibration == "auto":
        path = calibration_path(model_dir) if model_dir else None
    else:
        path = named_calibration_path(config.route_calibration, config.route_task)
    cal = load_calibration(path) if path else {}
    file_values = {"temperature": cal.get("temperature"), **cal.get("thresholds", {})}
    env_values = {
        "temperature": config.route_temperature,
        "tau": config.route_tau,
        "tau_low": config.route_tau_low,
        "delta": config.route_delta,
    }
    params, source = {}, {}
    for key, example in EXAMPLE_PARAMS.items():
        if env_values[key] is not None:
            params[key], source[key] = env_values[key], "env"
        elif file_values.get(key) is not None:
            params[key], source[key] = file_values[key], "calibration"
        else:
            params[key], source[key] = example, "example"
    return {"params": params, "source": source, "calibration": calibration_name(path) if cal else None}


@lru_cache
def get_routing_config() -> RoutingConfig:
    return RoutingConfig.from_env()
