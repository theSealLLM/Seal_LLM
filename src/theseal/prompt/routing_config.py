"""분류 후처리 설정. 켜고 끄기 · 검증 설정은 .env에서 읽는다 (theseal.llm.config와 같은 방식).

경로 규칙 값의 우선순위: .env 개별 값(ROUTE_TAU 등) > 이 파일의 모델별 보정값(ROUTE_CALIBRATIONS) > 예시 값
보정 스크립트가 남기는 route_rule.json은 기록용이고 서비스는 읽지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from theseal.classifier.config import DEFAULT_TASK, MODEL_SUBDIR, display_path, task_paths
from theseal.common.config import ENV_FILE, config_from_env

CALIBRATION_FILENAME = "route_rule.json"
EXAMPLE_PARAMS = {"temperature": 1.0, "tau": 0.60, "tau_low": 0.45, "delta": 0.15}
VERIFY_MODES = ("off", "uncertain", "all")
UNCERTAIN_ACTIONS = ("top1", "unknown")

# 모델별 경로 규칙 보정값. scripts/classifier/calibrate_route_rule.py가 출력한 줄을 붙여 넣는다.
# 키: runs/classifier/<task>/ 아래 모델 폴더 이름 (<모델>/<실행 ID>, 예전 구조는 <모델>) → calibration_key()
# 보정값은 그 가중치에서만 맞다. 가중치는 git에 없으므로, 각 값은 아래 주석의 PC에서 학습한 가중치 기준이다.
ROUTE_CALIBRATIONS: dict[str, dict[str, float]] = {
    # yunpilsoo PC · 2026-10-13 보정 (inquiries validation 120건, 범위 밖 평가는 documents validation 120건)
    # FINAL_MODEL. 같은 설정으로 다시 학습한 기준선 (설정 · validation 결과가 팀 실행과 같음 → 같은 모델)
    "baseline/20261013-1057": {"temperature": 2.009732, "tau": 0.504869, "tau_low": 0.45, "delta": 0.095286},
    # 예전 구조 (runs/classifier/inquiries/<모델>/model, 10:22~10:24 저장). 팀 실행 ID(1029 · 1055 …)와는 다른 가중치
    "baseline": {"temperature": 2.009732, "tau": 0.504869, "tau_low": 0.45, "delta": 0.095286},
    "bert-base": {"temperature": 6.232053, "tau": 0.505653, "tau_low": 0.45, "delta": 0.109825},
    "koelectra-small-v3-discriminator": {"temperature": 1.772721, "tau": 0.456737, "tau_low": 0.45, "delta": 0.15622},
    "roberta-base": {"temperature": 9.842287, "tau": 0.454914, "tau_low": 0.45, "delta": 0.167659},
}


@dataclass
class RoutingConfig:
    # ② 경로 규칙
    route_rule_enabled: bool = True
    # 보정값 고르기: auto면 쓰는 분류 모델의 키, 키를 적으면 그 항목 (ROUTE_CALIBRATIONS에 있어야 한다)
    route_calibration: str = "auto"
    route_task: str = DEFAULT_TASK
    route_temperature: float | None = None  # 아래 네 값은 모든 모델에 적용. 비우면 모델별 보정값 → 예시 값
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
        if self.route_calibration != "auto" and self.route_calibration not in ROUTE_CALIBRATIONS:
            raise ValueError(f"ROUTE_CALIBRATION={self.route_calibration!r}이 prompt/routing_config.py의 "
                             f"ROUTE_CALIBRATIONS에 없습니다 (있는 키: {list(ROUTE_CALIBRATIONS)})")

    @classmethod
    def from_env(cls, env_file: Path | None = ENV_FILE) -> RoutingConfig:
        return config_from_env(cls, env_file)


def check_calibrations(calibrations: dict) -> None:
    """ROUTE_CALIBRATIONS 형식 확인: 항목마다 temperature · tau · tau_low · delta 네 숫자, tau_low <= tau."""
    for key, values in calibrations.items():
        if not isinstance(values, dict) or set(values) != set(EXAMPLE_PARAMS) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in values.values()):
            raise ValueError(f"ROUTE_CALIBRATIONS[{key!r}]에는 {list(EXAMPLE_PARAMS)} 네 숫자가 있어야 합니다: "
                             f"{values!r}")
        if values["tau_low"] > values["tau"]:
            raise ValueError(f"ROUTE_CALIBRATIONS[{key!r}]: tau_low({values['tau_low']})가 "
                             f"tau({values['tau']})보다 큽니다")


check_calibrations(ROUTE_CALIBRATIONS)  # 붙여 넣다 틀리면 import할 때 바로 알 수 있게


def calibration_key(model_dir: Path, task: str = DEFAULT_TASK, runs_dir: Path | None = None) -> str:
    """분류 모델 폴더 → ROUTE_CALIBRATIONS 키. runs/classifier/<task>/ 기준 경로에서 끝의 model/을 뺀 것.

    runs/classifier/inquiries/baseline/20261013-2116/model → "baseline/20261013-2116"
    runs/classifier/inquiries/bert-base/model (예전 구조)  → "bert-base"
    runs 밖의 폴더는 저장소 기준 경로를 그대로 쓴다."""
    path = Path(model_dir).resolve()
    if path.name == MODEL_SUBDIR:
        path = path.parent
    try:
        return path.relative_to(task_paths(task, runs_dir=runs_dir).runs_dir.resolve()).as_posix()
    except ValueError:
        return display_path(path)


def calibration_path(model_dir: Path) -> Path:
    """보정 스크립트가 기록을 남기는 곳: 분류 모델 폴더 옆의 route_rule.json (서비스는 읽지 않는다)."""
    return Path(model_dir).parent / CALIBRATION_FILENAME


def resolve_params(config: RoutingConfig, model_dir: Path | None = None,
                   calibrations: dict[str, dict[str, float]] | None = None) -> dict:
    """경로 규칙에 넣을 값: .env 개별 값 > 모델별 보정값(ROUTE_CALIBRATIONS) > 예시 값.
    어디서 왔는지(source: env · calibration · example)도 함께 돌려준다.
    model_dir: 쓰는 분류 모델 폴더 (Classifier.model_dir). ROUTE_CALIBRATION=auto일 때 키를 만드는 데 쓴다
    calibrations: 테스트용 (없으면 ROUTE_CALIBRATIONS)
    calibration_key: 찾아본 키 (없으면 이 키로 ROUTE_CALIBRATIONS에 추가하면 된다) · calibration: 찾은 키"""
    calibrations = ROUTE_CALIBRATIONS if calibrations is None else calibrations
    if config.route_calibration != "auto":
        key = config.route_calibration
    else:
        key = calibration_key(model_dir, config.route_task) if model_dir else None
    model_values = calibrations.get(key, {}) if key else {}
    env_values = {
        "temperature": config.route_temperature,
        "tau": config.route_tau,
        "tau_low": config.route_tau_low,
        "delta": config.route_delta,
    }
    params, source = {}, {}
    for name, example in EXAMPLE_PARAMS.items():
        if env_values[name] is not None:
            params[name], source[name] = env_values[name], "env"
        elif name in model_values:
            params[name], source[name] = model_values[name], "calibration"
        else:
            params[name], source[name] = example, "example"
    return {"params": params, "source": source, "calibration_key": key,
            "calibration": key if model_values else None}


@lru_cache
def get_routing_config() -> RoutingConfig:
    return RoutingConfig.from_env()
