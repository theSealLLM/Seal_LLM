"""프롬프트 생성(prompt/) 설정: 분류 후처리의 경로 규칙 · LLM 검증.

.env에서 필드 이름을 대문자로 적어 바꾼다 (예: VERIFY_MODE=off).
경로 규칙 기준선(RULE)은 설정이 아니라 router.py의 고정값이다.
LLM 연결과 여러 모듈이 같이 쓰는 값은 common/config.py의 LLMConfig에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from theseal.common.config import ENV_FILE, config_from_env

VERIFY_MODES = ("off", "uncertain", "all")


@dataclass(frozen=True)
class RouterConfig:
    """켜고 끄기 · LLM 검증 설정. Router가 쓴다."""

    route_rule_enabled: bool = True  # ② 경로 규칙. 끄면 분류기 라벨을 그대로 쓴다
    verify_mode: str = "uncertain"  # ③ off | uncertain(확신이 아닌 것만) | all
    verify_provider: str = "local"
    verify_timeout_sec: float = 10  # 넘으면 검증 없이 분류 라벨을 그대로 쓴다

    def __post_init__(self) -> None:
        if self.verify_mode not in VERIFY_MODES:
            raise ValueError(f"VERIFY_MODE는 {VERIFY_MODES} 중 하나여야 합니다: {self.verify_mode!r}")

    @classmethod
    def from_env(cls, env_file: Path | None = ENV_FILE) -> RouterConfig:
        """환경변수 > .env > 코드 기본값 순서로 채운다."""
        return config_from_env(cls, env_file)
