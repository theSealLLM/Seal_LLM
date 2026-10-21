"""답변 생성(answer/) 설정: 툴 키 · 라벨별 툴 · 툴 반복 횟수 · 시간 제한.

.env에서 필드 이름을 대문자로 적어 바꾼다 (예: max_tool_rounds → MAX_TOOL_ROUNDS=3).
LLM 연결과 여러 모듈이 같이 쓰는 값은 common/config.py의 LLMConfig에 있다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from theseal.common.config import ENV_FILE, config_from_env

KMA_API_BASE = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0"


@dataclass(frozen=True)
class AnswerConfig:
    """답변 생성 설정. AnswerGenerator · ToolCaller · 툴 레지스트리가 쓴다."""

    # 툴
    tavily_api_key: str = field(default="", repr=False)  # .env (비밀 키). 비우면 검색 툴을 등록하지 않는다
    # .env (비밀 키): 기상청 API허브 인증키. 비우면 날씨 툴을 등록하지 않는다
    kma_auth_key: str = field(default="", repr=False)
    kma_api_base: str = KMA_API_BASE

    # 라벨별로 붙일 툴 그룹 (answer/tools의 TOOL_GROUPS). 키가 없는 툴은 여기 적혀 있어도 빠진다.
    # .env에서는 JSON으로 적는다 (예: TOOLS_BY_LABEL={"unknown": ["date_tools"]})
    tools_by_label: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {"unknown": ("date_tools", "weather", "web_search")}
    )
    max_tool_rounds: int = 2  # 툴 실행 라운드 수. 툴 결과로 답을 쓸 LLM 호출 1회는 항상 더 허용

    # 제한 (LLM 호출 한 번은 LLMConfig.llm_timeout_sec)
    tool_timeout_sec: float = 10  # 툴 실행 한 번
    # 답변 생성 한 번 전체 (툴 경로는 LLM 최대 max_tool_rounds+1회 + 툴). 넘으면 llm_timeout. 실측 툴 경로 5~10초
    answer_timeout_sec: float = 90

    def __post_init__(self) -> None:
        # frozen이어도 dict · list 안쪽은 바뀔 수 있으므로 읽기 전용 매핑 · 튜플로 바꿔 둔다
        frozen = MappingProxyType({label: tuple(groups) for label, groups in self.tools_by_label.items()})
        object.__setattr__(self, "tools_by_label", frozen)

    @classmethod
    def from_env(cls, env_file: Path | None = ENV_FILE) -> AnswerConfig:
        """환경변수 > .env > 코드 기본값 순서로 채운다."""
        return config_from_env(cls, env_file)
