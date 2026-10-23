"""프롬프트 생성 → 답변 생성 사이의 데이터 형식. 모두 내부 값이라 frozen dataclass다.

이 파일은 두 담당자(prompt/ · answer/)가 함께 쓰는 약속이라, 바꿀 때는 양쪽이 같이 확인한다.

    PromptSpec     prompt가 만들고 answer가 받아 LLM 메시지로 가공한다
    AnswerResult   answer의 결과 전체 (툴 기록 · 토큰 · 지연 · 모델 등). E2E · 비교 실험은 asdict()로 그대로 기록하고,
                   API는 그중 일부만 골라 응답(api/schemas.py의 AnswerResponse, pydantic)으로 내보낸다
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

Label = Literal["shipping", "refund", "account", "unknown"]
Provider = Literal["local", "openai"]
ErrorCode = Literal["invalid_input", "config_error", "llm_timeout", "llm_error"]


@dataclass(frozen=True)
class PromptSpec:
    """프롬프트 생성 모듈 → 답변 생성 모듈.

    같은 PromptSpec을 로컬 · 상용 LLM에 그대로 넣는다 (같은 입력과 지시문 조건).
    """

    label: Label
    """분류 단계의 최종 라벨. 툴 사용 여부를 정한다 (기본: unknown만)"""

    system: str
    """지시문 (공통 규칙 + 라벨별 규칙 + 예시). 비어 있으면 ValueError"""

    customer_inquiry: str
    """고객 문의 원문. 비어 있으면 ValueError"""

    prompt_version: str
    """템플릿 버전. 비교 기록이 섞이지 않게 템플릿이 바뀌면 같이 바뀐다"""

    def __post_init__(self) -> None:
        if self.label not in get_args(Label):
            raise ValueError(f"label은 {get_args(Label)} 중 하나여야 합니다: {self.label!r}")
        if not self.system:
            raise ValueError("system(지시문)이 비어 있습니다")
        if not self.customer_inquiry:
            raise ValueError("customer_inquiry(고객 문의)가 비어 있습니다")


@dataclass(frozen=True)
class ToolRecord:
    """실행한 툴 하나의 기록. 툴이 실패해도 답변은 계속 만들고 ok=False로 남는다."""

    name: str
    """LLM이 부른 툴 이름 (get_today, web_search …). 허용되지 않은 이름도 그대로 남는다"""

    args: Mapping[str, Any]
    """LLM이 넘긴 인자"""

    ok: bool
    latency_ms: int

    error: str | None = None
    """실패 이유. 성공이면 None"""


@dataclass(frozen=True)
class Usage:
    """토큰 수. 툴 반복 안의 모든 LLM 호출 합산이고, 모르면 None."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ErrorInfo:
    """오류 코드와 메시지."""

    code: ErrorCode
    message: str


@dataclass(frozen=True, kw_only=True)
class AnswerResult:
    """답변 생성 모듈의 출력 전체. 필드 순서는 E2E · 비교 실험 기록(asdict)의 키 순서다."""

    request_id: str
    status: Literal["ok", "error"]
    label: Label
    answer: str | None
    """평문 답변. 오류면 None"""

    provider: Provider
    model: str
    prompt_version: str
    tools_used: tuple[ToolRecord, ...] = ()
    latency_ms: int
    usage: Usage = field(default_factory=Usage)

    llm_options: Mapping[str, Any] = field(default_factory=dict)
    """생성 조건 기록 (thinking, reasoning_effort 등). 비교 조건 확인용"""

    error: ErrorInfo | None = None

    created_at: str
    """ISO 8601, KST"""
