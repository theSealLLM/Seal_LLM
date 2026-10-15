"""프롬프트 생성 → 답변 생성 사이의 데이터 형식.

이 파일은 두 담당자(prompt/ · answer/)가 함께 쓰는 약속이라, 바꿀 때는 양쪽이 같이 확인한다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["shipping", "refund", "account", "unknown"]
Provider = Literal["local", "openai"]
ErrorCode = Literal["invalid_input", "config_error", "llm_timeout", "llm_error"]


class PromptSpec(BaseModel):
    """프롬프트 생성 모듈 → 답변 생성 모듈.

    같은 PromptSpec을 로컬 · 상용 LLM에 그대로 넣는다 (같은 입력과 지시문 조건).
    """

    label: Label  # 분류 단계의 최종 라벨. 툴 사용 여부를 정한다 (기본: unknown만)
    system: str = Field(min_length=1)  # 지시문 (공통 규칙 + 라벨별 규칙 + 예시)
    customer_inquiry: str = Field(min_length=1)  # 고객 문의 원문
    prompt_version: str  # 템플릿 버전. 비교 기록이 섞이지 않게 템플릿이 바뀌면 같이 바뀐다


class ToolRecord(BaseModel):
    name: str  # 실제 호출된 툴 이름 (get_today, web_search …)
    args: dict
    ok: bool  # 툴이 실패해도 답변은 계속 만든다
    latency_ms: int
    error: str | None = None


class Usage(BaseModel):
    input_tokens: int | None = None  # 툴 반복 안의 모든 LLM 호출 합산. 모르면 null
    output_tokens: int | None = None


class ErrorInfo(BaseModel):
    code: ErrorCode
    message: str


class AnswerResult(BaseModel):
    """답변 생성 모듈의 출력. 프론트 응답과 비교 실험 기록에 같은 형식을 쓴다."""

    request_id: str
    status: Literal["ok", "error"]
    label: Label
    answer: str | None  # 평문 답변. 오류면 null
    provider: Provider
    model: str
    prompt_version: str
    tools_used: list[ToolRecord] = []
    latency_ms: int
    usage: Usage = Usage()
    llm_options: dict = {}  # 생성 조건 기록 (thinking, reasoning_effort 등). 비교 조건 확인용
    error: ErrorInfo | None = None
    created_at: str  # ISO 8601, KST
