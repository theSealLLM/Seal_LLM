"""API 요청 형식과 응답 형식 (/api/answer · /api/answer/stream의 done).

응답은 내부 결과(theseal.common.schemas.AnswerResult, dataclass)에서 밖으로 보낼 필드만 골라 담는다.
    request_id · status · answer · error · usage · created_at
라벨 · 분류 근거 · 툴 기록 · 모델 같은 내부 정보는 응답에 넣지 않는다 (E2E · 비교 실험 기록에는 AnswerResult 전체가 남는다).

pydantic이 실행 중에 타입 힌트를 읽는 경계 모듈이라 `from __future__ import annotations`를 넣지 않는다.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from theseal.common.schemas import AnswerResult, ErrorCode, Label, Provider

MAX_INPUT_CHARS = 2000


class TextRequest(BaseModel):
    """고객 문의 원문 하나. 앞뒤 공백은 지우고, 공백만 있으면 400."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS, description="고객 문의 원문")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        """공백만 있는 문의를 막고 앞뒤 공백을 지운다."""
        if not v.strip():
            raise ValueError("빈 문의입니다 (공백만 있음)")
        return v.strip()


class AnswerRequest(TextRequest):
    """/api/answer · /api/answer/stream 요청."""

    label: Label | None = Field(default=None, description="넣으면 분류를 건너뛰고 이 라벨로 답한다 (테스트용)")
    provider: Provider | None = Field(default=None, description="local · openai. 비우면 .env의 DEFAULT_PROVIDER")


class AnswerError(BaseModel):
    """응답의 error: 오류 코드와 메시지."""

    model_config = ConfigDict(frozen=True, from_attributes=True)

    code: ErrorCode = Field(description="invalid_input · config_error · llm_error · llm_timeout")
    message: str


class AnswerUsage(BaseModel):
    """응답의 usage: 답변 하나에 쓴 토큰 수 (툴 반복 안의 모든 LLM 호출 합산). 모르면 null."""

    model_config = ConfigDict(frozen=True, from_attributes=True)

    input_tokens: int | None = None
    output_tokens: int | None = None


class AnswerResponse(BaseModel):
    """/api/answer 응답 (오류도 같은 형식). 내부 결과 AnswerResult에서 밖으로 보낼 필드만 골라 담는다."""

    model_config = ConfigDict(frozen=True, from_attributes=True)

    request_id: str
    status: Literal["ok", "error"]
    answer: str | None = Field(description="평문 답변. 오류면 null")
    error: AnswerError | None = None
    usage: AnswerUsage = AnswerUsage()
    created_at: str = Field(description="ISO 8601, KST")

    @classmethod
    def from_result(cls, result: AnswerResult) -> Self:
        """내부 결과에서 응답 필드만 골라 만든다 (이름이 같은 속성을 읽는다)."""
        return cls.model_validate(result)
