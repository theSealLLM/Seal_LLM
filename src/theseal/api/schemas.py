"""API 요청 형식과 /api/answer 응답 형식.

응답 본문은 기능 모듈의 형식을 그대로 쓴다.
  /api/classify → theseal.routing.router.RoutingResult
  /api/answer   → theseal.llm.schemas.AnswerResult + classification(RoutingResult)
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from theseal.common.schemas import AnswerResult, Label, Provider
from theseal.prompt.router import RoutingResult

MAX_INPUT_CHARS = 2000


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS, description="고객 문의 원문")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("빈 문의입니다 (공백만 있음)")
        return v.strip()


class ClassifyRequest(TextRequest):
    pass


class AnswerRequest(TextRequest):
    label: Label | None = Field(default=None, description="넣으면 분류를 건너뛰고 이 라벨로 답한다 (테스트용)")
    provider: Provider | None = Field(default=None, description="local · openai. 비우면 .env의 DEFAULT_PROVIDER")


class AnswerResponse(AnswerResult):
    classification: RoutingResult | None = None  # 분류를 거쳤을 때만. label을 직접 넣으면 null


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """/api/classify의 오류 응답 (/api/answer는 오류도 AnswerResponse 형식)"""

    error: ErrorBody
