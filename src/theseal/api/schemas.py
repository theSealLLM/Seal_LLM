"""API 요청 형식과 /api/answer 응답 형식.

응답 본문은 기능 모듈의 형식을 그대로 쓴다.
  /api/answer   → theseal.common.schemas.AnswerResult + classification(theseal.prompt.router.RoutingResult)

pydantic이 실행 중에 타입 힌트를 읽는 경계 모듈이라 `from __future__ import annotations`를 넣지 않는다.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from theseal.common.schemas import AnswerResult, Label, Provider
from theseal.prompt.router import RoutingResult

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


class AnswerResponse(AnswerResult):
    """/api/answer 응답 (오류도 같은 형식). AnswerResult에 분류 근거를 더한다."""

    classification: RoutingResult | None = None  # 분류를 거쳤을 때만. label을 직접 넣으면 null
