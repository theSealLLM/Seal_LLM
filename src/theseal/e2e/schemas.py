"""채점 LLM의 구조화 출력 형식. LLM은 지표마다 만족 여부와 근거만 내고, 점수 합계는 코드가 계산한다."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndicatorVerdict(BaseModel):
    id: str  # F01 … I10 … P09 (rubric.INDICATORS)
    applicable: bool = Field(description="이 문의 · 답변에 해당하는 지표면 true. 해당하지 않으면 false (만족으로 센다)")
    satisfied: bool = Field(description="기준을 만족하면 true. applicable이 false면 true")
    evidence: str | None = Field(description="판단 근거가 된 답변 문장을 글자 그대로. 없으면 null")
    reason: str = Field(description="판단 이유 한 문장")


class Scorecard(BaseModel):
    answerable: bool = Field(description="문의만 보고 무엇을 묻는지 알 수 있으면 true. 앞 대화가 있어야 뜻이 통하면 false")
    verdicts: list[IndicatorVerdict]
