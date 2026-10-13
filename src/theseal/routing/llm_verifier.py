"""③ LLM 검증. 분류 라벨이 맞는지 LLM에 예/아니오로 한 번 더 묻는다.

분류기는 단어 패턴으로 판단해, 고객 문의가 아닌 글(공지 · 매뉴얼 등)도 비슷한 단어가 있으면 라벨을 붙인다.
LLM은 의미를 보고 "고객이 보낸 문의이고 주제가 그 라벨인가"를 확인한다. 아니면 unknown으로 돌린다.
LLM 객체는 호출한 쪽에서 받는다 (theseal.llm.client.make_llm).
"""

from __future__ import annotations

import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

LABEL_DESCRIPTIONS = {
    "shipping": "배송 · 수령: 배송 진행, 수령 장소와 방법, 운송 및 인도 상태",
    "refund": "환불 · 반품 · 주문 취소: 결제 취소, 반품, 환급 금액과 수단, 처리",
    "account": "계정: 가입, 로그인, 인증, 회원정보, 계정 권한, 계정 자료 관리, 회원 탈퇴",
}

SYSTEM = """당신은 온라인 쇼핑몰 고객센터의 문의 분류를 검수합니다.
분류기가 붙인 라벨이 맞는지 판단합니다.

다음 두 조건을 모두 만족하면 match를 true로 답합니다.
1. 입력이 고객이 고객센터에 보낸 문의입니다. 공지문, 매뉴얼, 회의록, 안내문 같은 문서나 잡담, 일반 상식 질문, 날짜나 날씨 자체만 묻는 질문은 문의가 아닙니다.
2. 문의의 주된 요청이 아래 라벨 설명에 해당합니다. 특정 단어가 아니라 고객이 원하는 것을 기준으로 봅니다.

날씨, 날짜, 명절, 연휴 같은 상황이 함께 나와도 고객이 그 상황 때문에 배송 · 환불 · 계정에 관해 묻는다면 상황은 배경일 뿐입니다. 주된 요청으로 판단합니다.
예) "오늘 날씨가 안 좋아서 배송이 안 되려나?" → 배송을 묻는 문의 (shipping이면 true)
예) "오늘 날씨 어때요?" → 날씨 자체만 묻는 질문 (false)

하나라도 아니면 match를 false로 답합니다. reason에는 판단 근거를 한 문장으로 씁니다."""


class VerifyOutput(BaseModel):
    match: bool = Field(description="두 조건을 모두 만족하면 true")
    reason: str = Field(description="판단 근거 한 문장")


class VerifyResult(BaseModel):
    match: bool
    reason: str
    label: str  # 검증한 라벨
    final_label: str  # match면 label, 아니면 unknown
    latency_ms: int
    error: str | None = None  # LLM 호출 실패 시 (이때는 분류 라벨을 그대로 둔다)


def needs_verification(label: str, route: str | None, mode: str) -> bool:
    """all: unknown이 아닌 라벨 전부 / uncertain: 경로 규칙이 confident라고 하지 않은 것
    (경로 규칙을 쓰지 않아 route가 없으면 confident 판단이 없으므로 uncertain도 전부 검증한다)"""
    if mode == "off" or label == "unknown":
        return False
    if mode == "all":
        return True
    return route != "confident"


def build_messages(text: str, label: str) -> list:
    human = f"[라벨]\n{label} — {LABEL_DESCRIPTIONS[label]}\n\n[입력]\n{text}"
    return [SystemMessage(SYSTEM), HumanMessage(human)]


async def verify(llm: BaseChatModel, text: str, label: str) -> VerifyResult:
    """label이 검증 대상이 아니면(unknown 등) 그대로 둔다. LLM 오류는 삼키고 분류 라벨을 유지한다 (fail-open)."""
    start = time.perf_counter()
    if label not in LABEL_DESCRIPTIONS:
        return VerifyResult(match=True, reason="검증 대상 아님", label=label, final_label=label, latency_ms=0)
    try:
        out: VerifyOutput = await llm.with_structured_output(VerifyOutput).ainvoke(
            build_messages(text, label))
        return VerifyResult(
            match=out.match,
            reason=out.reason,
            label=label,
            final_label=label if out.match else "unknown",
            latency_ms=round((time.perf_counter() - start) * 1000),
        )
    except Exception as e:  # noqa: BLE001 — 검증 실패로 서비스가 멈추지 않게 한다
        return VerifyResult(
            match=True, reason="", label=label, final_label=label,
            latency_ms=round((time.perf_counter() - start) * 1000), error=f"{type(e).__name__}: {e}",
        )
