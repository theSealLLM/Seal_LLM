"""채점: 답변 하나를 채점표(rubric.py)로 채점한다. 우리 서비스 답변과 OpenAI 답변에 똑같이 쓴다.

채점 LLM에는 문의 · 문의 유형 · 정책 원문 · 도구 결과 · 답변만 넘긴다. 누가 쓴 답변인지와 다른 쪽 답변은 넘기지 않는다
(두 답변을 같은 조건으로 따로 채점해야 점수 차이를 비교할 수 있다).

채점 결과(score) 형식
    {"version", "answerable", "points": {"fact": 9, "info": 7, "policy": 9}, "total": 25,
     "verdicts": [{id, applicable, satisfied, point, evidence, reason}, ...]}
"""

from __future__ import annotations

import hashlib
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .rate_limit import RateLimiter
from .rubric import CATEGORIES, INDICATORS, LABEL_GUIDE, POLICY_TEXT
from .schemas import Scorecard

INDICATOR_TEXT = "\n".join(f"- {i.id} [{CATEGORIES[i.category]}] {i.name}: {i.satisfied}" for i in INDICATORS)

JUDGE_SYSTEM = f"""당신은 온라인 쇼핑몰 고객센터 답변의 채점자다. 아래 서비스 정책과 채점 지표로 answer를 채점한다.

{POLICY_TEXT}

[채점 지표] 지표마다 기준을 만족하면 satisfied=true(1점), 아니면 false(0점)
{INDICATOR_TEXT}

[채점 방법]
- 모든 지표를 하나도 빠뜨리지 않고 채점한다. 목록에 없는 id를 만들지 않는다.
- 이 문의 · 답변에 해당하지 않는 지표는 applicable=false, satisfied=true로 둔다
  (예: 숫자를 말하지 않았으면 F03, 필요한 서류가 없는 문의면 I04, 배송 문의면 P01 · P03).
- 사실 여부(F*)와 정책 위반(P*)은 답변에 틀린 내용이나 위반이 있을 때만 0점이다.
- 중요 정보 포함(I*)은 이 문의를 해결하는 데 꼭 필요한 정보가 빠졌을 때만 0점이다. 길게 쓰지 않았다고 감점하지 않는다.
- 표현이 달라도 뜻이 같으면 같은 내용으로 본다.
- 실제 주문 상태 · 배송 위치 · 처리 결과는 이 창구가 알 수 없다. 이런 내용을 단정하면 근거 없는 주장이다.
- evidence는 answer에서 글자 그대로 옮긴다. 근거 문장이 없으면 null."""

JUDGE_VERSION = "judge-v2-" + hashlib.sha256(JUDGE_SYSTEM.encode("utf-8")).hexdigest()[:8]
INDICATOR_IDS = sorted(i.id for i in INDICATORS)
CATEGORY_OF = {i.id: i.category for i in INDICATORS}


def judge_messages(case: dict, answer: str, tool_outputs: list[dict]) -> list:
    label = case["expected_label"]
    payload = {
        "inquiry_type": LABEL_GUIDE[label] if label else None,
        "customer_question": case["text"],
        "tool_outputs": tool_outputs,
        "answer": answer,
    }
    # 데이터에 사람이 적은 기준이 있으면(generation_*.jsonl) 참고로 준다
    if case.get("required_points"):
        payload["reference_required_points"] = case["required_points"]
    if case.get("forbidden_claims"):
        payload["reference_forbidden_claims"] = case["forbidden_claims"]
    return [SystemMessage(JUDGE_SYSTEM), HumanMessage(json.dumps(payload, ensure_ascii=False, indent=2))]


def tally(verdicts: list[dict], answerable: bool) -> dict:
    points = {cat: sum(v["point"] for v in verdicts if CATEGORY_OF[v["id"]] == cat) for cat in CATEGORIES}
    return {"version": JUDGE_VERSION, "answerable": answerable, "points": points, "total": sum(points.values()),
            "verdicts": verdicts}


def zero_score(reason: str) -> dict:
    """답변이 없을 때 (서비스 오류): 모든 지표 0점."""
    verdicts = [{"id": i.id, "applicable": True, "satisfied": False, "point": 0, "evidence": None, "reason": reason}
                for i in INDICATORS]
    return tally(verdicts, answerable=True)


async def score_answer(case: dict, answer: str | None, tool_outputs: list[dict], *, llm: BaseChatModel,
                       limiter: RateLimiter, retries: int = 1) -> dict:
    if not answer:
        return zero_score("답변이 없음 (서비스 오류)")
    messages = judge_messages(case, answer, tool_outputs)
    for _ in range(retries + 1):
        card: Scorecard = await limiter.call(lambda: llm.with_structured_output(Scorecard).ainvoke(messages),
                                             label=case["id"])
        got = sorted(v.id for v in card.verdicts)
        if got == INDICATOR_IDS:  # 빠뜨리거나 지어낸 지표가 있으면 다시 묻는다
            break
    else:
        raise ValueError(f"채점 지표가 맞지 않습니다: 기대 {INDICATOR_IDS}, 받음 {got}")
    order = {i.id: n for n, i in enumerate(INDICATORS)}
    verdicts = [{**v.model_dump(), "point": int(v.satisfied or not v.applicable)}
                for v in sorted(card.verdicts, key=lambda v: order[v.id])]
    return tally(verdicts, card.answerable)


def module_hint(trace: dict, service_ok: bool) -> str | None:
    """우리 점수가 낮을 때 먼저 볼 모듈. 최종 라벨이 맞았으면 프롬프트 · 답변, 틀렸으면 틀린 단계.
    분류기는 shipping · refund · account만 내고 unknown은 분류 후처리(경로 규칙 · LLM 검증)가 정한다."""
    if not service_ok:
        return "answer"
    gold = trace["expected_label"]
    if gold is None:
        return None
    if trace["final_label"] == gold:
        return "prompt/answer"
    if gold != "unknown" and trace["classifier_label"] != gold:
        return "classifier"
    return "routing"
