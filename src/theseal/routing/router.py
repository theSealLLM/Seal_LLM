"""② 경로 규칙 → ③ LLM 검증을 설정대로 이어서 실행한다.

입력은 분류기 Predictor.predict(text)의 결과를 그대로 받는다: {"label", "confidence", "scores"}
출력 RoutingResult의 label · text가 프롬프트 생성 모듈로 넘어간다. route · verification은 기록용.
②·③은 각각 .env로 끈다 (ROUTE_RULE_ENABLED, VERIFY_MODE=off). 둘 다 끄면 분류기 라벨을 그대로 쓴다.
"""

from __future__ import annotations

import asyncio

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from theseal.llm.client import make_llm
from theseal.llm.config import LLMConfig, get_config

from . import llm_verifier, route_rule
from .config import RoutingConfig, get_routing_config, resolve_params


class RoutingResult(BaseModel):
    label: str  # 최종 라벨 → 프롬프트 생성
    text: str  # 원래 질문 → 프롬프트 생성 (그대로 전달)
    classifier_label: str  # 분류기가 낸 1등 라벨
    scores: dict[str, float]  # 분류기 확률 (보정 전)
    route: dict | None = None  # ② 결과 (RouteDecision + 값 출처). 경로 규칙을 끄면 null
    verification: dict | None = None  # ③ 결과 (VerifyResult). 검증하지 않았으면 null


def apply_route_rule(
    prediction: dict, config: RoutingConfig, model_name: str | None = None
) -> tuple[str, dict | None]:
    """② 경로 규칙. 꺼져 있으면 분류기 라벨을 그대로 쓴다."""
    if not config.route_rule_enabled:
        return prediction["label"], None
    resolved = resolve_params(config, model_name)
    decision = route_rule.decide(
        prediction["scores"], **resolved["params"], uncertain_action=config.route_uncertain_action)
    return decision.label, {**decision.model_dump(), "source": resolved["source"],
                            "calibration": resolved["calibration"]}


async def run(
    prediction: dict,
    text: str,
    *,
    model_name: str | None = None,
    config: RoutingConfig | None = None,
    llm_config: LLMConfig | None = None,
    llm: BaseChatModel | None = None,
) -> RoutingResult:
    """prediction: Predictor.predict(text) 결과
    model_name: 분류 모델 이름 (Predictor.meta["name"]). ROUTE_CALIBRATION=auto일 때 보정 파일을 찾는 데 쓴다
    llm: 검증에 쓸 LLM (없으면 VERIFY_PROVIDER로 만든다)"""
    config = config or get_routing_config()
    label, route = apply_route_rule(prediction, config, model_name)
    result = RoutingResult(label=label, text=text, classifier_label=prediction["label"],
                           scores=prediction["scores"], route=route)

    if not llm_verifier.needs_verification(label, route and route["route"], config.verify_mode):
        return result
    try:
        llm = llm or make_llm(config.verify_provider, llm_config or get_config())
        v = await asyncio.wait_for(llm_verifier.verify(llm, text, label), config.verify_timeout_sec)
    except Exception as e:  # noqa: BLE001 — 설정 오류(키 없음) · 시간 초과도 분류 라벨 유지
        err = "시간 초과" if isinstance(e, TimeoutError) else f"{type(e).__name__}: {e}"
        v = llm_verifier.VerifyResult(match=True, reason="", label=label, final_label=label,
                                      latency_ms=0, error=err)
    result.verification = {**v.model_dump(), "provider": config.verify_provider,
                           "mode": config.verify_mode}
    result.label = v.final_label
    return result
