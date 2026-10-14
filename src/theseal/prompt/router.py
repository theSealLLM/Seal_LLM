"""② 경로 규칙 → ③ LLM 검증을 설정대로 이어서 실행한다.

입력은 분류기 Classifier.predict(text)의 결과(ClassificationResult) 하나다: run(result). 원문은 result.text
라벨별 확률(scores)에는 task의 labels.json 순서로 이름을 붙이고, 보정값은 FINAL_MODEL 것을 쓴다.
안쪽(경로 규칙 · 검증 · 출력)은 라벨 이름을 쓴다.
출력 RoutingResult의 label · text가 프롬프트 생성 모듈로 넘어간다. route · verification은 기록용.
②·③은 각각 .env로 끈다 (ROUTE_RULE_ENABLED, VERIFY_MODE=off). 둘 다 끄면 분류기 라벨을 그대로 쓴다.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from theseal.common.client import make_llm
from theseal.common.config import LLMConfig, get_config

from . import llm_verifier, route_rule
from .routing_config import RoutingConfig, get_routing_config, resolve_params, task_labels

if TYPE_CHECKING:
    from theseal.classifier.classifier import ClassificationResult


class RoutingResult(BaseModel):
    label: str  # 최종 라벨 → 프롬프트 생성
    text: str  # 원래 질문 → 프롬프트 생성 (그대로 전달)
    classifier_label: str  # 분류기가 낸 1등 라벨
    scores: dict[str, float]  # 분류기 확률 (보정 전)
    route: dict | None = None  # ② 결과 (RouteDecision + 값 출처). 경로 규칙을 끄면 null
    verification: dict | None = None  # ③ 결과 (VerifyResult). 검증하지 않았으면 null


def to_scores(prediction: ClassificationResult, labels: Sequence[str]) -> tuple[str, dict[str, float]]:
    """Classifier.predict 결과 → (라벨 이름, {라벨 이름: 확률})
    labels와 결과가 어긋나면(라벨 수 · 번호 · 이름) 다른 모델의 라벨을 붙이는 것이므로 막는다."""
    index, probs = prediction.label_num, prediction.scores
    if len(probs) != len(labels) or not 0 <= index < len(labels):
        raise ValueError(f"분류 결과와 라벨 수가 맞지 않습니다: 번호 {index}, 확률 {len(probs)}개, 라벨 {list(labels)}")
    if prediction.label_name != labels[index]:
        raise ValueError(f"분류 결과의 라벨 이름({prediction.label_name!r})이 labels[{index}]"
                         f"({labels[index]!r})와 다릅니다")
    return labels[index], {name: float(p) for name, p in zip(labels, probs, strict=True)}


def apply_route_rule(
    label: str, scores: dict[str, float], config: RoutingConfig, model_dir: Path | None = None
) -> tuple[str, dict | None]:
    """② 경로 규칙. 꺼져 있으면 분류기 라벨을 그대로 쓴다."""
    if not config.route_rule_enabled:
        return label, None
    resolved = resolve_params(config, model_dir)
    decision = route_rule.decide(scores, **resolved["params"], uncertain_action=config.route_uncertain_action)
    return decision.label, {**decision.model_dump(), "source": resolved["source"],
                            "calibration": resolved["calibration"],
                            "calibration_key": resolved["calibration_key"]}


async def run(
    prediction: ClassificationResult,
    *,
    config: RoutingConfig | None = None,
    llm_config: LLMConfig | None = None,
    llm: BaseChatModel | None = None,
) -> RoutingResult:
    """prediction: Classifier.predict(text) 결과 (ClassificationResult: text · label_num · label_name · scores)
    라벨 이름은 ROUTE_TASK의 labels.json 순서, 보정값은 FINAL_MODEL 것을 쓴다.
    config · llm_config · llm: 테스트에서 설정이나 가짜 LLM을 끼울 때만 넘긴다 (없으면 .env와 코드 기본값)

    예) clf = Classifier.from_final()
        result = await router.run(clf.predict(text))

    FINAL_MODEL이 아닌 모델로 실험할 때는 to_scores(labels) · apply_route_rule(model_dir)를 직접 쓴다."""
    config = config or get_routing_config()
    text = prediction.text
    classifier_label, scores = to_scores(prediction, task_labels(config.route_task))
    label, route = apply_route_rule(classifier_label, scores, config)
    result = RoutingResult(label=label, text=text, classifier_label=classifier_label,
                           scores=scores, route=route)

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
