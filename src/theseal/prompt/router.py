"""분류 후처리: 분류기 결과 → ② 경로 규칙 → ③ LLM 검증 → 최종 라벨. 처리는 모두 Router가 한다."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, get_args

from langchain_core.language_models import BaseChatModel

from theseal.classifier.config import DEFAULT_TASK, task_paths
from theseal.classifier.data_loader import load_labels
from theseal.common.client import ConfigError, make_llm
from theseal.common.config import LLMConfig
from theseal.common.schemas import Label
from theseal.prompt import llm_verifier
from theseal.prompt.config import RouterConfig
from theseal.prompt.route_rule import CONFIDENT, REASONS, UNKNOWN, RouteRule

if TYPE_CHECKING:
    from theseal.classifier.classifier import ClassificationResult

logger = logging.getLogger(__name__)

# 경로 규칙 기준선. 지금은 분류 모델과 상관없이 이 값 하나로 고정해 쓴다.
# 모델별로 다른 값(확률 보정 포함)이 필요해지면 그때 Router가 모델에 맞는 RouteRule을 고르도록 확장한다.
RULE = RouteRule(tau=0.60, tau_low=0.45, delta=0.15)

NO_VERIFY_LLM = "검증 LLM이 없습니다 (Router.from_env()로 만들거나 llm을 넘긴다)"


def check_labels(labels: Sequence[str]) -> None:
    """분류 모델 라벨(labels.json)이 Label 정의 · 검증 라벨 설명(LABEL_DESCRIPTIONS)과 맞는지 확인한다.

    서버 시작 때(Router.from_env) 한 번 부른다. 첫 요청에서야 KeyError가 나지 않게, 어긋나면 바로 ValueError.
    """
    defined = set(get_args(Label)) - {UNKNOWN}
    if extra := set(labels) - defined:
        raise ValueError(f"분류 모델 라벨이 Label 정의에 없습니다: {sorted(extra)} (common/schemas.py의 Label)")
    descriptions = set(llm_verifier.LABEL_DESCRIPTIONS)
    if missing := set(labels) - descriptions:
        raise ValueError(f"검증 라벨 설명이 없습니다: {sorted(missing)} (llm_verifier.LABEL_DESCRIPTIONS)")
    if extra := descriptions - defined:
        raise ValueError(f"검증 라벨 설명에 Label 정의에 없는 라벨이 있습니다: {sorted(extra)}")


@dataclass(frozen=True)
class RouteDecision:
    """② 경로 규칙 결과. RoutingResult.route에 담긴다 (기록용)."""

    route: str
    """경로. confident · ambiguous · unknown"""

    label: str
    """경로 규칙 뒤 라벨. unknown이면 unknown, 나머지는 1등 라벨"""

    top1: str
    top2: str

    p1: float
    """1등 확률"""

    p2: float
    """2등 확률"""

    margin: float
    """p1 - p2"""

    reason: str
    """경로를 고른 조건 (route_rule.REASONS)"""

    tau: float
    tau_low: float
    delta: float


@dataclass(frozen=True)
class RoutingResult:
    """분류 후처리 결과. Router.run의 반환값.

        result = await router.run(clf.predict("환불은 언제 되나요?"))

        result.label             # 'refund'  최종 라벨 → 프롬프트 생성
        result.text              # '환불은 언제 되나요?'
        result.classifier_label  # 'refund'
        result.route.route       # 'confident'
        result.verification      # None (확신이라 검증하지 않음)

    label · text가 프롬프트 생성 모듈로 넘어가고, 나머지는 기록용이다 (API 응답의 classification).
    """

    label: str
    """최종 라벨 (shipping · refund · account · unknown)"""

    text: str
    """고객 문의 원문. ClassificationResult.text 그대로다 (API 응답 키라 이름 유지)"""

    classifier_label: str
    """분류기가 낸 1등 라벨"""

    scores: dict[str, float]
    """분류기 확률. {라벨 이름: 확률}, labels.json 순서"""

    route: RouteDecision | None = None
    """② 결과. 경로 규칙을 끄면 None"""

    verification: dict[str, Any] | None = None
    """③ 결과 (VerifyResult 필드 + provider · mode). 검증하지 않았으면 None"""


class Router:
    """분류기 결과 하나를 ② 경로 규칙 → ③ LLM 검증 순서로 거쳐 최종 라벨을 정한다.

        router = Router.from_env()
        result = await router.run(clf.predict("환불은 언제 되나요?"))
        result.label  →  'refund'

    Args:
      labels: 분류기 확률 순서 (labels.json).
      config: 경로 규칙 · 검증 설정. 없으면 코드 기본값.
      rule: 경로 규칙 기준선 (기본: 고정값 RULE).
      llm: 검증 LLM. 없으면 검증할 라벨을 그대로 둔다 (fail-open). 서비스는 from_env()가 만들어 넘기고,
        테스트는 가짜 LLM을 넘긴다.
      llm_error: llm이 없는 이유 (예: 키 없음). 검증 결과의 error에 남긴다.
    """

    def __init__(
        self,
        labels: Sequence[str],
        config: RouterConfig | None = None,
        *,
        rule: RouteRule = RULE,
        llm: BaseChatModel | None = None,
        llm_error: str | None = None,
    ) -> None:
        self.labels = tuple(labels)
        self.config = config or RouterConfig()
        self.rule = rule
        self.llm = llm
        self.llm_error = llm_error

    @classmethod
    def from_env(cls, task: str = DEFAULT_TASK, *, llm_config: LLMConfig | None = None) -> Router:
        """서비스 기본 구성: 라벨은 data/<task>/labels.json, 설정은 .env(RouterConfig), 기준선은 고정값 RULE.

        라벨이 Label 정의 · 검증 라벨 설명과 맞는지 여기서 확인하고(check_labels), 검증 LLM도 여기서 만든다.
        검증을 끄면(VERIFY_MODE=off) LLM을 만들지 않는다. 키가 없어 만들 수 없으면 검증만 건너뛴다 (fail-open).
        llm_config: 검증 LLM 설정 (Prompt가 들고 있는 것을 넘긴다). 없으면 .env + 코드 기본값
        """
        labels = load_labels(task_paths(task).data_dir).labels
        check_labels(labels)
        config = RouterConfig.from_env()
        llm, llm_error = None, None
        if config.verify_mode != "off":
            try:
                llm = make_llm(config.verify_provider, llm_config or LLMConfig.from_env())
            except ConfigError as e:
                llm_error = f"{type(e).__name__}: {e}"
                logger.warning("검증 LLM을 만들지 못해 LLM 검증을 건너뜀: %s", llm_error)
        return cls(labels, config, llm=llm, llm_error=llm_error)

    async def run(self, prediction: ClassificationResult) -> RoutingResult:
        """분류 결과 하나 → RoutingResult (② 경로 규칙 → ③ LLM 검증).

        예) result = await router.run(clf.predict("환불은 언제 되나요?"))
            result.label  →  'refund'
        """
        customer_inquiry = prediction.text  # 고객 문의 원문 (Classifier 입력 text → 이 모듈부터 customer_inquiry)
        classifier_label, scores = self.to_scores(prediction)
        route = self.apply_route_rule(scores) if self.config.route_rule_enabled else None
        label = route.label if route else classifier_label

        verification = None
        if self.needs_verification(label, route):
            v = await self.verify(customer_inquiry, label)
            verification = {**asdict(v), "provider": self.config.verify_provider,
                            "mode": self.config.verify_mode}
            label = v.final_label

        return RoutingResult(label=label, text=customer_inquiry, classifier_label=classifier_label,
                             scores=scores, route=route, verification=verification)

    def to_scores(self, prediction: ClassificationResult) -> tuple[str, dict[str, float]]:
        """ClassificationResult → (1등 라벨 이름, {라벨 이름: 확률}).
        labels와 결과가 어긋나면(라벨 수 · 번호 · 이름) 다른 모델의 라벨을 붙이는 것이므로 막는다."""
        index, probs = prediction.label_num, prediction.scores
        if len(probs) != len(self.labels) or not 0 <= index < len(self.labels):
            raise ValueError(f"분류 결과와 라벨 수가 맞지 않습니다: 번호 {index}, 확률 {len(probs)}개, "
                             f"라벨 {list(self.labels)}")
        if prediction.label_name != self.labels[index]:
            raise ValueError(f"분류 결과의 라벨 이름({prediction.label_name!r})이 labels[{index}]"
                             f"({self.labels[index]!r})와 다릅니다")
        return self.labels[index], {name: float(p) for name, p in zip(self.labels, probs, strict=True)}

    def apply_route_rule(self, scores: dict[str, float]) -> RouteDecision:
        """② 분류기 확률에서 p1 · margin을 구해 경로 규칙에 넘긴다. 확신 아님(ambiguous)도 1등 라벨을 쓴다."""
        (top1, p1), (top2, p2) = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
        margin = p1 - p2
        route = self.rule.route(p1, margin)
        return RouteDecision(
            route=route,
            label=UNKNOWN if route == UNKNOWN else top1,
            top1=top1,
            top2=top2,
            p1=round(p1, 4),
            p2=round(p2, 4),
            margin=round(margin, 4),
            reason=REASONS[route],
            tau=self.rule.tau,
            tau_low=self.rule.tau_low,
            delta=self.rule.delta,
        )

    def needs_verification(self, label: str, route: RouteDecision | None) -> bool:
        """all: unknown이 아닌 라벨 전부 / uncertain: 경로 규칙이 confident라고 하지 않은 것
        (경로 규칙을 꺼서 route가 없으면 confident 판단이 없으므로 uncertain도 전부 검증한다)"""
        mode = self.config.verify_mode
        if mode == "off" or label == UNKNOWN:
            return False
        if mode == "all":
            return True
        return route is None or route.route != CONFIDENT

    async def verify(self, customer_inquiry: str, label: str) -> llm_verifier.VerifyResult:
        """③ LLM 검증. 검증 LLM이 없거나(키 없음) 시간 초과여도 분류 라벨을 유지한다 (fail-open)."""
        if self.llm is None:
            return llm_verifier.VerifyResult(match=True, reason="", label=label, final_label=label,
                                             latency_ms=0, error=self.llm_error or NO_VERIFY_LLM)
        try:
            return await asyncio.wait_for(llm_verifier.verify(self.llm, customer_inquiry, label),
                                          self.config.verify_timeout_sec)
        except Exception as e:  # 시간 초과 등도 분류 라벨을 유지한다
            logger.exception("LLM 검증을 건너뜀 (label=%s, provider=%s)", label,
                             self.config.verify_provider)
            err = "시간 초과" if isinstance(e, TimeoutError) else f"{type(e).__name__}: {e}"
            return llm_verifier.VerifyResult(match=True, reason="", label=label, final_label=label,
                                             latency_ms=0, error=err)
