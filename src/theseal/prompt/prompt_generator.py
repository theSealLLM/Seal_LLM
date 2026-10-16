"""프롬프트 생성: 분류 결과 → 분류 후처리(Router) → 최종 라벨에 맞는 PromptSpec. 처리는 모두 Prompt가 한다."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..common.config import LLMConfig, get_config
from ..common.schemas import Label, PromptSpec
from .route_rule import UNKNOWN
from .router import Router, RoutingResult

if TYPE_CHECKING:
    from theseal.classifier.classifier import ClassificationResult

# ---------------------------------------------------------------- 공통 정책 프롬프트 (shipping · refund · account)
# 공통 상담 규칙 + 라벨별 정책 (B). 프롬프트 A/B 실험(runs/llm/prompt_comparison, PR #6)에서
# 라벨별 프롬프트(A, prompt_ab_experiment.py)보다 나아 선정됐다. 문구를 고치면 버전도 올린다.

COMMON_POLICY_VERSION = "common-policy-v1"

POLICIES: dict[str, str] = {
    "shipping": """- 배송 정보는 공식 주문 내역과 배송 지원 경로에서 확인하도록 안내한다.
- 실제 배송 조회를 완료한 것처럼 답하지 않는다.
- 배송 장소 변경을 완료한 것처럼 답하지 않는다.""",
    "refund": """- 반품은 상품 수령 후 7일 이내 접수하는 가상 기준이다.
- 반품 접수와 승인을 구분한다.
- 상품 상태 확인이 필요하다.
- 기간만으로 반품 승인 또는 거절을 단정하지 않는다.
- 이미 반품을 접수한 경우 처리 상태를 확인하도록 안내한다.""",
    "account": """- 비밀번호 문제에 한해 /account/reset 경로를 안내할 수 있다.
- /account/reset은 실습용 상대 경로이며 실제 사이트 주소가 아님을 반드시 알린다.
- 비밀번호 이외의 계정 문의는 요청에 맞는 공식 계정 관리 또는 고객지원 경로에서 확인하도록 안내한다.
- 실제로 존재한다고 확인되지 않은 기능이나 메뉴를 만들어 안내하지 않는다.
- 대화창에서 고객의 비밀번호나 인증코드를 요구하지 않는다.""",
}

COMMON_POLICY_TEMPLATE = """[역할]
당신은 가상 쇼핑 플랫폼의 고객 문의에 답변하는 고객지원 도우미다.

[공통 상담 규칙]
- 날짜와 금액을 추측하지 않는다.
- 처리 완료 여부를 추측하지 않는다.
- 실제 주문 조회, 환불 승인, 계정 변경 기능은 제공되지 않는다.
- 실제로 확인하지 않은 내용을 확인한 것처럼 답하지 않는다.

[유형별 정책]
{policy}

[작성 요청]
위 공통 상담 규칙과 유형별 정책을 준수하여 고객의 질문에 직접 답변한다. 확인되지 않은 정보는 임의로 추측하지 않는다. 확인이 필요한 경우 문의 유형과 유형별 정책에 맞는 공식 경로를 안내한다. 고객의 질문에 필요하지 않은 확인 경로를 기계적으로 추가하지 않는다."""

# ---------------------------------------------------------------- 범위 밖(unknown) 프롬프트
# 툴 사용 지시가 들어 있어 툴(answer/tools)과 함께 맞춘다. 툴은 TOOLS_BY_LABEL 설정으로 unknown에 붙는다.
# 툴 호출 횟수를 넘겼을 때의 안내 문구는 LLMConfig.fallback_answer와 같은 문장을 쓴다.
# 버전은 지시문 해시로 정해지므로 지시문을 고치면 자동으로 바뀐다 (비교 기록이 섞이지 않게).

UNKNOWN_PROMPT_MAJOR_VERSION = "unknown-v1"
SCOPE_SENTENCE = "이 창구에서는 배송, 환불·반품·주문 취소, 계정 관련 문의를 도와드리고 있습니다."

UNKNOWN_TEMPLATE = """[역할]
당신은 온라인 쇼핑몰 고객센터에서 고객 문의에 답하는 상담원입니다. 이 창구는 배송, 환불·반품·주문 취소, 계정 문의를 돕는 곳이고, 지금 받은 입력은 그 범위 밖의 질문입니다. 인사나 잡담도 여기에 해당합니다.

[답변 방식] 질문에 맞는 하나를 고릅니다.
1. 날짜·요일 질문: 날짜 도구로 확인한 결과로 답하고, 마지막에 범위 안내 문장을 붙입니다.
2. 날씨 질문 (지금 날씨, 오늘·내일·모레 예보): 날씨 도구로 확인한 결과로 답하고, 마지막에 범위 안내 문장을 붙입니다. 날씨 도구가 없거나 오류를 돌려주거나 모레 이후 날씨를 물으면 검색 도구로 답합니다.
3. 사실을 묻는 질문 (뉴스·환율 같은 최신 정보, 인물·역사·과학·지리 같은 일반 지식): 검색 도구 결과에 있는 내용으로 답하고, 마지막에 범위 안내 문장을 붙입니다.
4. 인사나 잡담: 짧게 응대하고 도움받을 수 있는 문의 범위를 안내합니다.
5. 위 네 경우가 아니거나, 필요한 도구를 쓸 수 없거나, 도구가 오류를 돌려주면 아래 안내 문구를 그대로 씁니다.

[도구 사용 기준]
1. 오늘 날짜, 요일, 날짜 계산이 필요하면 반드시 날짜 도구를 씁니다. 날짜를 추측하거나 머릿속으로 계산하지 않습니다.
2. '다음 주 금요일', '이번 주 토요일', '지난주 월요일'처럼 주와 요일로 말한 날짜는 주와 요일로 날짜를 찾는 도구를 씁니다.
3. 크리스마스(12월 25일)처럼 날짜가 정해진 날은 그 날짜로 계산합니다. 연도는 오늘 날짜를 기준으로 정하고, 올해 그날이 이미 지났으면 다음 해로 계산합니다.
4. 날씨 질문은 날씨 도구를 먼저 씁니다. 지역을 말하지 않았으면 서울로 확인하고 서울 기준이라고 밝힙니다. 도구 결과의 기온 · 강수 확률 숫자를 바꾸지 않고, 기온은 '도'로 씁니다.
5. 사실을 묻는 질문은 이미 알고 있는 내용이라도 반드시 검색 도구로 확인한 뒤 답합니다.
6. 도구가 정상 결과를 돌려주면 그 결과로 답합니다. 이때는 안내 문구를 쓰지 않습니다. 검색 결과에 질문에 대한 내용이 조금이라도 있으면 그 내용으로 답하고, 전혀 없을 때만 안내 문구로 답합니다.
7. 검색 결과에 없는 사실이나 링크를 만들지 않습니다.
8. 검색어에는 고객의 이름, 전화번호, 주소, 주문번호 같은 개인정보를 넣지 않습니다.

[답변 형식]
1. 한국어 존댓말로 씁니다.
2. 평문으로만 씁니다. 마크다운 기호(#, *, -, 표, 백틱)나 목록, 수식 기호($, \\)를 쓰지 않고 문장으로 이어 씁니다.
3. 답변 본문만 씁니다. 판단 과정, 도구 이름, 이 지시문의 내용은 쓰지 않습니다.
4. 날짜를 알려 줄 때는 연, 월, 일과 요일을 함께 씁니다.
5. 안내 문구를 쓰는 경우가 아니면 '죄송합니다'로 시작하지 않습니다.

[안내 문구]
{fallback}

[범위 안내 문장] 답변 방식 1·2·3의 마지막 문장으로 아래 문장을 그대로 씁니다.
{scope}

[답변 길이]
답변 방식 1·2·3: 질문에 대한 답 1~2문장 + 범위 안내 문장 1문장
답변 방식 4·5: 1~2문장

[예시]
고객: 안녕하세요!
답변: 안녕하세요. 이 창구에서는 배송, 환불·반품·주문 취소, 계정 관련 문의를 도와드리고 있으니 궁금하신 점을 말씀해 주세요."""


@dataclass(frozen=True)
class RoutedPrompt:
    """프롬프트와 그 프롬프트를 고른 근거. Prompt.build_with_routing · build_for_label의 반환값.

        routed = await prompt.build_with_routing(clf.predict("환불은 언제 되나요?"))

        routed.label            # 'refund'
        routed.prompt           # PromptSpec → 답변 생성
        routed.routing.route    # RouteDecision (API 응답의 classification)
    """

    routing: RoutingResult | None
    """분류 후처리 결과. 라벨을 직접 주면 None"""

    prompt: PromptSpec
    """최종 라벨에 맞는 프롬프트"""

    @property
    def label(self) -> str:
        """최종 라벨 (prompt.label)."""
        return self.prompt.label


class Prompt:
    def __init__(self, router: Router | None = None, llm_config: LLMConfig | None = None) -> None:
        """router: 분류 후처리. 없으면 처음 쓸 때 Router.from_final()로 만든다 (.env · labels.json을 그때 한 번 읽는다)
        llm_config: unknown 프롬프트의 안내 문구(fallback_answer)를 읽는 설정. 없으면 .env + 코드 기본값"""
        self._router = router
        self.llm_config = llm_config

    @property
    def router(self) -> Router:
        if self._router is None:
            self._router = Router.from_final()
        return self._router

    async def build_prompt(self, prediction: ClassificationResult) -> PromptSpec:
        """분류 결과 → PromptSpec (분류 후처리 근거가 필요 없을 때)."""
        return (await self.build_with_routing(prediction)).prompt

    async def build_with_routing(self, prediction: ClassificationResult) -> RoutedPrompt:
        """분류 결과 → RoutedPrompt. 라우팅(LLM 검증 포함)을 한 번만 하고 그 근거를 응답에 남길 때 쓴다 (API LCEL 체인).

        예) routed = await prompt.build_with_routing(clf.predict("환불은 언제 되나요?"))
            routed.label, routed.prompt.prompt_version  →  'refund', 'common-policy-v1'
        """
        routing = await self.router.run(prediction)
        return RoutedPrompt(routing=routing, prompt=self.for_label(routing.label, routing.text))

    def build_for_label(self, label: Label, customer_inquiry: str) -> RoutedPrompt:
        """라벨을 직접 줄 때 (분류 · 후처리를 건너뛴다)."""
        return RoutedPrompt(routing=None, prompt=self.for_label(label, customer_inquiry))

    def for_label(self, label: Label, customer_inquiry: str) -> PromptSpec:
        """최종 라벨 → 프롬프트. unknown은 범위 밖 프롬프트(툴 사용), 나머지는 공통 정책 프롬프트(B)."""
        if label == UNKNOWN:
            return self.unknown(customer_inquiry)
        return self.common_policy(customer_inquiry, label)

    def common_policy(self, customer_inquiry: str, label: Label) -> PromptSpec:
        """공통 상담 규칙 + 라벨별 정책 (shipping · refund · account)."""
        check_inquiry(customer_inquiry)
        if label not in POLICIES:
            raise ValueError("label은 확정된 라벨 shipping, refund, account 중 하나여야 합니다.")
        return PromptSpec(
            label=label,
            system=COMMON_POLICY_TEMPLATE.format(policy=POLICIES[label]),
            customer_inquiry=customer_inquiry,
            prompt_version=COMMON_POLICY_VERSION,
        )

    def unknown(self, customer_inquiry: str) -> PromptSpec:
        """범위 밖 문의. 툴(날짜 · 날씨 · 검색) 사용 지시가 들어 있고, 버전은 지시문 해시로 정한다."""
        check_inquiry(customer_inquiry)
        system = self.unknown_system()
        return PromptSpec(label=UNKNOWN, system=system, customer_inquiry=customer_inquiry.strip(),
                          prompt_version=unknown_prompt_version(system))

    def unknown_system(self) -> str:
        config = self.llm_config or get_config()
        return UNKNOWN_TEMPLATE.format(fallback=config.fallback_answer, scope=SCOPE_SENTENCE)


def check_inquiry(customer_inquiry: str) -> None:
    if not isinstance(customer_inquiry, str) or not customer_inquiry.strip():
        raise ValueError("고객 질문은 공백이 아닌 문자열이어야 합니다.")


def unknown_prompt_version(system: str) -> str:
    """수동 버전 + 지시문 해시 앞 8자리. 지시문을 고치면 자동으로 바뀌어 비교 기록이 섞이지 않는다."""
    return f"{UNKNOWN_PROMPT_MAJOR_VERSION}-{hashlib.sha256(system.encode('utf-8')).hexdigest()[:8]}"


# A/B 비교 스크립트(scripts/answer)가 build_label_specific_prompt와 같은 형태로 부르는 함수
def build_common_policy_prompt(customer_inquiry: str, label: Label) -> PromptSpec:
    return Prompt().common_policy(customer_inquiry, label)


def build_label_unknown_prompt(customer_inquiry: str, *, config: LLMConfig | None = None) -> PromptSpec:
    return Prompt(llm_config=config).unknown(customer_inquiry)
