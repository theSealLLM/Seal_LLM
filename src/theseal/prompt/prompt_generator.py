"""프롬프트 생성: 서비스가 쓰는 프롬프트는 이 파일 한 곳에서 관리한다.

Prompt().build_prompt(result)          분류 결과 → 분류 후처리(router) → 최종 라벨에 맞는 PromptSpec
Prompt().build_with_routing(result)    위와 같고 분류 후처리 결과(RoutingResult)도 함께 (API · LCEL 체인용)
Prompt().build_for_label(label, text)  라벨을 직접 줄 때 (분류 · 후처리 건너뜀)
build_common_policy_prompt(...)   shipping · refund · account: 공통 상담 규칙 + 라벨별 정책 (B, A/B 실험으로 선정)
build_label_unknown_prompt(...)   unknown(범위 밖): 툴(날짜 · 날씨 · 검색) 사용 지시 포함
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel

from ..common.config import LLMConfig, get_config
from ..common.schemas import Label, PromptSpec
from . import router


class RoutedPrompt(BaseModel):
    """프롬프트와 그 프롬프트를 고른 근거. API 응답의 classification이 routing에서 나온다."""

    routing: router.RoutingResult | None  # 분류 후처리 결과. 라벨을 직접 주면 None
    prompt: PromptSpec

    @property
    def label(self) -> str:
        return self.prompt.label


class Prompt:
    """분류 결과를 라우팅하고 최종 PromptSpec을 생성한다."""

    async def build_prompt(self, prediction) -> PromptSpec:
        """Classifier 분류 결과를 받아 PromptSpec을 반환한다."""
        return (await self.build_with_routing(prediction)).prompt

    async def build_with_routing(self, prediction) -> RoutedPrompt:
        """build_prompt와 같고, 분류 후처리 결과(RoutingResult)도 함께 돌려준다.
        라우팅(LLM 검증 포함)을 한 번만 하고 그 근거를 응답에 남길 때 쓴다 (API LCEL 체인)."""

        # 분류 결과를 후처리해 최종 라벨 결정
        routing_result = await router.run(prediction)

        # 최종 라벨과 원본 질문으로 PromptSpec 생성
        return RoutedPrompt(
            routing=routing_result,
            prompt=prompt_for_label(routing_result.label, routing_result.text),
        )

    def build_for_label(self, label: Label, text: str) -> RoutedPrompt:
        """라벨을 직접 줄 때 (분류 · 후처리를 건너뛴다)."""
        return RoutedPrompt(routing=None, prompt=prompt_for_label(label, text))


def prompt_for_label(label: Label, text: str) -> PromptSpec:
    """최종 라벨 → 프롬프트. unknown은 범위 밖 프롬프트(툴 사용), 나머지는 공통 정책 프롬프트(B).
    B는 프롬프트 A/B 실험(runs/llm/prompt_comparison, PR #6)에서 라벨별 프롬프트(A)보다 나아 선정됐다.
    Prompt 객체의 메서드들이 이 함수로 프롬프트를 고른다."""
    if label == "unknown":
        return build_label_unknown_prompt(text)
    return build_common_policy_prompt(text, label)
    
    
def build_common_policy_prompt(text: str, label: Label) -> PromptSpec:
    """공통 상담 지침과 라벨별 정책을 조합한 프롬프트를 생성한다."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("고객 질문은 공백이 아닌 문자열이어야 합니다.")

    policies = {
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

    if label not in policies:
        raise ValueError(
            "label은 확정된 라벨 shipping, refund, account 중 하나여야 합니다."
        )

    selected_policy = policies[label].strip()

    system = f"""
        [역할]
        당신은 가상 쇼핑 플랫폼의 고객 문의에 답변하는 고객지원 도우미다.

        [공통 상담 규칙]
        - 날짜와 금액을 추측하지 않는다.
        - 처리 완료 여부를 추측하지 않는다.
        - 실제 주문 조회, 환불 승인, 계정 변경 기능은 제공되지 않는다.
        - 실제로 확인하지 않은 내용을 확인한 것처럼 답하지 않는다.

        [유형별 정책]
        {selected_policy}

        [작성 요청]
        위 공통 상담 규칙과 유형별 정책을 준수하여 고객의 질문에 직접 답변한다. 확인되지 않은 정보는 임의로 추측하지 않는다. 확인이 필요한 경우 문의 유형과 유형별 정책에 맞는 공식 경로를 안내한다. 고객의 질문에 필요하지 않은 확인 경로를 기계적으로 추가하지 않는다.
        """

    return PromptSpec(
        label=label,
        system=system.strip(),
        user=text,
        prompt_version="common-policy-v1",
    )


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


def build_unknown_system(config: LLMConfig | None = None) -> str:
    config = config or get_config()
    return UNKNOWN_TEMPLATE.format(fallback=config.fallback_answer, scope=SCOPE_SENTENCE)


def unknown_prompt_version(system: str) -> str:
    """수동 버전 + 지시문 해시 앞 8자리. 지시문을 고치면 자동으로 바뀌어 비교 기록이 섞이지 않는다."""
    return f"{UNKNOWN_PROMPT_MAJOR_VERSION}-{hashlib.sha256(system.encode('utf-8')).hexdigest()[:8]}"


def build_label_unknown_prompt(text: str, *, config: LLMConfig | None = None) -> PromptSpec:
    """범위 밖 문의의 PromptSpec. 프롬프트 생성 모듈의 함수와 같이 text를 받는다."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("고객 질문은 공백이 아닌 문자열이어야 합니다.")
    system = build_unknown_system(config)
    return PromptSpec(label="unknown", system=system, user=text.strip(),
                      prompt_version=unknown_prompt_version(system))
