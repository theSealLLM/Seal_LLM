"""프롬프트 생성: 분류 결과 → 분류 후처리(Router) → 최종 라벨에 맞는 PromptSpec. 처리는 모두 Prompt가 한다.

프롬프트 문구는 코드가 아니라 templates/ 아래 문서 두 개에 있다. 문구를 고칠 때는 문서만 고친다.
    templates/common_policy.toml   공통 정책 프롬프트 (shipping · refund · account): 지시문 + 라벨별 정책 + 버전
    templates/unknown.toml         범위 밖(unknown) 프롬프트: 지시문 + 범위 안내 문장 + 버전
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from theseal.common.config import LLMConfig
from theseal.common.schemas import Label, PromptSpec
from theseal.prompt.route_rule import UNKNOWN
from theseal.prompt.router import Router, RoutingResult

if TYPE_CHECKING:
    from theseal.classifier.classifier import ClassificationResult

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
COMMON_POLICY_FILE = "common_policy.toml"
UNKNOWN_FILE = "unknown.toml"


@dataclass(frozen=True)
class CommonPolicyTemplate:
    """공통 정책 프롬프트 문서(common_policy.toml)의 내용."""

    version: str
    """프롬프트 버전 (PromptSpec.prompt_version). 문구를 고치면 문서에서 같이 올린다"""

    template: str
    """지시문. {policy} 자리에 라벨별 정책이 들어간다"""

    policies: Mapping[str, str]
    """라벨 → 정책 문구 (shipping · refund · account)"""

    @classmethod
    def from_file(cls, path: Path = TEMPLATE_DIR / COMMON_POLICY_FILE) -> CommonPolicyTemplate:
        """문서를 읽는다. 항목이 빠졌으면 KeyError."""
        data = _read_toml(path)
        return cls(version=data["version"], template=data["template"],
                   policies=MappingProxyType(dict(data["policies"])))


@dataclass(frozen=True)
class UnknownTemplate:
    """범위 밖(unknown) 프롬프트 문서(unknown.toml)의 내용."""

    major_version: str
    """수동 버전. 실제 버전은 여기에 채운 지시문의 해시를 붙인다"""

    scope_sentence: str
    """답변 끝에 붙이는 범위 안내 문장. 지시문의 {scope} 자리에 들어간다"""

    template: str
    """지시문. {fallback}(안내 문구) · {scope}(범위 안내 문장) 자리를 채워 쓴다"""

    @classmethod
    def from_file(cls, path: Path = TEMPLATE_DIR / UNKNOWN_FILE) -> UnknownTemplate:
        """문서를 읽는다. 항목이 빠졌으면 KeyError."""
        data = _read_toml(path)
        return cls(major_version=data["major_version"], scope_sentence=data["scope_sentence"],
                   template=data["template"])


@dataclass(frozen=True)
class PromptTemplates:
    """프롬프트 문서 두 개. Prompt가 만들 때 한 번 읽어 들고 있는다."""

    common_policy: CommonPolicyTemplate
    unknown: UnknownTemplate

    @classmethod
    def from_dir(cls, directory: Path = TEMPLATE_DIR) -> PromptTemplates:
        """폴더에서 common_policy.toml · unknown.toml을 읽는다."""
        return cls(common_policy=CommonPolicyTemplate.from_file(directory / COMMON_POLICY_FILE),
                   unknown=UnknownTemplate.from_file(directory / UNKNOWN_FILE))


def _read_toml(path: Path) -> dict[str, Any]:
    # read_text는 줄바꿈을 \n으로 맞춘다. Windows에서 CRLF로 체크아웃돼도 문구 · 버전 해시가 같다
    return tomllib.loads(path.read_text(encoding="utf-8"))


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
    """분류 결과 · 라벨에 맞는 프롬프트(PromptSpec)를 만든다. 분류 결과를 받으면 Router로 분류 후처리를 먼저 한다.

        prompt = Prompt.from_env()               # 서비스 구성 (Router 포함)
        routed = await prompt.build_with_routing(clf.predict("환불은 언제 되나요?"))
        Prompt().for_label("refund", "환불은 언제 되나요?")   # 라벨을 알면 Router 없이

    문구는 만들 때 templates/의 문서 두 개(PromptTemplates)에서 한 번 읽는다.

    Args:
      router: 분류 후처리. 없으면 라벨을 직접 주는 메서드(for_label · build_for_label)만 쓸 수 있다.
      llm_config: unknown 프롬프트의 안내 문구(fallback_answer). 없으면 .env + 코드 기본값.
    """

    def __init__(self, router: Router | None = None, llm_config: LLMConfig | None = None) -> None:
        self.router = router
        self.llm_config = llm_config or LLMConfig.from_env()
        self.templates = PromptTemplates.from_dir()

    @classmethod
    def from_env(cls, *, llm_config: LLMConfig | None = None) -> Prompt:
        """서비스 기본 구성: Router.from_env()(labels.json · .env)를 붙인다. llm_config는 Router 검증 LLM도 같이 쓴다."""
        llm_config = llm_config or LLMConfig.from_env()
        return cls(Router.from_env(llm_config=llm_config), llm_config)

    async def build_prompt(self, prediction: ClassificationResult) -> PromptSpec:
        """분류 결과 → PromptSpec (분류 후처리 근거가 필요 없을 때)."""
        return (await self.build_with_routing(prediction)).prompt

    async def build_with_routing(self, prediction: ClassificationResult) -> RoutedPrompt:
        """분류 결과 → RoutedPrompt. 라우팅(LLM 검증 포함)을 한 번만 하고 그 근거를 응답에 남길 때 쓴다 (API LCEL 체인).

        예) routed = await prompt.build_with_routing(clf.predict("환불은 언제 되나요?"))
            routed.label, routed.prompt.prompt_version  →  'refund', 'common-policy-v1'

        Raises:
          RuntimeError: Router 없이 만든 Prompt일 때 (Prompt.from_env()로 만든다).
        """
        if self.router is None:
            raise RuntimeError("분류 후처리에는 Router가 필요합니다 (Prompt.from_env() 또는 Prompt(router=...))")
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
        doc = self.templates.common_policy
        if label not in doc.policies:
            raise ValueError("label은 확정된 라벨 shipping, refund, account 중 하나여야 합니다.")
        return PromptSpec(
            label=label,
            system=doc.template.format(policy=doc.policies[label]),
            customer_inquiry=customer_inquiry,
            prompt_version=doc.version,
        )

    def unknown(self, customer_inquiry: str) -> PromptSpec:
        """범위 밖 문의. 툴(날짜 · 날씨 · 검색) 사용 지시가 들어 있고, 버전은 지시문 해시로 정한다."""
        check_inquiry(customer_inquiry)
        system = self.unknown_system()
        version = unknown_prompt_version(self.templates.unknown.major_version, system)
        return PromptSpec(label=UNKNOWN, system=system, customer_inquiry=customer_inquiry.strip(),
                          prompt_version=version)

    def unknown_system(self) -> str:
        """unknown 지시문 (안내 문구 · 범위 안내 문장을 채운 unknown.toml의 template)."""
        doc = self.templates.unknown
        return doc.template.format(fallback=self.llm_config.fallback_answer, scope=doc.scope_sentence)


def check_inquiry(customer_inquiry: str) -> None:
    """고객 질문이 공백이 아닌 문자열인지 확인한다. 아니면 ValueError."""
    if not isinstance(customer_inquiry, str) or not customer_inquiry.strip():
        raise ValueError("고객 질문은 공백이 아닌 문자열이어야 합니다.")


def unknown_prompt_version(major_version: str, system: str) -> str:
    """수동 버전 + 지시문 해시 앞 8자리. 지시문을 고치면 자동으로 바뀌어 비교 기록이 섞이지 않는다."""
    return f"{major_version}-{hashlib.sha256(system.encode('utf-8')).hexdigest()[:8]}"


# A/B 비교 스크립트(scripts/answer)가 build_label_specific_prompt와 같은 형태로 부르는 함수
def build_common_policy_prompt(customer_inquiry: str, label: Label) -> PromptSpec:
    """`Prompt().common_policy()`의 단축 함수 (공통 정책 프롬프트 B)."""
    return Prompt().common_policy(customer_inquiry, label)


def build_label_unknown_prompt(customer_inquiry: str, *, config: LLMConfig | None = None) -> PromptSpec:
    """`Prompt(llm_config=config).unknown()`의 단축 함수 (범위 밖 프롬프트)."""
    return Prompt(llm_config=config).unknown(customer_inquiry)
