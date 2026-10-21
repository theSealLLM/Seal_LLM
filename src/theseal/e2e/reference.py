"""검증용 데이터(run_e2e.py 2번): 정답 라벨의 서비스 프롬프트로 OpenAI가 만든 답변 + 그 답변의 채점 결과.

OpenAI 답변은 서비스와 같은 경로로 만든다.
    프롬프트  theseal.prompt.Prompt().for_label(정답 라벨, 문의)   → PromptSpec (system · user)
    답변      theseal.answer.answer_generator.generate(spec, "openai")  → system · user 메시지로 OpenAI 호출
서비스와 다른 점은 라벨(분류기 결과 대신 정답 라벨)과 LLM(로컬 대신 OpenAI) 두 가지다. 그래서 점수 차이는
분류 · 라우팅 · 로컬 LLM에서 나온다. 프롬프트 자체의 결함은 두 답변에 같이 들어가 차이로 드러나지 않는다.

OpenAI 답변의 점수는 서비스를 다시 테스트해도 바뀌지 않으므로 여기서 한 번 채점해 둔다 (문의 하나에 OpenAI 호출 2번:
답변 + 채점). 1번(서비스 테스트)은 우리 답변을 같은 채점표 · 같은 버전으로 채점해 이 점수와 비교한다.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from theseal.answer.answer_generator import generate
from theseal.answer.config import AnswerConfig
from theseal.answer.tools import get_tools_for
from theseal.common.config import LLMConfig
from theseal.prompt import Prompt

from .judge import JUDGE_VERSION, score_answer
from .rate_limit import RateLimiter
from .service import recording_tools

PROVIDER = "openai"


def prompt_for_label(label: str, customer_inquiry: str):
    """서비스와 같은 프롬프트 (분류 · 후처리 없이 라벨만 주므로 Router를 만들지 않는다)."""
    return Prompt().for_label(label, customer_inquiry)


def gold_label(case: dict) -> str:
    return case["expected_label"] or "unknown"


def reference_version(case: dict) -> str:
    """서비스 프롬프트 버전 + 채점 버전. 둘 중 하나가 바뀌면 검증용 데이터를 다시 만든다."""
    return f"{prompt_for_label(gold_label(case), case['text']).prompt_version}+{JUDGE_VERSION}"


async def make_reference(case: dict, *, config: LLMConfig, answer_config: AnswerConfig, llm: BaseChatModel,
                         limiter: RateLimiter) -> dict:
    """문의 하나의 검증용 데이터 한 줄. llm: make_llm("openai", config) 하나를 모든 호출이 같이 쓴다."""
    spec = prompt_for_label(gold_label(case), case["text"])
    # 서비스처럼 라벨에 켜진 툴을 쓰되 (unknown: 날짜 · 날씨 · 검색), 채점에 쓰도록 툴 결과를 남긴다
    tool_outputs: list[dict] = []
    tools = recording_tools(get_tools_for(spec.label, answer_config), tool_outputs)

    async def answer():
        tool_outputs.clear()  # 다시 보낼 때 앞 시도의 툴 결과가 섞이지 않게
        return await generate(spec, PROVIDER, config=config, answer_config=answer_config, llm=llm,
                              tools=tools)

    result = await limiter.call(
        answer, error_of=lambda r: None if r.status == "ok" else f"{r.error.code}: {r.error.message}",
        label=case["id"])
    score = await score_answer(case, result.answer, tool_outputs, llm=llm, limiter=limiter)
    return {
        "id": case["id"],
        "text": case["text"],
        "expected_label": case["expected_label"],
        "version": f"{spec.prompt_version}+{JUDGE_VERSION}",
        "provider": result.provider,
        "model": result.model,
        "answer": result.answer,
        "tools_used": [t.model_dump() for t in result.tools_used],
        "tool_outputs": tool_outputs,
        "usage": result.usage.model_dump(),
        "score": score,
    }
