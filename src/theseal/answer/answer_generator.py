"""답변 생성: PromptSpec + provider → AnswerResult.

- LLM 분기(local · openai)는 client.make_llm 한 곳에서 한다
- 라벨에 켜진 툴이 있으면 tool_caller로 툴 호출 반복, 없으면 LLM 1회 호출
- 예외를 밖으로 던지지 않고 항상 AnswerResult를 돌려준다 (오류도 같은 형식)
- generate_stream: 같은 답변을 글자 조각(delta)으로 흘려보낸다. 툴을 쓰는 경로는 툴 결과를 본 뒤에야
  답이 정해지므로 스트리밍하지 않고 generate 결과만 보낸다
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool

from ..common.client import ConfigError, llm_options, make_llm, model_name
from ..common.config import LLMConfig, get_config
from ..common.schemas import AnswerResult, ErrorInfo, PromptSpec, ToolRecord, Usage
from . import tool_caller
from .tools import get_tools_for


def to_messages(spec: PromptSpec) -> list[BaseMessage]:
    return [SystemMessage(spec.system), HumanMessage(spec.user)]


def content_text(content: Any) -> str:
    """content가 문자열이 아니라 블록 목록(Responses API, 추론 블록 등)일 수 있으므로 텍스트만 뽑는다."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def extract_text(ai: AIMessage) -> str:
    return content_text(ai.content).strip()


def now_iso(config: LLMConfig) -> str:
    return datetime.now(ZoneInfo(config.timezone)).isoformat(timespec="seconds")


def result_builder(spec: PromptSpec, provider: str, config: LLMConfig, label_tools: list[BaseTool],
                   usage: Usage, records: list[ToolRecord]):
    """AnswerResult를 만드는 함수를 돌려준다. 시작 시각부터 지연을 잰다."""
    start = time.perf_counter()

    def result(answer: str | None = None, error: ErrorInfo | None = None) -> AnswerResult:
        return AnswerResult(
            request_id=str(uuid.uuid4()),
            status="error" if error else "ok",
            label=spec.label,
            answer=answer,
            provider=provider,
            model=model_name(provider, config),
            prompt_version=spec.prompt_version,
            tools_used=records,
            latency_ms=round((time.perf_counter() - start) * 1000),
            usage=usage,
            llm_options=llm_options(provider, config, with_tools=bool(label_tools)),
            error=error,
            created_at=now_iso(config),
        )

    return result


def error_info(e: Exception, config: LLMConfig) -> ErrorInfo:
    if isinstance(e, ConfigError):
        return ErrorInfo(code="config_error", message=str(e))
    if isinstance(e, TimeoutError):
        return ErrorInfo(code="llm_timeout", message=f"LLM 응답 시간 초과 ({config.llm_timeout_sec}초)")
    return ErrorInfo(code="llm_error", message=f"{type(e).__name__}: {e}")


EMPTY_ANSWER = ErrorInfo(code="llm_error", message="모델이 빈 답변을 반환했습니다.")


async def generate(
    spec: PromptSpec,
    provider: str | None = None,
    *,
    config: LLMConfig | None = None,
    llm: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
) -> AnswerResult:
    """답변을 만든다.

    provider: local · openai (없으면 DEFAULT_PROVIDER)
    llm: 넘기면 그 객체를 쓴다 (테스트용 흉내 객체 등)
    tools: None이면 라벨별 설정(TOOLS_BY_LABEL)을 따른다. []이면 툴 없이 답한다
    """
    config = config or get_config()
    provider = provider or config.default_provider
    usage = Usage()
    records: list[ToolRecord] = []
    label_tools = get_tools_for(spec.label, config) if tools is None else tools
    result = result_builder(spec, provider, config, label_tools, usage, records)
    deadline: asyncio.Timeout | None = None

    try:
        llm = llm or make_llm(provider, config, with_tools=bool(label_tools))
        messages = to_messages(spec)

        async with asyncio.timeout(config.answer_timeout_sec) as deadline:  # 호출 · 툴을 다 합친 상한
            if label_tools:
                # 도중에 예외가 나도 기록이 남는다
                outcome = tool_caller.ToolCallOutcome(records=records, usage=usage)
                await tool_caller.run(
                    llm,
                    messages,
                    label_tools,
                    max_rounds=config.max_tool_rounds,
                    tool_timeout=config.tool_timeout_sec,
                    llm_timeout=config.llm_timeout_sec,
                    out=outcome,
                )
                if outcome.exhausted:
                    return result(answer=config.fallback_answer)
                ai = outcome.final
            else:
                ai = await asyncio.wait_for(llm.ainvoke(messages), config.llm_timeout_sec)
                tool_caller.add_usage(usage, ai)

        answer = extract_text(ai)
        if not answer:
            return result(error=EMPTY_ANSWER)
        return result(answer=answer)

    except Exception as e:  # noqa: BLE001 — 연결 실패(Ollama 꺼짐), API 오류 등은 error JSON으로
        if isinstance(e, TimeoutError) and deadline is not None and deadline.expired():
            return result(error=ErrorInfo(code="llm_timeout",
                                          message=f"답변 생성 전체 시간 초과 ({config.answer_timeout_sec}초)"))
        return result(error=error_info(e, config))


async def generate_stream(
    spec: PromptSpec,
    provider: str | None = None,
    *,
    config: LLMConfig | None = None,
    llm: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
) -> AsyncIterator[dict]:
    """generate와 같은 답변을 이벤트로 흘려보낸다.

    {"type": "delta", "text": "..."}  답변 글자 조각 (툴을 쓰는 경로는 보내지 않는다)
    {"type": "done", "result": AnswerResult}  마지막 한 번. answer는 조각을 이어 앞뒤 공백을 지운 전체 답변
    도중에 오류가 나면 done의 result가 status=error다 (이미 보낸 조각은 버린다)
    """
    config = config or get_config()
    provider = provider or config.default_provider
    label_tools = get_tools_for(spec.label, config) if tools is None else tools
    if label_tools:
        yield {"type": "done", "result": await generate(spec, provider, config=config, llm=llm, tools=label_tools)}
        return

    usage = Usage()
    result = result_builder(spec, provider, config, label_tools, usage, [])
    try:
        llm = llm or make_llm(provider, config, with_tools=False)
        merged: AIMessageChunk | None = None
        async with asyncio.timeout(config.llm_timeout_sec):
            async for chunk in llm.astream(to_messages(spec)):
                merged = chunk if merged is None else merged + chunk
                text = content_text(chunk.content)
                if text:
                    yield {"type": "delta", "text": text}
        if merged is not None:
            tool_caller.add_usage(usage, merged)
        answer = extract_text(merged) if merged is not None else ""
        yield {"type": "done", "result": result(answer=answer) if answer else result(error=EMPTY_ANSWER)}
    except Exception as e:  # noqa: BLE001 — generate와 같은 오류 형식
        yield {"type": "done", "result": result(error=error_info(e, config))}
