"""답변 생성: PromptSpec + provider → AnswerResult. 처리는 모두 AnswerGenerator가 한다."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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
from .tool_caller import ToolCaller, ToolCallOutcome, add_usage
from .tools import get_tools_for

EMPTY_ANSWER = ErrorInfo(code="llm_error", message="모델이 빈 답변을 반환했습니다.")


@dataclass
class AnswerRun:
    """답변 하나를 만드는 동안 쌓이는 값. 툴 기록 · 토큰은 도중에 예외가 나도 결과에 남도록 여기에 쌓는다."""

    spec: PromptSpec
    provider: str
    tools: list[BaseTool]
    """이 답변에 쓰는 툴. 비어 있으면 LLM 1회 호출"""

    start: float = field(default_factory=time.perf_counter)
    """지연(latency_ms)을 재는 시작 시각"""

    usage: Usage = field(default_factory=Usage)
    records: list[ToolRecord] = field(default_factory=list)


class AnswerGenerator:
    """라벨에 켜진 툴이 있으면 툴 호출 반복(ToolCaller), 없으면 LLM 1회 호출로 답한다.
    LLM 분기(local · openai)는 client.make_llm 한 곳에서 하고, 예외를 밖으로 던지지 않고 항상 AnswerResult를 돌려준다.

        result = await AnswerGenerator().generate(spec, "local")
        result.status, result.answer  →  'ok', '배송 안내입니다 …'
    """

    def __init__(self, config: LLMConfig | None = None, *, llm: BaseChatModel | None = None,
                 tools: list[BaseTool] | None = None) -> None:
        """config: 없으면 .env + 코드 기본값
        llm: 넘기면 provider와 상관없이 그 객체를 쓴다 (테스트용 흉내 객체 등)
        tools: None이면 라벨별 설정(TOOLS_BY_LABEL)을 따른다. []이면 툴 없이 답한다"""
        self.config = config or get_config()
        self.llm = llm
        self.tools = tools

    async def generate(self, spec: PromptSpec, provider: str | None = None) -> AnswerResult:
        """답변 하나. provider: local · openai (없으면 DEFAULT_PROVIDER)"""
        config = self.config
        run = self._start(spec, provider)
        deadline: asyncio.Timeout | None = None
        try:
            llm = self.llm or make_llm(run.provider, config, with_tools=bool(run.tools))
            messages = to_messages(spec)
            async with asyncio.timeout(config.answer_timeout_sec) as deadline:  # 호출 · 툴을 다 합친 상한
                if run.tools:
                    outcome = ToolCallOutcome(records=run.records, usage=run.usage)
                    await ToolCaller.from_config(llm, run.tools, config).run(messages, outcome)
                    if outcome.exhausted:
                        return self._result(run, answer=config.fallback_answer)
                    ai = outcome.final
                else:
                    ai = await asyncio.wait_for(llm.ainvoke(messages), config.llm_timeout_sec)
                    add_usage(run.usage, ai)

            answer = extract_text(ai)
            return self._result(run, answer=answer) if answer else self._result(run, error=EMPTY_ANSWER)

        except Exception as e:  # noqa: BLE001 — 연결 실패(Ollama 꺼짐), API 오류 등은 error JSON으로
            if isinstance(e, TimeoutError) and deadline is not None and deadline.expired():
                return self._result(run, error=ErrorInfo(
                    code="llm_timeout", message=f"답변 생성 전체 시간 초과 ({config.answer_timeout_sec}초)"))
            return self._result(run, error=error_info(e, config))

    async def stream(self, spec: PromptSpec, provider: str | None = None) -> AsyncIterator[dict]:
        """generate와 같은 답변을 이벤트로 흘려보낸다.

        {"type": "delta", "text": "..."}  답변 글자 조각 (툴을 쓰는 경로는 보내지 않는다)
        {"type": "done", "result": AnswerResult}  마지막 한 번. answer는 조각을 이어 앞뒤 공백을 지운 전체 답변
        툴을 쓰는 경로는 툴 결과를 본 뒤에야 답이 정해지므로 generate 결과만 보낸다.
        도중에 오류가 나면 done의 result가 status=error다 (이미 보낸 조각은 버린다)
        """
        run = self._start(spec, provider)
        if run.tools:
            yield {"type": "done", "result": await self.generate(spec, provider)}
            return
        try:
            llm = self.llm or make_llm(run.provider, self.config, with_tools=False)
            merged: AIMessageChunk | None = None
            async with asyncio.timeout(self.config.llm_timeout_sec):
                async for chunk in llm.astream(to_messages(spec)):
                    merged = chunk if merged is None else merged + chunk
                    text = content_text(chunk.content)
                    if text:
                        yield {"type": "delta", "text": text}
            if merged is not None:
                add_usage(run.usage, merged)
            answer = extract_text(merged) if merged is not None else ""
            yield {"type": "done",
                   "result": self._result(run, answer=answer) if answer else self._result(run, error=EMPTY_ANSWER)}
        except Exception as e:  # noqa: BLE001 — generate와 같은 오류 형식
            yield {"type": "done", "result": self._result(run, error=error_info(e, self.config))}

    def _start(self, spec: PromptSpec, provider: str | None) -> AnswerRun:
        tools = get_tools_for(spec.label, self.config) if self.tools is None else self.tools
        return AnswerRun(spec=spec, provider=provider or self.config.default_provider, tools=tools)

    def _result(self, run: AnswerRun, answer: str | None = None, error: ErrorInfo | None = None) -> AnswerResult:
        config = self.config
        return AnswerResult(
            request_id=str(uuid.uuid4()),
            status="error" if error else "ok",
            label=run.spec.label,
            answer=answer,
            provider=run.provider,
            model=model_name(run.provider, config),
            prompt_version=run.spec.prompt_version,
            tools_used=run.records,
            latency_ms=round((time.perf_counter() - run.start) * 1000),
            usage=run.usage,
            llm_options=llm_options(run.provider, config, with_tools=bool(run.tools)),
            error=error,
            created_at=datetime.now(ZoneInfo(config.timezone)).isoformat(timespec="seconds"),
        )


def to_messages(spec: PromptSpec) -> list[BaseMessage]:
    return [SystemMessage(spec.system), HumanMessage(spec.customer_inquiry)]


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


def error_info(e: Exception, config: LLMConfig) -> ErrorInfo:
    if isinstance(e, ConfigError):
        return ErrorInfo(code="config_error", message=str(e))
    if isinstance(e, TimeoutError):
        return ErrorInfo(code="llm_timeout", message=f"LLM 응답 시간 초과 ({config.llm_timeout_sec}초)")
    return ErrorInfo(code="llm_error", message=f"{type(e).__name__}: {e}")


# API · E2E · 비교 스크립트가 부르는 함수 (AnswerGenerator를 그때그때 만든다)
async def generate(spec: PromptSpec, provider: str | None = None, *, config: LLMConfig | None = None,
                   llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None) -> AnswerResult:
    return await AnswerGenerator(config, llm=llm, tools=tools).generate(spec, provider)


async def generate_stream(spec: PromptSpec, provider: str | None = None, *, config: LLMConfig | None = None,
                          llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None
                          ) -> AsyncIterator[dict]:
    async for event in AnswerGenerator(config, llm=llm, tools=tools).stream(spec, provider):
        yield event
