"""답변 생성: PromptSpec + provider → AnswerResult.

- LLM 분기(local · openai)는 client.make_llm 한 곳에서 한다
- 라벨에 켜진 툴이 있으면 tool_caller로 툴 호출 반복, 없으면 LLM 1회 호출
- 예외를 밖으로 던지지 않고 항상 AnswerResult를 돌려준다 (오류도 같은 형식)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from ..client import ConfigError, llm_options, make_llm, model_name
from ..config import LLMConfig, get_config
from ..schemas import AnswerResult, ErrorInfo, PromptSpec, ToolRecord, Usage
from . import tool_caller
from .tools import get_tools_for


def to_messages(spec: PromptSpec) -> list[BaseMessage]:
    return [SystemMessage(spec.system), HumanMessage(spec.user)]


def extract_text(ai: AIMessage) -> str:
    """content가 문자열이 아니라 블록 목록(Responses API, 추론 블록 등)일 수 있으므로 텍스트만 뽑는다."""
    content = ai.content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def now_iso(config: LLMConfig) -> str:
    return datetime.now(ZoneInfo(config.timezone)).isoformat(timespec="seconds")


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
    start = time.perf_counter()
    usage = Usage()
    records: list[ToolRecord] = []
    label_tools = get_tools_for(spec.label, config) if tools is None else tools

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

    try:
        llm = llm or make_llm(provider, config, with_tools=bool(label_tools))
        messages = to_messages(spec)

        if label_tools:
            outcome = tool_caller.ToolCallOutcome()
            usage, records = outcome.usage, outcome.records  # 도중에 예외가 나도 기록이 남는다
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
            return result(error=ErrorInfo(code="llm_error", message="모델이 빈 답변을 반환했습니다."))
        return result(answer=answer)

    except ConfigError as e:
        return result(error=ErrorInfo(code="config_error", message=str(e)))
    except TimeoutError:
        return result(error=ErrorInfo(
            code="llm_timeout", message=f"LLM 응답 시간 초과 ({config.llm_timeout_sec}초)"))
    except Exception as e:  # noqa: BLE001 — 연결 실패(Ollama 꺼짐), API 오류 등은 error JSON으로
        return result(error=ErrorInfo(code="llm_error", message=f"{type(e).__name__}: {e}"))
