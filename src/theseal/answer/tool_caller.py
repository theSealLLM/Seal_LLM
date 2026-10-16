"""툴 호출 반복: LLM 호출 → 툴 실행 → 결과를 붙여 다시 호출. 처리는 ToolCaller가 한다."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..common.config import LLMConfig
from ..common.schemas import ToolRecord, Usage


@dataclass
class ToolCallOutcome:
    """툴 호출 반복의 결과. 도중에 예외가 나도 그때까지의 툴 기록 · 토큰이 남도록 반복하면서 채운다 (frozen 아님)."""

    final: AIMessage | None = None
    """툴을 더 부르지 않은 마지막 LLM 응답. 횟수 초과면 None"""

    records: list[ToolRecord] = field(default_factory=list)
    """실행한 툴 기록 (실패도 ok=false로 남는다)"""

    usage: Usage = field(default_factory=Usage)
    """반복 안의 모든 LLM 호출 토큰 합"""

    exhausted: bool = False
    """마지막 기회에도 툴을 요청해 횟수를 넘겼는지"""


class ToolCaller:
    """max_rounds는 툴 실행 라운드 수. 툴 결과로 답을 쓸 LLM 호출 1회가 항상 더 허용된다
    (예: max_rounds=2 → get_today → add_days → 최종 답변까지 가능). 툴 하나가 실패해도 답변은 계속 만든다.

        outcome = await ToolCaller.from_config(llm, tools, config).run(messages)
        outcome.final, outcome.records  →  AIMessage, [ToolRecord, …]
    """

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool], *, max_rounds: int, tool_timeout: float,
                 llm_timeout: float) -> None:
        """llm: AnswerGenerator가 만든 LLM (여기서 직접 만들지 않는다)"""
        self.llm = llm.bind_tools(tools)
        self.allowed = {t.name: t for t in tools}
        self.max_rounds = max_rounds
        self.tool_timeout = tool_timeout
        self.llm_timeout = llm_timeout

    @classmethod
    def from_config(cls, llm: BaseChatModel, tools: list[BaseTool], config: LLMConfig) -> ToolCaller:
        """횟수 · 시간 제한을 설정(MAX_TOOL_ROUNDS · TOOL_TIMEOUT_SEC · LLM_TIMEOUT_SEC)에서 읽는다."""
        return cls(llm, tools, max_rounds=config.max_tool_rounds, tool_timeout=config.tool_timeout_sec,
                   llm_timeout=config.llm_timeout_sec)

    async def run(self, messages: list[BaseMessage], out: ToolCallOutcome | None = None) -> ToolCallOutcome:
        """LLM 시간 초과(TimeoutError)와 LLM 오류는 호출한 쪽(AnswerGenerator)이 처리한다.
        out: 기록을 쌓을 객체. 넘기면 예외가 나도 호출한 쪽에 그때까지의 툴 기록 · 토큰이 남는다."""
        out = out or ToolCallOutcome()
        msgs = list(messages)
        for round_no in range(self.max_rounds + 1):
            ai = await asyncio.wait_for(self.llm.ainvoke(msgs), self.llm_timeout)
            add_usage(out.usage, ai)
            msgs.append(ai)
            if not ai.tool_calls:
                out.final = ai
                return out
            if round_no == self.max_rounds:  # 마지막 기회에도 툴을 요청하면 횟수 초과
                break
            for call in ai.tool_calls:
                content, record = await self.run_tool(call)
                msgs.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
                out.records.append(record)
        out.exhausted = True
        return out

    async def run_tool(self, call: dict) -> tuple[str, ToolRecord]:
        """툴 하나 실행 → (LLM에 돌려줄 내용, 기록). 허용되지 않은 툴 · 인자 오류 · 시간 초과도 기록하고 계속한다."""
        name, args = call["name"], call.get("args") or {}
        tool = self.allowed.get(name)
        start = time.perf_counter()
        try:
            if tool is None:
                raise LookupError(f"허용되지 않은 도구입니다: {name}")
            result = await asyncio.wait_for(tool.ainvoke(args), self.tool_timeout)
            content = json.dumps(result, ensure_ascii=False)
            ok, error = True, None
        except TimeoutError:
            ok, error = False, f"도구 실행 시간 초과 ({self.tool_timeout}초)"
            content = f"오류: {error}"
        except Exception as e:  # noqa: BLE001 — 인자 검증 실패, 잘못된 날짜, 외부 API 오류 등
            ok, error = False, f"{type(e).__name__}: {e}"
            content = f"오류: {error}"
        latency_ms = round((time.perf_counter() - start) * 1000)
        return content, ToolRecord(name=name, args=args, ok=ok, latency_ms=latency_ms, error=error)


def add_usage(total: Usage, ai: AIMessage) -> None:
    """LLM 응답의 토큰 수를 total에 더한다."""
    meta = getattr(ai, "usage_metadata", None)
    if not meta:
        return
    total.input_tokens = (total.input_tokens or 0) + meta.get("input_tokens", 0)
    total.output_tokens = (total.output_tokens or 0) + meta.get("output_tokens", 0)
