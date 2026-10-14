"""툴 호출 반복: LLM 호출 → 툴 실행 → 결과를 붙여 다시 호출.

- LLM 객체는 answer_generator에서 받아 쓰고 직접 만들지 않는다
- max_rounds는 툴 실행 라운드 수. 툴 결과로 답을 쓸 LLM 호출 1회가 항상 더 허용된다
  (예: max_rounds=2 → get_today → add_days → 최종 답변까지 가능)
- 툴 하나가 실패해도 답변은 계속 만든다 (ToolRecord.ok=false)
- 토큰 사용량은 반복 안의 모든 LLM 호출을 합산한다
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..common.schemas import ToolRecord, Usage


@dataclass
class ToolCallOutcome:
    final: AIMessage | None = None  # 횟수 초과면 None
    records: list[ToolRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    exhausted: bool = False


def add_usage(total: Usage, ai: AIMessage) -> None:
    meta = getattr(ai, "usage_metadata", None)
    if not meta:
        return
    total.input_tokens = (total.input_tokens or 0) + meta.get("input_tokens", 0)
    total.output_tokens = (total.output_tokens or 0) + meta.get("output_tokens", 0)


async def _run_tool(tool: BaseTool | None, call: dict, timeout: float) -> tuple[str, ToolRecord]:
    name, args = call["name"], call.get("args") or {}
    start = time.perf_counter()
    try:
        if tool is None:
            raise LookupError(f"허용되지 않은 도구입니다: {name}")
        result = await asyncio.wait_for(tool.ainvoke(args), timeout)
        content = json.dumps(result, ensure_ascii=False)
        ok, error = True, None
    except TimeoutError:
        ok, error = False, f"도구 실행 시간 초과 ({timeout}초)"
        content = f"오류: {error}"
    except Exception as e:  # noqa: BLE001 — 인자 검증 실패, 잘못된 날짜, 외부 API 오류 등
        ok, error = False, f"{type(e).__name__}: {e}"
        content = f"오류: {error}"
    latency_ms = round((time.perf_counter() - start) * 1000)
    return content, ToolRecord(name=name, args=args, ok=ok, latency_ms=latency_ms, error=error)


async def run(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    *,
    max_rounds: int,
    tool_timeout: float,
    llm_timeout: float,
    out: ToolCallOutcome | None = None,
) -> ToolCallOutcome:
    """LLM 시간 초과(TimeoutError)와 LLM 오류는 호출한 쪽(answer_generator)이 처리한다.
    out을 넘기면 그 객체에 기록을 쌓으므로, 예외가 나도 호출한 쪽에 그때까지의 툴 기록 · 토큰이 남는다."""
    llm_t = llm.bind_tools(tools)
    allowed = {t.name: t for t in tools}
    msgs = list(messages)
    out = out or ToolCallOutcome()

    for round_no in range(max_rounds + 1):
        ai = await asyncio.wait_for(llm_t.ainvoke(msgs), llm_timeout)
        add_usage(out.usage, ai)
        msgs.append(ai)
        if not ai.tool_calls:
            out.final = ai
            return out
        if round_no == max_rounds:  # 마지막 기회에도 툴을 요청하면 횟수 초과
            break
        for call in ai.tool_calls:
            content, record = await _run_tool(allowed.get(call["name"]), call, tool_timeout)
            msgs.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
            out.records.append(record)

    out.exhausted = True
    return out
