"""툴 호출 반복: LLM 호출 → 툴 실행 → 결과를 붙여 다시 호출. 처리는 ToolCaller가 한다."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool

from theseal.answer.config import AnswerConfig
from theseal.common.config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """LLM 호출 토큰 합. 호출할 때마다 add()로 더한다 (frozen 아님).

    AnswerResult에는 Usage로 옮겨 담는다.
    """

    input_tokens: int | None = None
    """한 번도 사용량이 오지 않았으면 None"""

    output_tokens: int | None = None

    def add(self, ai: AIMessage) -> None:
        """LLM 응답 하나의 토큰 수를 더한다. 응답에 사용량이 없으면 그대로 둔다."""
        meta = ai.usage_metadata
        if not meta:
            return
        self.input_tokens = (self.input_tokens or 0) + meta.get("input_tokens", 0)
        self.output_tokens = (self.output_tokens or 0) + meta.get("output_tokens", 0)


@dataclass(frozen=True)
class ToolCallRecord:
    """실행한 툴 하나의 기록. AnswerResult에는 ToolRecord로 옮겨 담는다."""

    name: str
    """LLM이 부른 툴 이름 (get_today, web_search …). 허용되지 않은 이름도 그대로 남는다"""

    args: Mapping[str, Any]
    """LLM이 넘긴 인자"""

    ok: bool
    latency_ms: int

    error: str | None = None
    """실패 이유. 성공이면 None"""


@dataclass
class ToolCallOutcome:
    """툴 호출 반복의 결과. 도중에 예외가 나도 그때까지의 툴 기록 · 토큰이 남도록 반복하면서 채운다 (frozen 아님)."""

    final: AIMessage | None = None
    """툴을 더 부르지 않은 마지막 LLM 응답. 횟수 초과면 None"""

    records: list[ToolCallRecord] = field(default_factory=list)
    """실행한 툴 기록 (실패도 ok=False로 남는다)"""

    usage: TokenUsage = field(default_factory=TokenUsage)
    """반복 안의 모든 LLM 호출 토큰 합"""

    exhausted: bool = False
    """마지막 기회에도 툴을 요청해 횟수를 넘겼는지"""


class ToolCaller:
    """LLM과 툴을 오가며 답을 만드는 반복. 툴 하나가 실패해도 답변은 계속 만든다.

    max_rounds는 툴 실행 라운드 수다. 툴 결과로 답을 쓸 LLM 호출 1회가 항상 더 허용된다
    (예: max_rounds=2 → get_today → add_days → 최종 답변까지 가능).

        outcome = await ToolCaller.from_config(llm, tools, config, answer_config).run(messages)
        outcome.final, outcome.records  →  AIMessage, [ToolCallRecord, …]

    Args:
      llm: AnswerGenerator가 만든 LLM. 여기서 직접 만들지 않는다.
      tools: LLM에 붙일 툴. 목록에 없는 이름을 부르면 실패로 기록한다.
      max_rounds: 툴 실행 라운드 수.
      tool_timeout: 툴 실행 한 번의 시간 제한(초).
      llm_timeout: LLM 호출 한 번의 시간 제한(초).
    """

    def __init__(self, llm: BaseChatModel, tools: Sequence[BaseTool], *, max_rounds: int,
                 tool_timeout: float, llm_timeout: float) -> None:
        self.llm = llm.bind_tools(tools)
        self.allowed = {t.name: t for t in tools}
        self.max_rounds = max_rounds
        self.tool_timeout = tool_timeout
        self.llm_timeout = llm_timeout

    @classmethod
    def from_config(cls, llm: BaseChatModel, tools: Sequence[BaseTool], config: LLMConfig,
                    answer_config: AnswerConfig) -> ToolCaller:
        """횟수 · 툴 시간 제한은 AnswerConfig(MAX_TOOL_ROUNDS · TOOL_TIMEOUT_SEC)에서,
        LLM 호출 시간 제한은 LLMConfig(LLM_TIMEOUT_SEC)에서 읽는다."""
        return cls(llm, tools, max_rounds=answer_config.max_tool_rounds,
                   tool_timeout=answer_config.tool_timeout_sec, llm_timeout=config.llm_timeout_sec)

    async def run(self, messages: Sequence[BaseMessage],
                  out: ToolCallOutcome | None = None) -> ToolCallOutcome:
        """툴을 더 부르지 않는 응답이 나오거나 횟수를 넘길 때까지 반복한 결과를 돌려준다.

        LLM 시간 초과(TimeoutError)와 LLM 오류는 호출한 쪽(AnswerGenerator)이 처리한다.

        Args:
          messages: 첫 LLM 호출에 보낼 메시지. 고치지 않고 복사해서 쓴다.
          out: 기록을 쌓을 객체. 넘기면 예외가 나도 호출한 쪽에 그때까지의 툴 기록 · 토큰이 남는다.
        """
        out = out or ToolCallOutcome()
        msgs = list(messages)
        for round_no in range(self.max_rounds + 1):
            ai = await asyncio.wait_for(self.llm.ainvoke(msgs), self.llm_timeout)
            out.usage.add(ai)
            msgs.append(ai)
            if not ai.tool_calls:
                out.final = ai
                return out
            if round_no == self.max_rounds:  # 마지막 기회에도 툴을 요청하면 횟수 초과
                break
            for call in ai.tool_calls:
                content, record = await self._run_tool(call)
                msgs.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
                out.records.append(record)
        out.exhausted = True
        return out

    async def _run_tool(self, call: ToolCall) -> tuple[str, ToolCallRecord]:
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
            logger.warning("툴 실행 시간 초과: %s (%s초)", name, self.tool_timeout)
            ok, error = False, f"도구 실행 시간 초과 ({self.tool_timeout}초)"
            content = f"오류: {error}"
        except Exception as e:  # 인자 검증 실패, 잘못된 날짜, 외부 API 오류 등. 기록하고 계속한다
            logger.exception("툴 실행 실패: %s", name)
            ok, error = False, f"{type(e).__name__}: {e}"
            content = f"오류: {error}"
        latency_ms = round((time.perf_counter() - start) * 1000)
        record = ToolCallRecord(name=name, args=MappingProxyType(dict(args)), ok=ok,
                                latency_ms=latency_ms, error=error)
        return content, record
