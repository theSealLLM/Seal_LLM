"""우리 서비스 답변(run_e2e.py 1번): API와 같은 체인(분류기 → 분류 후처리 → 프롬프트 → 답변)을 문의 하나에 돌린다.

README의 curl처럼 서버에 HTTP로 보내지 않고 같은 함수를 이 프로세스에서 바로 부른다. 문의 하나에 걸리는 시간은
LLM 생성(수 초)이 거의 전부라 HTTP를 거쳐도 몇 ms 차이지만, 서버를 따로 띄울 필요가 없고 툴 결과도 남길 수 있다.

pipeline.answer_chain = prepare_chain | 답변 생성(generate)과 같은 순서로 돌리되, 툴이 돌려준 결과도 남긴다.
AnswerResult.tools_used(ToolRecord)에는 툴 이름 · 인자 · 성공 여부만 있고 결과 내용이 없어서
도구 결과 불일치(E09)를 판정할 수 없다. 그래서 라벨에 켜진 툴을 같은 이름 · 설명 · 인자 형식의 툴로 감싸
결과를 함께 적는다. LLM이 보는 툴 정의와 호출 흐름은 서비스와 같다.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool

from theseal.answer.answer_generator import generate
from theseal.answer.config import AnswerConfig
from theseal.answer.tools import get_tools_for
from theseal.common.config import LLMConfig


def recording_tools(tools: list[BaseTool], outputs: list[dict]) -> list[BaseTool]:
    """툴을 감싸 호출할 때마다 outputs에 {name, args, output | error}를 쌓는다."""

    def wrap(original: BaseTool) -> BaseTool:
        async def call(**kwargs):
            try:
                output = await original.ainvoke(kwargs)
            except Exception as e:
                outputs.append({"name": original.name, "args": kwargs, "error": f"{type(e).__name__}: {e}"})
                raise
            outputs.append({"name": original.name, "args": kwargs, "output": output})
            return output

        return StructuredTool.from_function(coroutine=call, name=original.name,
                                            description=original.description,
                                            args_schema=original.args_schema)

    return [wrap(t) for t in tools]


async def call_service(case: dict, *, chain: Runnable, provider: str, config: LLMConfig,
                       answer_config: AnswerConfig, classifier: str,
                       llm: BaseChatModel | None = None) -> dict:
    """results.jsonl 한 줄 (채점 전). 채점하면 같은 줄에 module · score가 붙는다 (runner.score_service).
    chain: pipeline.prepare_chain(clf). classifier: 분류 모델 이름 (<모델>/<실행 ID>).
    llm: 테스트용 가짜 LLM (없으면 provider로 만든다)."""
    start = time.perf_counter()
    prepared = await chain.ainvoke({"text": case["text"], "label": None})
    spec, routing = prepared["prompt"], prepared["classification"]

    tool_outputs: list[dict] = []
    tools = recording_tools(get_tools_for(spec.label, answer_config), tool_outputs)
    result = await generate(spec, provider, config=config, answer_config=answer_config, llm=llm,
                            tools=tools)
    answer = result.answer
    return {
        "id": case["id"],
        "text": case["text"],
        "category": case.get("category"),
        "status": result.status,
        "answer": answer,
        "error": result.error.model_dump() if result.error else None,
        "trace": {
            "expected_label": case["expected_label"],
            "classifier_label": routing.classifier_label if routing else None,
            "final_label": spec.label,
            "classifier_scores": routing.scores if routing else None,
            "route": asdict(routing.route) if routing and routing.route else None,
            "verification": routing.verification if routing else None,
            "prompt_version": spec.prompt_version,
            # 범위 밖 경로의 안내 문구 (unknown 프롬프트의 답변 방식 5, 툴 횟수 초과)를 그대로 썼는지
            "fallback": answer is not None and answer.strip() == config.fallback_answer.strip(),
            "tools_used": [t.model_dump() for t in result.tools_used],
            "tool_outputs": tool_outputs,
        },
        "provider": result.provider,
        "model": result.model,
        "classifier": classifier,
        "usage": result.usage.model_dump(),
        "answer_latency_ms": result.latency_ms,
        "e2e_latency_ms": round((time.perf_counter() - start) * 1000),
    }
