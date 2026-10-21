"""/api/answer/stream의 SSE 이벤트를 만든다. 분류 → 프롬프트는 /api/answer와 같은 체인, 답변만 글자 조각으로 흘린다.

    event: status          {"stage": "classifying" | "answering" | "adventure", "message": "..."}
    event: classification  RoutingResult (분류를 거쳤을 때만)
    event: delta           {"text": "..."} 답변 글자 조각. unknown(adventure)은 보내지 않는다
    event: done            AnswerResponse (마지막 한 번, 오류도 여기로: status=error)

pydantic이 실행 중에 타입 힌트를 읽는 경계 모듈이라 `from __future__ import annotations`를 넣지 않는다.
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from theseal.api import errors
from theseal.api.pipeline import Pipeline
from theseal.api.schemas import AnswerRequest, AnswerResponse

STAGE_MESSAGES = {
    "classifying": "문의를 분류하는 중...",
    "answering": "답변을 작성하는 중...",
    "adventure": "챗봇이 알 수 없는 것을 찾아 모험을 떠나는 중...",
}
# 프록시 · 브라우저가 모아서 보내지 않게 한다
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# __name__ 대신 uvicorn 로거를 쓴다. 루트 로거에는 핸들러가 없어 __name__ 로거의 info가 서버 로그에 나오지 않는다
logger = logging.getLogger("uvicorn.error")


async def answer_events(p: Pipeline, req: AnswerRequest) -> AsyncIterator[str]:
    """요청 하나의 SSE 이벤트 문자열. 스트림이 열린 뒤의 오류는 done 이벤트(status=error)로 보낸다."""
    try:
        async for chunk in _events(p, req):
            yield chunk
    except Exception as e:  # 스트림이 열린 뒤에는 상태 코드를 바꿀 수 없어 done으로 알린다
        logger.exception("스트리밍 중 오류")
        yield sse("done", errors.error_response("llm_error", f"{type(e).__name__}: {e}").model_dump())


def sse(event: str, data: dict[str, Any]) -> str:
    """SSE 이벤트 하나 (event · data 줄과 빈 줄)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _events(p: Pipeline, req: AnswerRequest) -> AsyncIterator[str]:
    if req.label is None:
        if p.classifier is None:
            yield sse("done", errors.classifier_missing(p.classifier_error).model_dump())
            return
        yield _status_event("classifying")

    prepared = await p.prepare_chain().ainvoke({"text": req.text, "label": req.label})
    classification, label = prepared["classification"], prepared["label"]
    if classification is not None:
        yield sse("classification", asdict(classification))

    yield _status_event("adventure" if label == "unknown" else "answering")
    async for event in p.answer_stream(prepared["prompt"], req.provider):
        if event["type"] == "delta":
            yield sse("delta", {"text": event["text"]})
        else:
            yield sse("done", AnswerResponse.from_result(event["result"]).model_dump())


def _status_event(stage: str) -> str:
    return sse("status", {"stage": stage, "message": STAGE_MESSAGES[stage]})
