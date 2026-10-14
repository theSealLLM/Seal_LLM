"""FastAPI 앱: 분류 · 답변 생성 기능을 HTTP로 연결한다.

uv run uvicorn --app-dir src theseal.api.main:app --reload --port 8000

| 메서드 | 경로            | 응답                                       |
| GET    | /health         | 상태 · 분류 모델 · 설정                    |
| POST   | /api/classify   | RoutingResult                              |
| POST   | /api/answer     | AnswerResult + classification              |
| POST   | /api/answer/stream | SSE: status · classification · delta · done |

HTTP 상태: 입력 오류 400 · 설정 오류 500 · LLM 오류 502 · 시간 초과 504
/api/answer는 오류여도 본문이 AnswerResponse 형식이다 (status=error, error.code).

/api/answer/stream (text/event-stream, 요청 본문은 /api/answer와 같다)
    event: status          {"stage": "classifying" | "answering" | "adventure", "message": "..."}
    event: classification  RoutingResult (분류를 거쳤을 때만)
    event: delta           {"text": "..."} 답변 글자 조각. unknown(adventure)은 보내지 않는다
    event: done            AnswerResponse (마지막 한 번, 오류도 여기로: status=error)
입력 오류만 스트림을 열기 전에 400 JSON(AnswerResponse)으로 돌려준다.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from theseal.common.client import model_name
from theseal.common.config import PROVIDERS, get_config
from theseal.prompt.routing_config import get_routing_config
from theseal.prompt.router import RoutingResult

from . import pipeline
from .schemas import AnswerRequest, AnswerResponse, ClassifyRequest, ErrorBody, ErrorResponse

STATUS_BY_ERROR = {"invalid_input": 400, "config_error": 500, "llm_error": 502, "llm_timeout": 504}
STAGE_MESSAGES = {
    "classifying": "문의를 분류하는 중...",
    "answering": "답변을 작성하는 중...",
    "adventure": "챗봇이 알 수 없는 것을 찾아 모험을 떠나는 중...",
}
logger = logging.getLogger("uvicorn.error")
state: dict = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 분류 모델은 서버 시작 때 한 번 불러온다. 없어도 서버는 뜬다 (label을 직접 넣은 /api/answer는 동작)
    api_config = pipeline.ApiConfig.from_env()
    try:
        state["predictor"] = await run_in_threadpool(pipeline.load_classifier, api_config)
        p = state["predictor"]
        logger.info("분류 모델 로드: %s (%s)", pipeline.classifier_name(p), getattr(p.model, "device", "cpu"))
    except Exception as e:  # noqa: BLE001 — 모델이 없어도 서버는 띄운다
        state["error"] = f"{type(e).__name__}: {e}"
        logger.warning("분류 모델 없음: %s", state["error"])
    yield


app = FastAPI(title="Seal_LLM", version="0.1.0", description="고객 문의 분류 · 답변 초안 API",
              lifespan=lifespan)

# 프론트(다른 포트 · 주소)가 브라우저에서 직접 부를 수 있게 허용한다. 미들웨어는 앱을 만들 때 붙여야 해서 여기서 .env를 읽는다
_cors_origins = pipeline.ApiConfig.from_env().cors_origins()
if _cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])


def _now() -> str:
    return datetime.now(ZoneInfo(get_config().timezone)).isoformat(timespec="seconds")


def _error_response(code: str, message: str, provider: str | None, label: str = "unknown",
                    classification: RoutingResult | None = None) -> AnswerResponse:
    config = get_config()
    provider = provider if provider in PROVIDERS else config.default_provider
    return AnswerResponse(
        request_id=str(uuid.uuid4()), status="error", label=label, answer=None, provider=provider,
        model=model_name(provider, config), prompt_version="-", latency_ms=0,
        error={"code": code, "message": message}, created_at=_now(), classification=classification,
    )


def _error_answer(code: str, message: str, provider: str | None, label: str = "unknown",
                  classification: RoutingResult | None = None) -> JSONResponse:
    result = _error_response(code, message, provider, label, classification)
    return JSONResponse(result.model_dump(), status_code=STATUS_BY_ERROR[code])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error(code: str, message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(error=ErrorBody(code=code, message=message)).model_dump(),
                        status_code=STATUS_BY_ERROR[code])


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    """빈 입력 · 길이 초과 · 잘못된 라벨 · 잘못된 JSON → 400"""
    message = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors())
    if request.url.path.startswith("/api/answer"):
        body = exc.body if isinstance(exc.body, dict) else {}
        return _error_answer("invalid_input", message, body.get("provider"))
    return _error("invalid_input", message)


@app.get("/health")
async def health() -> dict:
    p = state["predictor"]
    return {
        "status": "ok",
        "classifier": pipeline.classifier_name(p) if p else None,
        "classifier_device": str(getattr(p.model, "device", "cpu")) if p else None,
        "classifier_error": state["error"],
        "route_rule_enabled": get_routing_config().route_rule_enabled,
        "verify_mode": get_routing_config().verify_mode,
        "default_provider": get_config().default_provider,
    }


@app.post("/api/classify", response_model=RoutingResult,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def classify(req: ClassifyRequest):
    """분류기 → 경로 규칙 → LLM 검증. 최종 라벨과 판정 근거를 돌려준다."""
    if state["predictor"] is None:
        return _error("config_error", f"분류 모델을 불러오지 못했습니다: {state['error']}")
    return await pipeline.classify(state["predictor"], req.text)


@app.post("/api/answer", response_model=AnswerResponse,
          responses={400: {"model": AnswerResponse}, 500: {"model": AnswerResponse},
                     502: {"model": AnswerResponse}, 504: {"model": AnswerResponse}})
async def answer(req: AnswerRequest) -> JSONResponse:
    """질문 → 분류 → 프롬프트 → 답변. label을 넣으면 분류를 건너뛴다."""
    classification = None
    label = req.label
    if label is None:
        if state["predictor"] is None:
            return _error_answer("config_error", f"분류 모델을 불러오지 못했습니다: {state['error']}",
                                 req.provider)
        classification = await pipeline.classify(state["predictor"], req.text)
        label = classification.label

    result = await pipeline.answer(label, req.text, req.provider, get_config())
    response = AnswerResponse(**result.model_dump(), classification=classification)
    status = STATUS_BY_ERROR[result.error.code] if result.error else 200
    return JSONResponse(response.model_dump(), status_code=status)


@app.post("/api/answer/stream", response_class=StreamingResponse,
          responses={200: {"content": {"text/event-stream": {}}}, 400: {"model": AnswerResponse}})
async def answer_stream(req: AnswerRequest) -> StreamingResponse:
    """/api/answer와 같은 처리를 SSE로 흘려보낸다. 오류도 done 이벤트(status=error)로 보낸다."""

    async def events() -> AsyncIterator[str]:
        ctx: dict = {}  # 오류가 나도 그때까지 정해진 라벨 · 분류 결과를 done에 담는다
        try:
            async for chunk in _answer_events(req, ctx):
                yield chunk
        except Exception as e:  # 스트림이 열린 뒤에는 상태 코드를 바꿀 수 없어 done으로 알린다
            logger.exception("스트리밍 중 오류")
            err = _error_response("llm_error", f"{type(e).__name__}: {e}", req.provider,
                                  ctx.get("label", "unknown"), ctx.get("classification"))
            yield _sse("done", err.model_dump())

    # 프록시 · 브라우저가 모아서 보내지 않게 한다
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _answer_events(req: AnswerRequest, ctx: dict) -> AsyncIterator[str]:
    """status · classification · delta · done 이벤트 (SSE 문자열). ctx에 정해진 label · classification을 남긴다"""
    def status(stage: str) -> str:
        return _sse("status", {"stage": stage, "message": STAGE_MESSAGES[stage]})

    classification = None
    label = req.label
    if label is None:
        if state["predictor"] is None:
            err = _error_response("config_error", f"분류 모델을 불러오지 못했습니다: {state['error']}",
                                  req.provider)
            yield _sse("done", err.model_dump())
            return
        yield status("classifying")
        classification = await pipeline.classify(state["predictor"], req.text)
        label = classification.label
        ctx["classification"] = classification
        yield _sse("classification", classification.model_dump())

    ctx["label"] = label
    yield status("adventure" if label == "unknown" else "answering")
    async for event in pipeline.answer_stream(label, req.text, req.provider, get_config()):
        if event["type"] == "delta":
            yield _sse("delta", {"text": event["text"]})
        else:
            response = AnswerResponse(**event["result"].model_dump(), classification=classification)
            yield _sse("done", response.model_dump())
