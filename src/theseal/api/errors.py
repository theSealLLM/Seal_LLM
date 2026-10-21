"""오류 응답: 오류 코드 → HTTP 상태, 답변 없는 AnswerResponse, 입력 오류 핸들러. 오류 응답은 이 파일 한 곳에서 만든다.

/api/answer는 오류여도 본문이 AnswerResponse 형식이다 (status=error, error.code).
HTTP 상태: 입력 오류 400 · 설정 오류 500 · LLM 오류 502 · 시간 초과 504

pydantic · FastAPI가 실행 중에 타입 힌트를 읽는 경계 모듈이라 `from __future__ import annotations`를 넣지 않는다.
"""

import uuid
from datetime import datetime

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from theseal.api.schemas import AnswerResponse
from theseal.common.client import model_name
from theseal.common.config import PROVIDERS, TIMEZONE, LLMConfig
from theseal.prompt.router import RoutingResult

STATUS_BY_ERROR = {
    "invalid_input": status.HTTP_400_BAD_REQUEST,
    "config_error": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "llm_error": status.HTTP_502_BAD_GATEWAY,
    "llm_timeout": status.HTTP_504_GATEWAY_TIMEOUT,
}


def now_iso() -> str:
    """서비스 시간대(TIMEZONE) 기준 지금 시각 (ISO 8601, 초 단위)."""
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def error_response(config: LLMConfig, code: str, message: str, provider: str | None,
                   label: str = "unknown",
                   classification: RoutingResult | None = None) -> AnswerResponse:
    """답변 없이 오류만 담은 AnswerResponse. provider가 잘못됐으면 DEFAULT_PROVIDER로 적는다."""
    provider = provider if provider in PROVIDERS else config.default_provider
    return AnswerResponse(
        request_id=str(uuid.uuid4()), status="error", label=label, answer=None, provider=provider,
        model=model_name(provider, config), prompt_version="-", latency_ms=0,
        error={"code": code, "message": message}, created_at=now_iso(),
        classification=classification,
    )


def classifier_missing(config: LLMConfig, classifier_error: str | None,
                       provider: str | None) -> AnswerResponse:
    """분류 모델 없이 분류가 필요한 요청을 받았을 때의 config_error 응답."""
    message = f"분류 모델을 불러오지 못했습니다: {classifier_error}"
    return error_response(config, "config_error", message, provider)


def to_json_response(response: AnswerResponse) -> JSONResponse:
    """AnswerResponse를 오류 코드에 맞는 HTTP 상태로 돌려준다 (오류가 없으면 200)."""
    status_code = STATUS_BY_ERROR[response.error.code] if response.error else status.HTTP_200_OK
    return JSONResponse(response.model_dump(), status_code=status_code)


async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    """빈 입력 · 길이 초과 · 잘못된 라벨 · 잘못된 JSON → 400. main.py가 RequestValidationError 핸들러로 등록한다."""
    message = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors())
    body = exc.body if isinstance(exc.body, dict) else {}
    config = request.app.state.pipeline.llm_config
    return to_json_response(error_response(config, "invalid_input", message, body.get("provider")))
