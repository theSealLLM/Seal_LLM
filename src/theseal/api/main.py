"""서버 시작 · 앱 조립: FastAPI 앱을 만들고 lifespan · CORS · 라우터 · 오류 핸들러를 붙인다.

uv run uvicorn --app-dir src theseal.api.main:app --reload --port 8000

나눠 둔 곳
    main.py       서버 시작: lifespan에서 Pipeline을 한 번 만들어 app.state.pipeline에 둔다
    routes.py     외부 호출부: /health · /api/answer · /api/answer/stream
    pipeline.py   내부 처리: 분류 → 프롬프트 → 답변
    errors.py     내부 처리: 오류 응답 · HTTP 상태 · 입력 오류(400) 핸들러
    streaming.py  내부 처리: /api/answer/stream의 SSE 이벤트

FastAPI가 실행 중에 타입 힌트를 읽는 경계 모듈이라 `from __future__ import annotations`를 넣지 않는다.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from theseal.api import errors, routes
from theseal.api.config import ApiConfig
from theseal.api.pipeline import Pipeline, classifier_name

# __name__ 대신 uvicorn 로거를 쓴다. 루트 로거에는 핸들러가 없어 __name__ 로거의 info가 서버 로그에 나오지 않는다
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """서버 시작 때 Pipeline을 한 번 만든다 (분류 모델 · 설정 · 프롬프트 문서를 읽는다).

    분류 모델이 없어도 서버는 뜬다 (label을 직접 넣은 요청은 동작).
    """
    p = await run_in_threadpool(Pipeline.from_env)
    if p.classifier is not None:
        logger.info("분류 모델 로드: %s (%s)", classifier_name(p.classifier),
                    getattr(p.classifier.model, "device", "cpu"))
    else:
        logger.warning("분류 모델 없음: %s", p.classifier_error)
    app.state.pipeline = p
    yield


app = FastAPI(title="Seal_LLM", version="0.1.0", description="고객 문의 분류 · 답변 초안 API",
              lifespan=lifespan)
app.include_router(routes.router)
app.add_exception_handler(RequestValidationError, errors.invalid_input)

# 프론트(다른 포트 · 주소)가 브라우저에서 직접 부를 수 있게 허용한다. 미들웨어는 앱을 만들 때 붙여야 해서 여기서 .env를 읽는다
_cors_origins = ApiConfig.from_env().cors_origins()
if _cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])
