"""외부 호출부: HTTP 엔드포인트. 요청 → Pipeline → 응답만 하고 처리 로직은 두지 않는다.

| 메서드 | 경로               | 응답                                        |
| GET    | /health            | 상태 · 분류 모델 · 설정                     |
| POST   | /api/answer        | 답변 · 상태 · 오류 · 토큰 수 (AnswerResponse) |
| POST   | /api/answer/stream | SSE: status · classification · delta · done |

처리는 pipeline.py(분류 → 프롬프트 → 답변), 오류 응답은 errors.py, SSE 이벤트는 streaming.py에 있다.
FastAPI가 실행 중에 타입 힌트를 읽는 경계 모듈이라 `from __future__ import annotations`를 넣지 않는다.
"""

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from theseal.api.deps import PipelineDep
from theseal.api.errors import to_json_response
from theseal.api.schemas import AnswerRequest, AnswerResponse
from theseal.api.streaming import SSE_HEADERS, answer_events

router = APIRouter()


@router.get("/health")
async def health(p: PipelineDep) -> dict[str, Any]:
    """서버 상태 · 분류 모델 · 분류 후처리 설정 · 기본 provider."""
    return p.status()


@router.post("/api/answer", response_model=AnswerResponse,
             responses={400: {"model": AnswerResponse}, 500: {"model": AnswerResponse},
                        502: {"model": AnswerResponse}, 504: {"model": AnswerResponse}})
async def answer(req: AnswerRequest, p: PipelineDep) -> JSONResponse:
    """질문 → 분류 → 프롬프트 → 답변 (LCEL 체인 Pipeline.answer_chain). label을 넣으면 분류를 건너뛴다."""
    return to_json_response(await p.answer(req.text, req.label, req.provider))


@router.post("/api/answer/stream", response_class=StreamingResponse,
             responses={200: {"content": {"text/event-stream": {}}},
                        400: {"model": AnswerResponse}})
async def answer_stream(req: AnswerRequest, p: PipelineDep) -> StreamingResponse:
    """/api/answer와 같은 처리를 SSE로 흘려보낸다. 오류도 done 이벤트(status=error)로 보낸다."""
    return StreamingResponse(answer_events(p, req), media_type="text/event-stream",
                             headers=SSE_HEADERS)
