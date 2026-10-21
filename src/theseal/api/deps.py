"""라우터가 Depends로 받는 의존성. 서비스 객체는 lifespan이 만들어 app.state에 둔다 (전역 싱글턴 없음).

FastAPI가 실행 중에 타입 힌트를 읽는 경계 모듈이라 `from __future__ import annotations`를 넣지 않는다.
테스트는 app.state.pipeline만 바꿔 끼우면 된다.
"""

from typing import Annotated

from fastapi import Depends, Request

from theseal.api.pipeline import Pipeline


def get_pipeline(request: Request) -> Pipeline:
    """lifespan이 만든 Pipeline (app.state.pipeline)을 돌려준다."""
    return request.app.state.pipeline


PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]
