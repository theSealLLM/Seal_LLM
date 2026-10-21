"""FastAPI: 분류 · 답변 생성 기능을 HTTP로 연결한다.

서버 시작
  main.py       앱 조립: lifespan(Pipeline 생성) · CORS · 라우터 · 오류 핸들러 등록
  config.py     ApiConfig: 분류 모델 task · 장치, CORS 허용 주소
외부 호출부
  routes.py     엔드포인트 (GET /health, POST /api/answer, POST /api/answer/stream). 요청 → Pipeline → 응답만
  schemas.py    요청 형식과 /api/answer 응답 형식
  deps.py       라우터가 Depends로 받는 의존성 (app.state.pipeline)
내부 처리
  pipeline.py   기능 연결 (분류기 → 분류 후처리 → 프롬프트 → 답변 생성)
  errors.py     오류 응답 · 오류 코드 → HTTP 상태 · 입력 오류(400) 핸들러
  streaming.py  /api/answer/stream의 SSE 이벤트

실행: uv run uvicorn --app-dir src theseal.api.main:app --reload --port 8000 → http://127.0.0.1:8000/docs
분류 모델은 classifier/config.py의 FINAL_MODEL을 쓴다
"""
