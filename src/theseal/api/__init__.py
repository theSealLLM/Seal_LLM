"""FastAPI: 분류 · 답변 생성 기능을 HTTP로 연결한다.

main.py     앱과 엔드포인트 (GET /health, POST /api/classify, POST /api/answer, POST /api/answer/stream)
schemas.py  요청 형식과 /api/answer 응답 형식
pipeline.py  실제 기능 연결 (분류기 → 분류 후처리 → 프롬프트 → 답변 생성)

실행: uv run uvicorn --app-dir src theseal.api.main:app --reload --port 8000 → http://127.0.0.1:8000/docs
분류 모델은 .env의 API_CLASSIFIER로 고른다 (기본 final = classifier/config.py FINAL_MODEL)
"""
