"""공용 모듈: api/ · prompt/ · answer/ · scripts/answer/가 함께 쓴다.

config.py   설정 (LLMConfig 기본값 + .env를 읽는 config_from_env)
client.py   LLM 연결 (local · openai 분기는 이 파일 한 곳에만)
schemas.py  모듈 간 데이터 형식 (PromptSpec · AnswerResult 등)
"""
