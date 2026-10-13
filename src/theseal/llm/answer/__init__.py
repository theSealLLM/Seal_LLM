"""답변 생성 모듈: PromptSpec → AnswerResult.

answer_generator.py  LLM 1회 호출 또는 툴 호출 경로
tool_caller.py       툴 호출 반복 (LLM ↔ 툴)
tools/               날짜 계산 · 웹 검색 툴과 라벨별 레지스트리
"""
