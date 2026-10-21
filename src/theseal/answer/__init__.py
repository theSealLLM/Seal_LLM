"""답변 생성 모듈: PromptSpec → AnswerResult.

answer_generator.py  AnswerGenerator: LLM 1회 호출 또는 툴 호출 경로 (generate · generate_stream은 이것을 부르는 함수)
tool_caller.py       ToolCaller: 툴 호출 반복 (LLM ↔ 툴)
tools/               날짜 · 날씨 · 웹 검색 툴과 라벨별 레지스트리 (TOOL_GROUPS · get_tools_for)
config.py            AnswerConfig: 툴 키 · 라벨별 툴 · 툴 반복 횟수 · 답변 시간 제한
"""
