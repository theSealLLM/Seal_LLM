"""분류 후처리: ① 분류기 결과 → ② 경로 규칙 → ③ LLM 검증 → 최종 라벨.

route_rule.py    ② 확률(보정)로 confident · low_confidence · ambiguous · unknown 판정 (LLM 없음)
llm_verifier.py  ③ 라벨이 맞는지 LLM에 예/아니오로 확인
router.py        ②·③을 설정대로 이어서 실행 (Classifier.predict() 결과를 그대로 받는다)
config.py        .env 설정 · 모델별 보정값(route_rule.json) 위치
"""
