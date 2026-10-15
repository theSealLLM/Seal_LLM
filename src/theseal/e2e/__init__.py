"""E2E 답변 품질 평가: 외부 API의 참조 답변과 우리 서비스(분류 → 분류 후처리 → 프롬프트 → 답변)의 답변을 비교한다.

    ① reference.py  참조 답변 생성 + 핵심 주장 · 필수 항목 추출 (데이터마다 한 번, 캐시)
    ② service.py    우리 서비스 답변 + 단계별 기록(trace)
    ③ judge.py      항목별 판정 → error_codes.py가 E01~E10을 붙인다
    ④ metrics.py    지표 3개 + 치명적 오류율 + 오류 코드 · 모듈 · 라벨별 집계

평가 기준(정책 원문 · 정책 항목 · unknown 필수 항목)은 policy.py 한 곳에 둔다. 실행은 scripts/e2e/run_e2e.py.
"""
