"""프롬프트 생성 모듈 (담당: 윤필수, 김지수).

서비스용 프롬프트는 prompt_generator.py 한 곳에서 관리한다. 출력 형식은 theseal.common.schemas.PromptSpec.
"""

# A/B 프롬프트 비교 실험 재현을 위해 라벨별 독립 프롬프트 유지
from .prompt_ab_experiment import build_label_specific_prompt
# 서비스에서 사용하는 Prompt 객체와 최종 프롬프트 생성 함수 공개
from .prompt_generator import (
    Prompt,
    build_common_policy_prompt,
    build_label_unknown_prompt,
    build_prompt,
)


__all__ = [
    "Prompt",
    "build_common_policy_prompt",
    "build_label_specific_prompt",
    "build_label_unknown_prompt",
    "build_prompt",
]