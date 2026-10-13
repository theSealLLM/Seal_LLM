"""프롬프트 생성 모듈 (담당: 김지수).

출력 형식은 theseal.llm.schemas.PromptSpec.
"""

from .label_unknown import build_unknown_prompt
from .prompt_generator import build_common_policy_prompt, build_label_specific_prompt

__all__ = [
    "build_label_specific_prompt",
    "build_common_policy_prompt",
    "build_unknown_prompt",
]