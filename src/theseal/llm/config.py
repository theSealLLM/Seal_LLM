"""LLM 설정. 모델 이름 · 키 · 툴 on/off는 .env에서 읽는다 (코드에 직접 쓰지 않는다).

우선순위: 이미 설정된 환경변수 > 프로젝트 루트 .env > 아래 기본값. 값이 빈 항목은 기본값을 쓴다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"

PROVIDERS = ("local", "openai")

# 팀이 확인한 로컬 모델 ID (`ollama list`의 ID 열 = digest 앞 12자리).
# 다르면 Hugging Face 저장소 파일이 갱신된 것이므로 팀에 알린다 (scripts/llm/fetch_models.py가 확인)
OLLAMA_MODEL_IDS = {
    "hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M": "487eacf5e94b",
}
LABELS = ("shipping", "refund", "account", "unknown")


@dataclass
class LLMConfig:
    # 로컬 LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"
    # thinking: None이면 모델 기본값 (Gemma 4 E4B는 켜짐, true를 명시하면 Ollama가 거부한다)
    ollama_reasoning: bool | None = None  # 툴 없는 경로 (shipping · refund · account)
    ollama_reasoning_with_tools: bool | None = None  # 툴 경로 (unknown). 끄면 툴을 부르지 않는다
    ollama_keep_alive: str | None = None  # VRAM 유지 시간. "-1"이면 계속 유지

    # 상용 LLM (OpenAI)
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-luna"
    openai_reasoning_effort: str | None = None  # None이면 API 기본값(medium)

    default_provider: str = "local"

    # 툴
    tavily_api_key: str = ""
    kma_auth_key: str = ""  # 기상청 API허브 인증키. 비우면 날씨 툴을 등록하지 않는다
    kma_api_base: str = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0"
    tool_date_tools: bool = True
    tool_weather: bool = True
    tool_web_search: bool = True
    tools_by_label: dict[str, list[str]] = field(
        default_factory=lambda: {"unknown": ["date_tools", "weather", "web_search"]}
    )
    max_tool_rounds: int = 2  # 툴 실행 라운드 수. 툴 결과로 답을 쓸 LLM 호출 1회는 항상 더 허용

    # 툴 호출 횟수를 넘겼을 때 답변으로 쓰는 안내 문구
    fallback_answer: str = (
        "죄송합니다. 이 질문은 지금 확인해 드리기 어렵습니다. "
        "이 창구에서는 배송, 환불·반품·주문 취소, 계정 관련 문의를 도와드릴 수 있습니다."
    )

    # 제한
    llm_timeout_sec: float = 60
    tool_timeout_sec: float = 10

    timezone: str = "Asia/Seoul"

    def __post_init__(self) -> None:
        if self.default_provider not in PROVIDERS:
            raise ValueError(f"DEFAULT_PROVIDER는 {PROVIDERS} 중 하나여야 합니다: {self.default_provider!r}")

    @classmethod
    def from_env(cls, env_file: Path | None = ENV_FILE) -> LLMConfig:
        return config_from_env(cls, env_file)


def config_from_env(cls, env_file: Path | None = ENV_FILE):
    """설정 dataclass를 환경변수로 채운다. 필드 이름을 대문자로 바꾼 이름을 읽는다
    (예: ollama_model → OLLAMA_MODEL). theseal.routing.config도 같은 방식을 쓴다."""
    if env_file is not None:
        load_dotenv(env_file, override=False)
    values = {}
    for f in fields(cls):
        raw = os.environ.get(f.name.upper(), "").strip()
        if raw:
            values[f.name] = _parse(raw, f.type)
    return cls(**values)


def _parse(raw: str, type_hint: str):
    """dataclass 필드 타입(문자열, 예: "float | None")에 맞춰 환경변수 값을 바꾼다."""
    base = type_hint.split("|")[0].strip()
    if base == "bool":
        return raw.lower() in ("1", "true", "yes", "on")
    if base == "int":
        return int(raw)
    if base == "float":
        return float(raw)
    if base.startswith("dict"):
        return json.loads(raw)
    return raw


@lru_cache
def get_config() -> LLMConfig:
    return LLMConfig.from_env()
