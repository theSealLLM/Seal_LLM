"""공용 설정: 여러 모듈(answer · prompt · api · e2e)이 함께 쓰는 값과 .env를 읽는 config_from_env.

한 모듈에서만 쓰는 설정은 그 모듈의 config.py에 둔다.
    answer/config.py   AnswerConfig (툴 키 · 라벨별 툴 · 툴 반복 횟수 · 답변 시간 제한)
    prompt/config.py   RouterConfig (경로 규칙 · LLM 검증)

.env에는 비밀 키와 PC마다 다른 값만 둔다 (.env.example 참고)
    비밀 키    OPENAI_API_KEY · KMA_AUTH_KEY · TAVILY_API_KEY
    PC 설정    OLLAMA_BASE_URL · OLLAMA_KEEP_ALIVE
그 밖의 필드도 이름을 대문자로 적으면 바꿀 수 있다 (예: llm_timeout_sec → LLM_TIMEOUT_SEC=30).
우선순위: 이미 설정된 환경변수 > 프로젝트 루트 .env > 코드 기본값. 값이 빈 항목은 기본값을 쓴다.
같은 값을 .env에 그대로 두면 코드 기본값을 바꿔도 .env 값이 이기므로, 바꿀 항목만 적는다.
설정 객체는 쓰는 쪽(Pipeline 등)이 from_env()로 한 번 만들어 들고 있다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"

PROVIDERS = ("local", "openai")

# 서비스 기준 시간대. 서버 OS 시간대(UTC 등)와 상관없이 "오늘" · 응답 시각(created_at)을 한국 시간으로 정한다
TIMEZONE = ZoneInfo("Asia/Seoul")

# 팀이 확인한 로컬 모델 ID (`ollama list`의 ID 열 = digest 앞 12자리).
# 다르면 Hugging Face 저장소 파일이 갱신된 것이므로 팀에 알린다 (scripts/answer/fetch_models.py가 확인)
OLLAMA_MODEL_IDS = {
    "hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M": "487eacf5e94b",
}
LABELS = ("shipping", "refund", "account", "unknown")


@dataclass(frozen=True)
class LLMConfig:
    """LLM 연결 설정과 여러 모듈이 같이 쓰는 값. 답변 생성 · LLM 검증 · API · E2E가 함께 쓴다."""

    # 로컬 LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"  # .env (PC 설정): Ollama를 다른 주소 · 포트에서 띄울 때
    ollama_model: str = "hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"
    # thinking: None이면 모델 기본값 (Gemma 4 E4B는 켜짐, true를 명시하면 Ollama가 거부한다). .env에서는 default로 None
    # 툴 없는 경로 (shipping · refund · account)와 분류 후처리의 LLM 검증. 켜 두면 검증이 VERIFY_TIMEOUT_SEC를 넘기기 쉬워 끈다
    ollama_reasoning: bool | None = False
    ollama_reasoning_with_tools: bool | None = None  # 툴 경로 (unknown). 끄면 툴을 부르지 않는다
    # .env (PC 설정): VRAM 유지 시간. "-1"이면 계속 유지, None이면 Ollama 기본(5분)
    ollama_keep_alive: str | None = None

    # 상용 LLM (OpenAI)
    openai_api_key: str = field(default="", repr=False)  # .env (비밀 키). 비우면 openai provider는 config_error
    openai_model: str = "gpt-5.6-luna"
    openai_reasoning_effort: str | None = None  # None이면 API 기본값(medium)

    default_provider: str = "local"

    # 범위 밖 안내 문구. 툴 호출 횟수를 넘겼을 때의 답변(answer)과 unknown 프롬프트(prompt)가 같은 문장을 쓴다
    fallback_answer: str = (
        "죄송합니다. 이 질문은 지금 확인해 드리기 어렵습니다. "
        "이 창구에서는 배송, 환불·반품·주문 취소, 계정 관련 문의를 도와드릴 수 있습니다."
    )

    llm_timeout_sec: float = 60  # LLM 호출 한 번 (답변 생성 · OpenAI 클라이언트)

    def __post_init__(self) -> None:
        if self.default_provider not in PROVIDERS:
            raise ValueError(f"DEFAULT_PROVIDER는 {PROVIDERS} 중 하나여야 합니다: {self.default_provider!r}")

    @classmethod
    def from_env(cls, env_file: Path | None = ENV_FILE) -> LLMConfig:
        """환경변수 > .env > 코드 기본값 순서로 채운다."""
        return config_from_env(cls, env_file)


def config_from_env[ConfigT](cls: type[ConfigT], env_file: Path | None = ENV_FILE) -> ConfigT:
    """설정 dataclass를 환경변수로 채운다. 필드 이름을 대문자로 바꾼 이름을 읽는다 (예: ollama_model → OLLAMA_MODEL).

    LLMConfig · AnswerConfig · RouterConfig · ApiConfig가 모두 이 함수를 쓴다.
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)
    values = {}
    for f in fields(cls):
        raw = os.environ.get(f.name.upper(), "").strip()
        if raw:
            values[f.name] = _parse(raw, f.type)
    return cls(**values)


def _parse(raw: str, type_hint: str) -> Any:
    """dataclass 필드 타입(문자열, 예: "float | None")에 맞춰 환경변수 값을 바꾼다.
    None을 받을 수 있는 필드는 default라고 적으면 None (빈 값은 코드 기본값이라 None과 다를 수 있다)."""
    base = type_hint.split("|")[0].strip()
    if "None" in type_hint and raw.lower() == "default":
        return None
    if base == "bool":
        return raw.lower() in ("1", "true", "yes", "on")
    if base == "int":
        return int(raw)
    if base == "float":
        return float(raw)
    if base.startswith(("dict", "Mapping")):
        return json.loads(raw)
    return raw
