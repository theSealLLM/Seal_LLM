"""API(api/) 설정: 분류 모델 task · 장치, CORS 허용 주소.

.env에서 필드 이름을 대문자로 적어 바꾼다 (예: API_CORS_ORIGINS=off).
LLM 연결과 여러 모듈이 같이 쓰는 값은 common/config.py의 LLMConfig에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from theseal.classifier.config import DEFAULT_TASK
from theseal.common.config import ENV_FILE, config_from_env

# 기본 CORS 허용 주소: 로컬 개발 서버 (Vite 5173 · CRA/Next 3000)
DEV_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"


@dataclass(frozen=True)
class ApiConfig:
    """API 서버 설정. Pipeline.from_env(분류 모델)와 main.py(CORS)가 쓴다."""

    # 분류 모델은 classifier/config.py의 FINAL_MODEL[api_classifier_task]를 쓴다.
    # 모델 폴더(runs/classifier/<task>/<모델>/<실행 ID>/model)가 없으면 같은 설정으로 학습해 그 자리에 둔다 (README)
    api_classifier_task: str = DEFAULT_TASK
    api_classifier_device: str | None = None  # None이면 자동 (GPU가 있으면 cuda)
    # 브라우저에서 직접 호출을 허용할 프론트 주소 (쉼표로 구분). "*"면 전부, "off"면 CORS를 켜지 않는다
    api_cors_origins: str = DEV_CORS_ORIGINS

    @classmethod
    def from_env(cls, env_file: Path | None = ENV_FILE) -> ApiConfig:
        """환경변수 > .env > 코드 기본값 순서로 채운다."""
        return config_from_env(cls, env_file)

    def cors_origins(self) -> list[str]:
        """CORS 허용 주소 목록. "off"면 빈 목록, 주소 끝의 / 는 뗀다."""
        if self.api_cors_origins.strip().lower() == "off":
            return []
        return [o.strip().rstrip("/") for o in self.api_cors_origins.split(",") if o.strip()]
