"""scripts/llm 공통: src 경로 등록, Ollama 서버 · 모델 확인."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

# pyproject가 package = false라서 theseal 패키지를 설치하지 않는다. src를 직접 경로에 넣는다.
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 콘솔 코드페이지가 한글을 못 받는 환경(cp1252 등)에서 print가 죽지 않도록
sys.stdout.reconfigure(errors="replace")

from theseal.llm.config import OLLAMA_MODEL_IDS, LLMConfig

OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"


def ollama_version(config: LLMConfig) -> str | None:
    """Ollama 서버 버전. 서버가 꺼져 있으면 None."""
    try:
        resp = httpx.get(f"{config.ollama_base_url}/api/version", timeout=5)
        resp.raise_for_status()
        return resp.json().get("version")
    except httpx.HTTPError:
        return None


def require_ollama(config: LLMConfig) -> str:
    version = ollama_version(config)
    if version is None:
        raise SystemExit(
            f"[중단] Ollama 서버에 연결할 수 없습니다: {config.ollama_base_url}\n"
            f"  - 설치: {OLLAMA_DOWNLOAD_URL}\n"
            "  - 설치 후 Ollama 앱을 실행하거나 터미널에서 `ollama serve`\n"
            "  - 다른 주소를 쓰면 .env의 OLLAMA_BASE_URL을 고친다")
    return version


def find_model(config: LLMConfig, model: str) -> dict | None:
    """받아 둔 모델 정보 (/api/tags). 없으면 None."""
    resp = httpx.get(f"{config.ollama_base_url}/api/tags", timeout=10)
    resp.raise_for_status()
    for m in resp.json().get("models", []):
        if m.get("name") == model or m.get("model") == model:
            return m
    return None


def short_id(info: dict) -> str:
    """`ollama list`의 ID 열과 같은 digest 앞 12자리."""
    return (info.get("digest") or "").removeprefix("sha256:")[:12]


def check_model_id(model: str, info: dict) -> tuple[bool | None, str]:
    """(일치 여부, 메시지). 기준 ID가 없는 모델이면 일치 여부는 None."""
    expected = OLLAMA_MODEL_IDS.get(model)
    actual = short_id(info)
    if expected is None:
        return None, f"ID {actual} (기준 ID가 등록되지 않은 모델이라 비교하지 않음)"
    if actual == expected:
        return True, f"ID {actual} (팀 기준과 같음)"
    return False, (f"ID {actual} ≠ 팀 기준 {expected}. Hugging Face 저장소 파일이 갱신된 것이므로 "
                   "팀에 알린다 (결과 비교가 섞일 수 있음)")
