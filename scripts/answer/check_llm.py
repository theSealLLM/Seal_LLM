"""② 답변 생성에 필요한 연결을 한 번씩 확인한다 (Ollama · OpenAI · 기상청 · Tavily).

uv run python scripts/answer/check_llm.py              # Ollama 호출 + 키 설정 여부만
uv run python scripts/answer/check_llm.py --openai     # + OpenAI 실제 호출 1회 (과금)
uv run python scripts/answer/check_llm.py --kma        # + 기상청 날씨 조회 1회 (서울)
uv run python scripts/answer/check_llm.py --tavily     # + Tavily 검색 1회 (사용량 차감)
uv run python scripts/answer/check_llm.py --all        # 전부

- 설정은 코드 기본값 + .env (theseal.common.config · theseal.prompt.router). 키 값은 출력하지 않는다
- 키는 쓰는 곳이 있을 때만 필수다. OpenAI 키는 기본 provider나 LLM 검증(VERIFY_PROVIDER)이 openai일 때,
  또는 --openai · --all로 호출을 요청했을 때만 없으면 실패. 그 밖에는 선택 사항으로 표시한다
- 하나라도 실패하면 종료 코드 1
"""

from __future__ import annotations

import argparse
import asyncio
import time

import _common
from langchain_core.messages import HumanMessage

from theseal.answer.tools.api_weather import get_weather
from theseal.answer.tools.api_web_search import web_search
from theseal.common.client import ConfigError, llm_options, make_llm
from theseal.common.config import LLMConfig
from theseal.prompt.router import RouterConfig

PING = "한 단어로만 답하세요: 안녕하세요"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM · 검색 연결 확인")
    p.add_argument("--openai", action="store_true", help="OpenAI 실제 호출 1회 (과금)")
    p.add_argument("--kma", action="store_true", help="기상청 날씨 조회 1회 (서울)")
    p.add_argument("--tavily", action="store_true", help="Tavily 검색 1회")
    p.add_argument("--all", action="store_true", help="--openai --kma --tavily")
    return p.parse_args(argv)


async def ping_llm(provider: str, config: LLMConfig) -> tuple[bool, str]:
    start = time.perf_counter()
    try:
        llm = make_llm(provider, config)
        ai = await asyncio.wait_for(llm.ainvoke([HumanMessage(PING)]), config.llm_timeout_sec)
    except ConfigError as e:
        return False, str(e)
    except TimeoutError:
        return False, f"응답 시간 초과 ({config.llm_timeout_sec}초)"
    except Exception as e:  # noqa: BLE001 — 연결 실패 등은 결과 표에 남긴다
        return False, f"{type(e).__name__}: {e}"[:200]
    ms = round((time.perf_counter() - start) * 1000)
    tokens = (ai.usage_metadata or {}).get("output_tokens")
    return True, f"{ms}ms · 출력 토큰 {tokens} · {llm_options(provider, config)}"


def check_ollama(config: LLMConfig) -> list[tuple[str, bool | None, str]]:
    version = _common.ollama_version(config)
    if version is None:
        return [("Ollama 서버", False, f"{config.ollama_base_url}에 연결할 수 없음 (fetch_models.py 안내 참고)")]
    rows = [("Ollama 서버", True, f"{config.ollama_base_url} · 버전 {version}")]
    info = _common.find_model(config, config.ollama_model)
    if info is None:
        return rows + [("로컬 모델", False, f"{config.ollama_model} 없음 → scripts/answer/fetch_models.py")]
    ok, message = _common.check_model_id(config.ollama_model, info)
    rows.append(("로컬 모델", ok if ok is not None else True, f"{config.ollama_model} · {message}"))
    ok, message = asyncio.run(ping_llm("local", config))
    rows.append(("로컬 호출", ok, message))
    return rows


def openai_users(config: LLMConfig, routing: RouterConfig) -> list[str]:
    """OpenAI를 기본으로 쓰는 곳 (있으면 키가 필수)."""
    users = []
    if config.default_provider == "openai":
        users.append("DEFAULT_PROVIDER")
    if routing.verify_provider == "openai" and routing.verify_mode != "off":
        users.append("VERIFY_PROVIDER")
    return users


def check_openai(config: LLMConfig, call: bool, routing: RouterConfig | None = None) -> list[tuple[str, bool | None, str]]:
    if not config.openai_api_key:
        users = openai_users(config, routing or RouterConfig())
        if users:
            return [("OpenAI 키", False, f"OPENAI_API_KEY가 비어 있음 (.env) · {', '.join(users)}=openai라 필요")]
        if call:
            return [("OpenAI 키", False, "OPENAI_API_KEY가 비어 있음 (.env) · --openai로 호출을 요청함")]
        return [("OpenAI 키", None, '비어 있음 → provider "openai" 요청만 config_error, 기본은 로컬 LLM (선택 사항)')]
    rows = [("OpenAI 키", True, f"설정됨 · 모델 {config.openai_model}")]
    if not call:
        return rows + [("OpenAI 호출", None, "건너뜀 (--openai로 확인)")]
    ok, message = asyncio.run(ping_llm("openai", config))
    return rows + [("OpenAI 호출", ok, message)]


def check_kma(config: LLMConfig, call: bool) -> list[tuple[str, bool | None, str]]:
    if not config.kma_auth_key:
        return [("기상청 키", None, "비어 있음 → 날씨 툴이 자동으로 빠지고 검색 툴이 날씨에 답함 (선택 사항)")]
    rows = [("기상청 키", True, "설정됨")]
    if not call:
        return rows + [("기상청 조회", None, "건너뜀 (--kma로 확인)")]
    start = time.perf_counter()
    try:
        w = get_weather.invoke({"location": "서울"})
    except Exception as e:  # noqa: BLE001
        return rows + [("기상청 조회", False, f"{type(e).__name__}: {e}"[:200])]
    ms = round((time.perf_counter() - start) * 1000)
    now = w["now"]
    message = (f"{ms}ms · 서울 {now['temperature_c']}도 · 예보 {len(w['forecast'])}일 "
               f"(발표 {w['forecast_issued']})")
    return rows + [("기상청 조회", True, message)]


def check_tavily(config: LLMConfig, call: bool) -> list[tuple[str, bool | None, str]]:
    if not config.tavily_api_key:
        return [("Tavily 키", None, "비어 있음 → web_search 툴이 자동으로 빠짐 (선택 사항)")]
    rows = [("Tavily 키", True, "설정됨")]
    if not call:
        return rows + [("Tavily 검색", None, "건너뜀 (--tavily로 확인)")]
    start = time.perf_counter()
    try:
        results = web_search.invoke({"query": "대한민국 수도"})
    except Exception as e:  # noqa: BLE001
        return rows + [("Tavily 검색", False, f"{type(e).__name__}: {e}"[:200])]
    ms = round((time.perf_counter() - start) * 1000)
    return rows + [("Tavily 검색", bool(results), f"{ms}ms · 결과 {len(results)}건")]


def main(argv: list[str] | None = None) -> list[tuple[str, bool | None, str]]:
    args = parse_args(argv)
    config = LLMConfig.from_env()
    rows = (check_ollama(config)
            + check_openai(config, args.openai or args.all, RouterConfig.from_env())
            + check_kma(config, args.kma or args.all)
            + check_tavily(config, args.tavily or args.all))

    print(f"\n=== LLM 연결 확인 (기본 provider: {config.default_provider}) ===")
    marks = {True: "OK  ", False: "실패", None: "-   "}
    for name, ok, message in rows:
        print(f"  [{marks[ok]}] {name:<12} {message}")
    failed = [name for name, ok, _ in rows if ok is False]
    print(f"\n{'실패: ' + ', '.join(failed) if failed else '모두 정상'}")
    if failed:
        raise SystemExit(1)
    return rows


if __name__ == "__main__":
    main()
