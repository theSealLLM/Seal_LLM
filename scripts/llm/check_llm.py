"""② 답변 생성에 필요한 연결을 한 번씩 확인한다 (Ollama · OpenAI · Tavily).

uv run python scripts/llm/check_llm.py              # Ollama 호출 + 키 설정 여부만
uv run python scripts/llm/check_llm.py --openai     # + OpenAI 실제 호출 1회 (과금)
uv run python scripts/llm/check_llm.py --tavily     # + Tavily 검색 1회 (사용량 차감)
uv run python scripts/llm/check_llm.py --all        # 전부

- 설정은 .env에서 읽는다 (theseal.llm.config). 키 값은 출력하지 않는다
- 하나라도 실패하면 종료 코드 1
"""

from __future__ import annotations

import argparse
import asyncio
import time

import _common
from langchain_core.messages import HumanMessage

from theseal.llm.answer.tools.web_search import web_search
from theseal.llm.client import ConfigError, llm_options, make_llm
from theseal.llm.config import LLMConfig

PING = "한 단어로만 답하세요: 안녕하세요"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM · 검색 연결 확인")
    p.add_argument("--openai", action="store_true", help="OpenAI 실제 호출 1회 (과금)")
    p.add_argument("--tavily", action="store_true", help="Tavily 검색 1회")
    p.add_argument("--all", action="store_true", help="--openai --tavily")
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
        return rows + [("로컬 모델", False, f"{config.ollama_model} 없음 → scripts/llm/fetch_models.py")]
    ok, message = _common.check_model_id(config.ollama_model, info)
    rows.append(("로컬 모델", ok if ok is not None else True, f"{config.ollama_model} · {message}"))
    ok, message = asyncio.run(ping_llm("local", config))
    rows.append(("로컬 호출", ok, message))
    return rows


def check_openai(config: LLMConfig, call: bool) -> list[tuple[str, bool | None, str]]:
    if not config.openai_api_key:
        return [("OpenAI 키", False, "OPENAI_API_KEY가 비어 있음 (.env)")]
    rows = [("OpenAI 키", True, f"설정됨 · 모델 {config.openai_model}")]
    if not call:
        return rows + [("OpenAI 호출", None, "건너뜀 (--openai로 확인)")]
    ok, message = asyncio.run(ping_llm("openai", config))
    return rows + [("OpenAI 호출", ok, message)]


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
            + check_openai(config, args.openai or args.all)
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
