"""① 로컬 LLM(Ollama 모델)을 내려받고 팀 기준 버전인지 확인한다.

uv run python scripts/llm/fetch_models.py

- 받을 모델: .env의 OLLAMA_MODEL (기본: hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M, 약 6GB)
- 저장 위치: Ollama가 관리하는 폴더(~/.ollama). 프로젝트의 models/에는 두지 않는다
- 미리 Ollama를 설치하고 실행해 두어야 한다 (https://ollama.com/download)
- 이미 받은 모델은 건너뛴다. 다시 받으려면 --force
- 받은 뒤 모델 ID를 팀 기준(theseal.llm.config.OLLAMA_MODEL_IDS)과 비교한다
- 상용 LLM(OpenAI)은 받을 것이 없다. 키 · 연결 확인은 check_llm.py
"""

from __future__ import annotations

import argparse
import json

import _common
import httpx

from theseal.llm.config import LLMConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ollama 모델 내려받기 · 버전 확인")
    p.add_argument("--model", default=None, help="받을 모델 (기본: .env의 OLLAMA_MODEL)")
    p.add_argument("--force", action="store_true", help="이미 있어도 다시 받는다")
    return p.parse_args(argv)


def pull(config: LLMConfig, model: str) -> None:
    """Ollama /api/pull 스트림을 읽으며 진행률을 10% 단위로 출력한다."""
    last = -10
    with httpx.stream("POST", f"{config.ollama_base_url}/api/pull",
                      json={"model": model, "stream": True}, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if "error" in event:
                raise SystemExit(f"[중단] 내려받기 실패: {event['error']}")
            total, done = event.get("total"), event.get("completed")
            if total and done is not None:
                pct = int(done / total * 100)
                if pct >= last + 10 or pct == 100:
                    last = pct
                    print(f"  {event.get('status', '')[:30]:<30} {pct:3d}% "
                          f"({done / 2**30:.2f} / {total / 2**30:.2f} GB)")
            elif event.get("status"):
                print(f"  {event['status']}")


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    config = LLMConfig.from_env()
    model = args.model or config.ollama_model

    version = _common.require_ollama(config)
    print(f"[ollama] {config.ollama_base_url} · 버전 {version}")
    print(f"[model] {model}")

    info = _common.find_model(config, model)
    if info and not args.force:
        print(f"  이미 받아 둔 모델입니다 ({info.get('size', 0) / 2**30:.1f} GB). 다시 받으려면 --force")
    else:
        pull(config, model)
        info = _common.find_model(config, model)
        if info is None:
            raise SystemExit(f"[중단] 내려받은 뒤에도 모델을 찾을 수 없습니다: {model}")
        print(f"  다운로드 완료 ({info.get('size', 0) / 2**30:.1f} GB)")

    ok, message = _common.check_model_id(model, info)
    print(f"  {'[경고] ' if ok is False else ''}{message}")
    print("\n다음: uv run python scripts/llm/check_llm.py 로 연결을 확인한다")
    return {"model": model, "id": _common.short_id(info), "id_matches": ok}


if __name__ == "__main__":
    main()
