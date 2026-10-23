"""단계 실행. run_e2e.py의 번호마다 함수 하나다.

    2번 make_references   OpenAI 답변 + 채점 → data/.../<입력>_reference.jsonl
    1번 test_service      서비스 답변 → 같은 채점표로 채점(OpenAI) → OpenAI 점수와 비교 → results.jsonl · report.md

단계마다 결과를 JSONL에 한 줄씩 덧붙인다. 중단한 뒤 다시 돌리면 끝난 건은 건너뛰고, 끝나면 id마다 한 줄로 정리한다.
    검증용 답변  모델이나 버전(서비스 프롬프트 · 채점표)이 바뀐 건만 다시
    서비스 답변  status가 ok가 아닌 건, 서비스 모델 · 분류 모델이 바뀐 건만 다시 (Ollama가 꺼져 있었던 경우 등)
    채점        채점이 없거나 채점표가 바뀐 건만 다시 (서비스 답변을 다시 받으면 채점도 다시)
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from theseal.answer.config import AnswerConfig
from theseal.common.client import model_name
from theseal.common.config import LLMConfig

from .dataset import append_jsonl, compact_jsonl, latest_rows
from .judge import JUDGE_VERSION, module_hint, score_answer
from .metrics import report_md, summarize
from .rate_limit import OpenAIStopped, RateLimiter
from .reference import PROVIDER, make_reference, reference_version
from .rubric import MAX_SCORE
from .service import call_service

MAX_TOTAL = sum(MAX_SCORE.values())

# 아래 기본값은 함수를 직접 부를 때만 쓴다. 실행 설정은 scripts/e2e/run_e2e.py 상단에 모아 두고 거기서 넘긴다
STOP_AFTER = 3  # 문의 이만큼이 연달아 실패하면 멈춘다


class StageAborted(Exception):
    """멈춘 단계. 끝난 건은 파일에 남아 있으므로 원인을 고친 뒤 같은 명령으로 다시 실행하면 이어서 한다."""


def clip(text: str | None, n: int = 200) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "…"


async def run_stage(
    name: str,
    cases: list[dict],
    path: Path,
    fn: Callable[[dict], Awaitable[dict]],
    *,
    concurrency: int,
    is_done: Callable[[dict], bool] = lambda row: True,
    describe: Callable[[dict], str] = lambda row: "",
    error_of: Callable[[dict], str | None] = lambda row: None,
    stop_after: int = STOP_AFTER,
) -> dict[str, dict]:
    """문의마다 fn을 돌려 path에 한 줄씩 덧붙이고, {id: 최신 줄}을 돌려준다.

    성공하면 describe(줄)을 (답변 · 점수), 실패하면 오류를 터미널에 찍는다.
    fn이 예외를 내면 기록하지 않는다 → 다음 실행에서 그 건만 다시 돈다.
    error_of(줄)이 문자열이면(서비스 답변 오류) 기록은 하되 실패로 센다.
    문의 stop_after건이 연달아 실패하면 멈춘다. 같은 오류를 계속 보내 봐야 소용이 없고, 상대 서버에 부담만 준다."""
    existing = latest_rows(path)
    todo = [c for c in cases if c["id"] not in existing or not is_done(existing[c["id"]])]
    print(f"\n[{name}] {len(cases) - len(todo)}건은 이미 있음, {len(todo)}건 실행 → {path}")
    sem = asyncio.Semaphore(concurrency)
    finished = 0
    failures: list[tuple[str, str]] = []
    streak = 0  # 연속 실패 수

    def failed(case_id: str, error: str) -> None:
        nonlocal streak
        failures.append((case_id, error))
        streak += 1
        print(f"[{name}] 오류 {case_id}: {clip(error, 300)}", flush=True)
        if streak >= stop_after:
            raise StageAborted(f"[{name}] {stop_after}건이 연달아 실패해 멈춥니다. 원인을 고친 뒤 같은 명령으로 다시 "
                               f"실행하면 남은 건만 이어서 합니다. 마지막 오류: {clip(error, 300)}")

    async def one(case: dict) -> None:
        nonlocal finished, streak
        async with sem:
            try:
                row = await fn(case)
            except OpenAIStopped as e:
                raise StageAborted(f"[{name}] OpenAI 호출을 멈춥니다: {e}") from e
            except StageAborted:
                raise
            except Exception as e:  # noqa: BLE001 — 한 건 실패로 바로 멈추지 않는다 (연달아 실패하면 failed가 멈춘다)
                failed(case["id"], f"{type(e).__name__}: {e}")
                return
        append_jsonl(path, row)
        finished += 1
        error = error_of(row)
        if error:
            failed(case["id"], error)
            return
        streak = 0
        print(f"[{name}] {finished}/{len(todo)} {case['id']} | {describe(row)}", flush=True)

    await asyncio.gather(*(one(c) for c in todo))
    if failures:
        print(f"[{name}] 실패 {len(failures)}건. 같은 명령으로 다시 실행하면 이 건만 이어서 돈다")
    rows = latest_rows(path)
    return {c["id"]: rows[c["id"]] for c in cases if c["id"] in rows}


async def make_references(cases: list[dict], path: Path, *, config: LLMConfig, answer_config: AnswerConfig,
                          llm: BaseChatModel,
                          limiter: RateLimiter, concurrency: int, order: list[str],
                          stop_after: int = STOP_AFTER) -> dict[str, dict]:
    """2번: 검증용 데이터. llm은 make_llm("openai", config). order: 입력 파일 전체의 id 순서."""
    model = model_name(PROVIDER, config)
    by_id = {c["id"]: c for c in cases}

    def is_done(row: dict) -> bool:  # 모델 · 프롬프트 · 채점표 · 문의 문장이 그대로면 다시 만들지 않는다
        case = by_id[row["id"]]
        return row["model"] == model and row["version"] == reference_version(case) and row["text"] == case["text"]

    def describe(row: dict) -> str:
        return f"{row['expected_label']} · OpenAI {row['score']['total']}/{MAX_TOTAL}점 | {clip(row['answer'])}"

    try:
        return await run_stage(
            "검증용 답변", cases, path,
            lambda c: make_reference(c, config=config, answer_config=answer_config, llm=llm, limiter=limiter),
            concurrency=concurrency, is_done=is_done, describe=describe, stop_after=stop_after)
    finally:  # 중단돼도 파일은 입력 순서 · id마다 한 줄로 정리해 둔다 (팀이 커밋해서 같이 쓰는 파일)
        compact_jsonl(path, order)


async def score_service(case: dict, row: dict, *, llm: BaseChatModel, limiter: RateLimiter) -> dict:
    """우리 답변 한 줄을 채점해 같은 줄에 module · score를 붙인다. 서비스 오류로 답변이 없으면 모든 지표 0점."""
    ok = row["status"] == "ok" and bool(row["answer"])
    score = await score_answer(case, row["answer"] if ok else None, row["trace"]["tool_outputs"],
                               llm=llm, limiter=limiter)
    return {**row, "module": module_hint(row["trace"], ok), "score": score}


async def test_service(cases: list[dict], run_dir: Path, *, refs: dict[str, dict], chain: Runnable,
                       config: LLMConfig, answer_config: AnswerConfig, classifier: str, judge_llm: BaseChatModel | None, limiter: RateLimiter,
                       concurrency: int, meta: dict, service_concurrency: int = 1, stop_after: int = STOP_AFTER,
                       service_llm: BaseChatModel | None = None) -> dict:
    """1번: 서비스 답변 → 채점 → OpenAI 점수와 비교 → 리포트. 결과 파일은 두 개다.
        results.jsonl  문의마다 한 줄: 우리 답변 · 단계별 기록 · 채점 결과 (두 단계가 같은 줄을 채운다)
        report.md      항목별 · 라벨별 점수와 상세 분석 (results.jsonl과 검증용 데이터로 만든다)
    refs: 검증용 데이터 {id: 줄} (채점 버전이 같은 것만). 없는 건은 비교할 수 없으므로 채점하지 않는다.
    chain: pipeline.prepare_chain(clf). classifier: 분류 모델 이름. service_llm: 테스트용 가짜 LLM."""
    path = run_dir / "results.jsonl"
    model = model_name(config.default_provider, config)
    try:
        answered = await run_stage(
            "서비스 답변", cases, path,
            lambda c: call_service(c, chain=chain, provider=config.default_provider, config=config,
                                   answer_config=answer_config,
                                   classifier=classifier, llm=service_llm),
            concurrency=service_concurrency,
            # 같은 모델 · 분류 모델로 정상 답한 건은 다시 돌리지 않는다 (조건이 바뀌면 다시 돌려 섞이지 않게)
            is_done=lambda row: (row["status"] == "ok" and row["model"] == model
                                 and row.get("classifier") == classifier),
            describe=lambda row: (f"{row['trace']['expected_label']} → {row['trace']['final_label']} | "
                                  f"{clip(row['answer'])}"),
            error_of=lambda row: (f"{row['error']['code']}: {row['error']['message']}"
                                  if row["status"] == "error" else None),
            stop_after=stop_after)

        ready = [c for c in cases if c["id"] in refs and c["id"] in answered]
        if ready and judge_llm is not None:
            await run_stage(
                "채점", ready, path,
                lambda c: score_service(c, answered[c["id"]], llm=judge_llm, limiter=limiter),
                concurrency=concurrency,
                is_done=lambda row: bool(row.get("score")) and row["score"]["version"] == JUDGE_VERSION,
                describe=lambda row: (f"우리 {row['score']['total']}/{MAX_TOTAL}점 · "
                                      f"OpenAI {refs[row['id']]['score']['total']}/{MAX_TOTAL}점"),
                stop_after=stop_after)
    finally:  # 다시 돌린 건이 덧붙어 있으므로 문의 순서 · id마다 한 줄로 정리한다
        compact_jsonl(path, [c["id"] for c in cases])

    rows = {row_id: row for row_id, row in latest_rows(path).items() if row_id in {c["id"] for c in cases}}
    report = {
        "meta": meta,
        "no_reference_ids": [c["id"] for c in cases if c["id"] not in refs],
        # 앞 단계에서 실패해 빠진 건 (다시 실행하면 채워진다)
        "missing_ids": [c["id"] for c in cases
                        if c["id"] not in rows or (c["id"] in refs and not rows[c["id"]].get("score"))],
        "summary": summarize(cases, rows, refs),
    }
    (run_dir / "report.md").write_text(report_md(report, cases, rows, refs), encoding="utf-8")
    return report
