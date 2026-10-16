"""E2E 평가. 번호로 할 일을 고른다.

  1) 우리 서비스 테스트    분류기 → 분류 후처리 → 프롬프트 → 로컬 LLM 답변을 받아 채점하고, OpenAI 점수와 비교 · 리포트
  2) 검증용 데이터 만들기  정답 라벨의 서비스 프롬프트로 OpenAI 답변을 받아 같은 채점표로 채점해 저장 (과금)

채점표는 src/theseal/e2e/rubric.py (사실 여부 10 · 중요 정보 포함 10 · 정책 위반 9개 지표, 지표마다 0/1점).

uv run python scripts/e2e/run_e2e.py                    # 번호 메뉴
uv run python scripts/e2e/run_e2e.py 2 --per-label 3    # 검증용 데이터를 라벨마다 3건만 (동작 확인)
uv run python scripts/e2e/run_e2e.py 2                  # 검증용 데이터 전체
uv run python scripts/e2e/run_e2e.py 1                  # 서비스 테스트

검증용 데이터  입력 파일 옆 <입력>_reference.jsonl (예: data/endtoend/questions_reference.jsonl). 팀이 같이 쓰도록 커밋
테스트 결과    runs/endtoend/<실행 시각>/  results.jsonl(문의마다 답변 · 채점 한 줄) · report.md(리포트)

진행하면서 답변 · 점수 · 오류를 터미널에 찍는다. 속도 · 재시도 · 멈춤 기준은 아래 '실행 설정'에 모아 두었다.
명령줄 옵션(--rpm · --concurrency · --stop-after)을 주면 그 실행에서만 설정값 대신 쓴다.

이어하기: 멈추거나 Ctrl+C로 끊으면 어디서 멈췄는지 runs/endtoend/resume.json에 적어 둔다. 같은 명령을 다시 치면
그 실행을 이어서 하고, 끝난 문의는 건너뛴다. 처음부터 새로 하려면 --new.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import _common  # noqa: F401 — src 경로 등록
import httpx

from theseal.api.pipeline import (
    ApiConfig,
    ClassifierNotReady,
    classifier_name,
    load_classifier,
    prepare_chain,
)
from theseal.classifier.config import PROJECT_ROOT, display_path, new_run_id
from theseal.common.client import ConfigError, make_llm, model_name
from theseal.common.config import LLMConfig, get_config
from theseal.e2e.dataset import latest_rows, load_cases, select_cases, write_json
from theseal.e2e.judge import JUDGE_VERSION
from theseal.e2e.rate_limit import RateLimiter
from theseal.e2e.reference import PROVIDER as OPENAI
from theseal.e2e.reference import reference_version
from theseal.e2e.rubric import CATEGORIES, MAX_SCORE
from theseal.e2e.runner import StageAborted, make_references, test_service
from theseal.prompt.router import RouterConfig

# ================================================================ 실행 설정 (여기만 고치면 된다)
# ---- OpenAI 호출 속도 (2번 검증용 데이터: 문의마다 답변 + 채점 2번, 1번 서비스 테스트: 채점 1번)
OPENAI_RPM = 40  # 분당 요청 수 상한. 호출 시작 간격 = 60 / OPENAI_RPM초 (20이면 3초, 문의 약 10건/분)
OPENAI_CONCURRENCY = 4  # 동시에 기다리는 호출 수. 답변이 느리면(10초 이상) 늘려야 RPM 상한까지 쓴다
# ---- 한도 초과(429) · 시간 초과 · 연결 오류일 때 (크레딧 소진 · 키 오류는 기다리지 않고 바로 멈춘다)
OPENAI_MAX_RETRIES = 3  # 한 호출을 다시 보내는 횟수
OPENAI_RETRY_DELAY = 5  # 첫 대기(초). 다시 보낼 때마다 두 배 (5 → 10 → 20초)
OPENAI_RETRY_MAX_DELAY = 120  # 대기 상한(초)
# ---- 멈춤 · 우리 서비스
STOP_AFTER = 3  # 문의가 이만큼 연달아 실패하면 멈춘다 (같은 명령으로 다시 실행하면 이어서)
SERVICE_CONCURRENCY = 1  # 우리 서비스(로컬 LLM, Ollama) 동시 실행 수. Ollama는 기본으로 한 번에 하나씩 답한다
# ---- 파일 위치
DEFAULT_DATA = PROJECT_ROOT / "data" / "endtoend" / "questions.csv"  # 검증용 데이터는 옆에 <입력>_reference.jsonl
RUNS_DIR = PROJECT_ROOT / "runs" / "endtoend"  # 1번 결과: RUNS_DIR/<실행 시각>/
RESUME_FILE = RUNS_DIR / "resume.json"  # 멈춘 실행을 기억해 두는 파일 (git에 올리지 않음)
# ================================================================
MODES = {
    "1": "우리 서비스 테스트 (채점해서 OpenAI 점수와 비교 · 리포트)",
    "2": "검증용 데이터 만들기 (OpenAI, 과금)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2E 평가 (1: 우리 서비스 테스트, 2: 검증용 데이터 만들기)")
    p.add_argument("mode", nargs="?", choices=list(MODES), help="생략하면 번호로 고른다")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help=f"CSV 또는 JSONL (기본: {display_path(DEFAULT_DATA)})")
    p.add_argument("--per-label", type=int, help="정답 라벨마다 앞에서 N건만")
    p.add_argument("--limit", type=int, help="앞에서 N건만 (--per-label 다음에 적용)")
    p.add_argument("--rpm", type=float, default=OPENAI_RPM, help=f"OpenAI 분당 요청 수 상한 (기본: {OPENAI_RPM})")
    p.add_argument("--concurrency", type=int, default=OPENAI_CONCURRENCY,
                   help=f"OpenAI 동시 호출 수 (기본: {OPENAI_CONCURRENCY})")
    p.add_argument("--stop-after", type=int, default=STOP_AFTER,
                   help=f"문의가 이만큼 연달아 실패하면 멈춘다 (기본: {STOP_AFTER})")
    p.add_argument("--new", action="store_true", help="멈춘 실행을 잇지 않고 처음부터 새로 한다")
    p.add_argument("--run-id", help="1번: 이어서 돌릴 실행 ID를 직접 고를 때")
    return p.parse_args()


def choose_mode() -> str:
    print("\n할 일을 고르세요.")
    for key, title in MODES.items():
        print(f"  {key}) {title}")
    while True:
        try:
            raw = input("번호: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n[중단] 번호를 고르지 않았습니다.") from None
        if raw in MODES:
            return raw
        print(f"  {' 또는 '.join(MODES)}를 입력하세요.")


def reference_path(data: Path) -> Path:
    return data.with_name(f"{data.stem}_reference.jsonl")


def openai_limiter(args: argparse.Namespace) -> RateLimiter:
    return RateLimiter(args.rpm, max_retries=OPENAI_MAX_RETRIES, base_delay=OPENAI_RETRY_DELAY,
                       max_delay=OPENAI_RETRY_MAX_DELAY)


def openai_llm(config: LLMConfig):
    try:
        return make_llm(OPENAI, config)
    except ConfigError as e:
        raise SystemExit(f"[중단] {e} (.env에 OPENAI_API_KEY)") from None


def require_ollama(config: LLMConfig) -> None:
    try:
        httpx.get(f"{config.ollama_base_url}/api/version", timeout=5).raise_for_status()
    except httpx.HTTPError:
        raise SystemExit(f"[중단] Ollama 서버에 연결할 수 없습니다: {config.ollama_base_url} "
                         "(Ollama 앱을 켜거나 `ollama serve`)") from None


# ---------------------------------------------------------------- 이어하기

def load_resume() -> dict:
    try:
        return json.loads(RESUME_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_resume(mode: str, entry: dict | None) -> None:
    """mode의 멈춘 지점을 적는다. entry가 None이면 (끝까지 했으면) 지운다."""
    state = load_resume()
    if entry is None:
        state.pop(mode, None)
    else:
        state[mode] = entry
    if state:
        RESUME_FILE.parent.mkdir(parents=True, exist_ok=True)
        write_json(RESUME_FILE, state)
    else:
        RESUME_FILE.unlink(missing_ok=True)


def apply_resume(mode: str, args: argparse.Namespace, data: Path) -> None:
    """같은 데이터로 멈춘 실행이 있으면 그 실행 · 선택 범위를 이어받는다 (명령에 직접 준 값이 먼저)."""
    entry = load_resume().get(mode)
    if not entry or entry["data"] != display_path(data) or args.new:
        return
    if args.per_label is None and args.limit is None:
        args.per_label, args.limit = entry["per_label"], entry["limit"]
    if mode == "1" and not args.run_id:
        args.run_id = entry["run_id"]
    print(f"[이어하기] {entry['stopped_time']}에 {entry['stopped_at']}에서 멈췄습니다 "
          f"({entry['done']}/{entry['total']}건 끝남). 남은 건부터 이어서 합니다. 처음부터 하려면 --new")


def stopped(mode: str, args: argparse.Namespace, data: Path, cases: list[dict], done_ids: set[str], reason: str,
            run_id: str | None = None) -> SystemExit:
    """멈춘 지점(아직 안 끝난 첫 문의)을 적어 두고, 다시 실행할 명령을 알려 준다."""
    done = sum(c["id"] in done_ids for c in cases)
    stopped_at = next((c["id"] for c in cases if c["id"] not in done_ids), None)
    save_resume(mode, {
        "data": display_path(data), "per_label": args.per_label, "limit": args.limit, "run_id": run_id,
        "stopped_at": stopped_at, "done": done, "total": len(cases), "reason": reason,
        "stopped_time": datetime.now(ZoneInfo(get_config().timezone)).strftime("%Y-%m-%d %H:%M:%S"),
    })
    return SystemExit(f"\n[중단] {reason}\n{done}/{len(cases)}건 끝남 · {stopped_at}에서 멈춤. "
                      f"같은 명령(uv run python scripts/e2e/run_e2e.py {mode})으로 다시 실행하면 이어서 합니다.")


# ---------------------------------------------------------------- 2) 검증용 데이터 만들기

def run_reference(args: argparse.Namespace, data: Path, all_cases: list[dict], cases: list[dict]) -> None:
    config = get_config()
    path = reference_path(data)
    print(f"검증용 데이터: {display_path(path)} · OpenAI {model_name(OPENAI, config)} · "
          f"{len(cases)}건 (호출 약 {len(cases) * 2}번, 분당 {args.rpm:g}번 이하)")

    def done_ids() -> set[str]:
        rows = latest_rows(path)
        return {c["id"] for c in cases if (row := rows.get(c["id"]))
                and row["version"] == reference_version(c) and row["text"] == c["text"]}

    try:
        refs = asyncio.run(make_references(
            cases, path, config=config, llm=openai_llm(config), limiter=openai_limiter(args),
            concurrency=args.concurrency, order=[c["id"] for c in all_cases], stop_after=args.stop_after))
    except StageAborted as e:
        raise stopped("2", args, data, cases, done_ids(), str(e)) from None
    except KeyboardInterrupt:
        raise stopped("2", args, data, cases, done_ids(), "Ctrl+C로 멈춤") from None

    # 이번 버전으로 만든 것만 센다 (실패한 건은 예전 줄이 남아 있을 수 있다)
    done = [refs[c["id"]] for c in cases if c["id"] in refs and refs[c["id"]]["version"] == reference_version(c)]
    tokens = [sum((r["usage"].get(k) or 0) for r in done) for k in ("input_tokens", "output_tokens")]
    print(f"\n=== 검증용 데이터 {len(done)}/{len(cases)}건 · 토큰 입력 {tokens[0]:,} 출력 {tokens[1]:,} (답변 생성분) ===")
    if done:
        means = {cat: sum(r["score"]["points"][cat] for r in done) / len(done) for cat in CATEGORIES}
        print("OpenAI 답변 평균 점수: " + " · ".join(f"{CATEGORIES[cat]} {means[cat]:.2f}/{MAX_SCORE[cat]}"
                                                  for cat in CATEGORIES)
              + f" · 문의만으로 뜻을 알 수 없다고 채점된 문의 {sum(not r['score']['answerable'] for r in done)}건")
    if len(done) < len(cases):
        raise stopped("2", args, data, cases, done_ids(), f"{len(cases) - len(done)}건이 실패해 남았습니다")
    save_resume("2", None)


# ---------------------------------------------------------------- 1) 우리 서비스 테스트

def run_test(args: argparse.Namespace, data: Path, cases: list[dict]) -> None:
    config = get_config()
    routing = RouterConfig.from_env()
    if "local" in (config.default_provider, routing.verify_provider):
        require_ollama(config)
    try:
        clf = load_classifier(ApiConfig.from_env())
    except ClassifierNotReady as e:
        raise SystemExit(f"[중단] {e}") from None

    # 검증용 데이터: 지금 모델 · 프롬프트 · 채점표 · 문의 문장으로 만든 것만 비교에 쓴다
    ref_path = reference_path(data)
    saved = latest_rows(ref_path)
    reference_model = model_name(OPENAI, config)
    refs = {c["id"]: saved[c["id"]] for c in cases
            if c["id"] in saved and saved[c["id"]]["model"] == reference_model
            and saved[c["id"]]["version"] == reference_version(c) and saved[c["id"]]["text"] == c["text"]}
    stale = sum(c["id"] in saved for c in cases) - len(refs)
    if len(refs) < len(cases):
        print(f"[알림] 검증용 데이터가 없는 {len(cases) - len(refs) - stale}건 · 모델이나 프롬프트 · 채점표가 바뀐 "
              f"{stale}건은 OpenAI와 비교하지 않습니다 (2번을 먼저 실행)")

    run_id = args.run_id or new_run_id()
    run_dir = RUNS_DIR / run_id  # 실행 ID = 실행 시각 (분류기와 같은 형식)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {  # report.md 머리말에 적는 실행 조건
        "run_id": run_id,
        "data": display_path(data),
        "n_cases": len(cases),
        "service": {
            "provider": config.default_provider,
            "model": model_name(config.default_provider, config),
            "classifier": classifier_name(clf),
            "route_rule_enabled": routing.route_rule_enabled,
            "verify_mode": routing.verify_mode,
        },
        "reference": {"provider": OPENAI, "model": reference_model, "path": display_path(ref_path)},
        "judge": {"provider": OPENAI, "model": model_name(OPENAI, config), "version": JUDGE_VERSION},
    }

    def done_ids() -> set[str]:
        rows = latest_rows(run_dir / "results.jsonl")
        return {c["id"] for c in cases if (row := rows.get(c["id"])) and row["status"] == "ok"
                and (c["id"] not in refs or bool(row.get("score")))}

    try:
        report = asyncio.run(test_service(
            cases, run_dir, refs=refs, chain=prepare_chain(clf), config=config, classifier=classifier_name(clf),
            judge_llm=openai_llm(config) if refs else None, limiter=openai_limiter(args),
            concurrency=args.concurrency, service_concurrency=SERVICE_CONCURRENCY, meta=meta,
            stop_after=args.stop_after))
    except StageAborted as e:
        raise stopped("1", args, data, cases, done_ids(), str(e), run_id) from None
    except KeyboardInterrupt:
        raise stopped("1", args, data, cases, done_ids(), "Ctrl+C로 멈춤", run_id) from None

    s, sv = report["summary"], report["summary"]["service"]
    print(f"\n=== {run_id} · 테스트 {sv['n_tested']}건 · OpenAI와 비교 {s['n_compared']}건 ===")
    if not s["n_compared"]:
        print("OpenAI와 비교할 검증용 데이터가 없어 점수는 건너뛰었습니다 (2번을 먼저 실행)")
    for cat, v in s["categories"].items() if s["n_compared"] else ():
        name = CATEGORIES.get(cat, "합계")
        print(f"{name:<10} 평균 우리 {v['ours_mean']} / OpenAI {v['openai_mean']} (차이 {v['gap_mean']}) · "
              f"전체 우리 {v['ours_total']} / OpenAI {v['openai_total']} (차이 {v['gap_total']}) · 만점 {v['max']}")
    print(f"라벨 정확도 {sv['label_accuracy']}% | 안내 문구 비율 {sv['fallback_rate']}% | "
          f"서비스 오류 {sv['n_service_error']}건 | 지연 중앙값 {sv['latency_ms_median']}ms")
    print(f"리포트: {display_path(run_dir / 'report.md')}")
    if report["missing_ids"]:
        raise stopped("1", args, data, cases, done_ids(), f"{len(report['missing_ids'])}건이 실패해 남았습니다",
                      run_id)
    save_resume("1", None)


def main() -> None:
    args = parse_args()
    mode = args.mode or choose_mode()
    data = args.data if args.data.is_absolute() else Path.cwd() / args.data
    apply_resume(mode, args, data)
    all_cases = load_cases(data)
    cases = select_cases(all_cases, args.per_label, args.limit)
    if not cases:
        raise SystemExit("[중단] 평가할 문의가 없습니다.")
    if mode == "2":
        run_reference(args, data, all_cases, cases)
    else:
        run_test(args, data, cases)


if __name__ == "__main__":
    main()
