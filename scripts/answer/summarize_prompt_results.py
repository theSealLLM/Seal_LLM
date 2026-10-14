from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs" / "llm" / "prompt_comparison"
TIMEZONE = ZoneInfo("Asia/Seoul")

def find_run_ids() -> list[str]:
    """설정과 응답 JSON이 모두 있는 실험 실행 ID를 찾는다."""

    run_ids = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        settings_files = list(
            run_dir.glob("*_experiment_settings_*.json")
        )
        response_files = list(
            run_dir.glob("*_responses_*.json")
        )

        if len(settings_files) == 1 and len(response_files) == 1:
            run_ids.append(run_id)

    return sorted(run_ids)


def load_run_results(run_id: str) -> tuple[dict, list[dict]]:
    """실행 ID에 해당하는 실험 설정과 개별 응답을 불러온다."""

    run_dir = RUNS_DIR / run_id
    settings_files = list(
        run_dir.glob("*_experiment_settings_*.json")
    )
    response_files = list(
        run_dir.glob("*_responses_*.json")
    )

    # 필요한 결과 파일이 없으면 잘못된 실행 ID를 바로 알림
    if len(settings_files) != 1 or len(response_files) != 1:
        raise FileNotFoundError(
            f"실험 결과 파일을 찾을 수 없습니다: {run_dir}"
        )

    settings_path = settings_files[0]
    responses_path = response_files[0]
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    responses = json.loads(responses_path.read_text(encoding="utf-8"))

    return settings, responses


def summarize_prompt_method(records: list[dict]) -> dict:
    """같은 프롬프트 방식의 성공 여부, 시간과 토큰 사용량을 집계한다."""

    status_counts = Counter(
        record["answer_result"]["status"] for record in records
    )
    latency_values = [
        record["answer_result"]["latency_ms"] for record in records
    ]
    prompt_generation_values = [
        record["prompt_generation_ms"] for record in records
    ]
    input_token_values = [
        record["answer_result"]["usage"]["input_tokens"] for record in records
    ]
    output_token_values = [
        record["answer_result"]["usage"]["output_tokens"] for record in records
    ]

    return {
        "response_count": len(records),
        "status_counts": dict(status_counts),
        "average_latency_ms": round(mean(latency_values), 2),
        "average_prompt_generation_ms": round(
            mean(prompt_generation_values),
            4,
        ),
        "average_input_tokens": round(mean(input_token_values), 2),
        "average_output_tokens": round(mean(output_token_values), 2),
    }
    
    
def summarize_run(run_id: str, settings: dict, responses: list[dict]) -> dict:
    """한 번의 실험 결과를 프롬프트 방식별로 집계한다."""

    prompt_methods = [
        ("A", "label_specific"),
        ("B", "common_policy"),
    ]
    method_summaries = {}

    for experiment_group, prompt_method in prompt_methods:
        method_records = [
            record
            for record in responses
            if record["prompt_method"] == experiment_group
        ]
        method_summary = summarize_prompt_method(method_records)
        method_summary["experiment_group"] = experiment_group
        method_summaries[prompt_method] = method_summary

    return {
        "run_id": run_id,
        # 이전 실행에 저장되지 않은 항목은 추측하지 않고 null로 기록
        "execution_order": settings.get("execution_order"),
        "total_response_count": len(responses),
        "prompt_methods": method_summaries,
    }
    
    
def save_summary(run_summaries: list[dict]) -> Path:
    """원본 실행 ID가 드러나는 이름으로 집계 결과를 저장한다."""

    created_at = datetime.now(TIMEZONE)
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    summary_id = f"prompt_ab_comparison_summary_{timestamp}"
    output_path = RUNS_DIR / f"{summary_id}.json"

    summary_result = {
        "summary_id": summary_id,
        "created_at": created_at.isoformat(),
        "source_run_ids": [
            run_summary["run_id"] for run_summary in run_summaries
        ],
        "runs": run_summaries,
    }

    output_path.write_text(
        json.dumps(summary_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def main():
    """저장된 모든 프롬프트 실험 결과를 집계한다."""

    run_ids = find_run_ids()
    run_summaries = []

    for run_id in run_ids:
        settings, responses = load_run_results(run_id)
        run_summary = summarize_run(run_id, settings, responses)
        run_summaries.append(run_summary)

    output_path = save_summary(run_summaries)

    print(f"집계한 실행 수: {len(run_summaries)}")
    print(f"집계 결과 저장 경로: {output_path}")
    
if __name__ == "__main__":
    main()
