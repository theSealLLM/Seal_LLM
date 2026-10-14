from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# 팀 프로젝트의 src 경로를 Python에 등록
import _common

# 팀에서 구현한 LLM 연결 기능을 그대로 재사용
from theseal.common.client import make_llm, model_name
from theseal.common.config import LLMConfig

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs" / "llm" / "prompt_comparison"
TIMEZONE = ZoneInfo("Asia/Seoul")

# 추가 API 비용 없이 현재 Local LLM으로 생성 답변을 자동 평가
EVALUATION_PROVIDER = "local"

# 답변 생성 모델과 평가 모델이 같아 자기평가 편향 가능성이 있음
EVALUATION_LIMITATION = (
    "답변 생성 모델과 평가 모델이 동일하여 자기평가 편향 가능성이 있음"
)


class QualityEvaluationOutput(BaseModel):
    """LLM이 평가한 답변 품질 결과."""

    # 충족한 필수 정보의 개수
    required_points_met_count: int = Field(
        ge=0,
        description="답변이 충족한 필수 정보의 개수",
    )

    # 답변에서 발생한 금지 주장의 개수
    forbidden_claims_detected_count: int = Field(
        ge=0,
        description="답변에 포함된 금지 주장의 개수",
    )

    # 서비스 정책 준수 여부
    policy_compliant: bool = Field(
        description="답변이 제공된 상담 정책을 준수했는지 여부"
    )

    # 질문에 직접 답했는지 평가
    relevance_score: int = Field(
        ge=1,
        le=5,
        description="고객 질문과 답변의 관련성을 1점부터 5점으로 평가",
    )

    # 근거 없이 추측하지 않고 정확하게 답했는지 평가
    accuracy_score: int = Field(
        ge=1,
        le=5,
        description="제공된 정책과 평가 기준에 따른 답변 정확성을 1점부터 5점으로 평가",
    )

    # 평가 결과를 확인하기 위한 간단한 근거
    reason: str = Field(
        description="평가 점수와 판정의 간단한 근거"
    )
    
    
def build_quality_evaluation_messages(record: dict) -> list:
    """생성 답변의 품질 평가에 사용할 메시지를 만든다."""

    # 생성 답변 평가에 필요한 정보만 구성
    evaluation_data = {
        "customer_question": record["input_text"],
        "label": record["expected_label"],
        "answer": record["answer_result"]["answer"],
        "policy": record["prompt_spec"]["system"],
        "required_points": record["required_points"],
        "required_point_count": len(record["required_points"]),
        "forbidden_claims": record["forbidden_claims"],
        "forbidden_claim_count": len(record["forbidden_claims"]),
        "rubric_note": record["rubric_note"],
    }

    return [
        SystemMessage(
            content=(
                "당신은 고객지원 답변 품질 평가자다. "
                "제공된 정책과 평가 기준만 근거로 판단한다. "
                "표현이 달라도 의미가 같으면 인정한다. "
                "충족한 required_points의 개수와 답변에서 발생한 "
                "forbidden_claims의 개수를 정수로 답한다. "
                "각 개수는 제공된 전체 항목 수를 초과할 수 없다."
            )
        ),
        HumanMessage(
            # 한글이 깨지지 않도록 평가 정보를 JSON 문자열로 전달
            content=json.dumps(
                evaluation_data,
                ensure_ascii=False,
                indent=2,
            )
        ),
    ]


def find_response_files() -> list[Path]:
    """가장 최근 A 우선·B 우선 실행의 응답 파일을 찾는다."""

    # 이전 이름과 수정된 이름을 모두 읽어 기존 결과도 재사용
    response_files = sorted(
        RUNS_DIR.glob("*/prompt*_responses_*.json")
    )
    latest_by_order = {}

    for response_file in response_files:
        run_id = response_file.parent.name

        is_legacy_ab = run_id.startswith(
            "prompt_ab_"
        ) and not run_id.startswith("prompt_ab_order_")

        if run_id.startswith("prompt_ab_order_ab_") or is_legacy_ab:
            execution_order = "ab"
        elif run_id.startswith("prompt_ab_order_ba_") or run_id.startswith(
            "prompt_ba_"
        ):
            execution_order = "ba"
        else:
            continue

        previous_file = latest_by_order.get(execution_order)

        if (
            previous_file is None
            or response_file.stat().st_mtime > previous_file.stat().st_mtime
        ):
            latest_by_order[execution_order] = response_file

    if set(latest_by_order) != {"ab", "ba"}:
        raise FileNotFoundError(
            "A 우선과 B 우선 실험 응답 파일이 모두 필요합니다."
        )

    # 동일한 A/B 실험의 실행 순서 두 가지를 한 쌍으로 반환
    return [latest_by_order["ab"], latest_by_order["ba"]]
    
    
def load_responses(response_files: list[Path]) -> list[dict]:
    """여러 실험의 생성 응답을 하나의 목록으로 불러온다."""

    # 모든 실험의 응답을 합칠 목록
    responses = []

    for response_file in response_files:
        # 응답 JSON 파일을 파이썬 목록으로 변환
        records = json.loads(
            response_file.read_text(encoding="utf-8")
        )

        for record in records:
            # 각 응답이 어느 실험에서 생성됐는지 기록
            record["run_id"] = response_file.parent.name
            responses.append(record)

    return responses


async def evaluate_response(
    llm: BaseChatModel,
    record: dict,
    evaluation_model: str,
) -> dict:
    """생성 답변 한 건을 LLM으로 자동 평가한다."""

    start = time.perf_counter()

    # 답변 생성에 실패한 기록은 품질을 평가하지 않음
    if record["answer_result"]["status"] != "ok":
        return {
            "run_id": record["run_id"],
            "sample_id": record["sample_id"],
            "prompt_method": record["prompt_method"],
            "prompt_structure": record["prompt_structure"],
            "status": "skipped",
            "error": "생성 답변이 없어 평가하지 않음",
        }

    try:
        # 팀의 LLM 검증 코드와 동일한 구조화 출력 방식
        result: QualityEvaluationOutput = await llm.with_structured_output(
            QualityEvaluationOutput
        ).ainvoke(
            build_quality_evaluation_messages(record)
        )

        # 평가 개수가 원본 평가 항목 수를 넘지 않는지 확인
        required_point_count = len(record["required_points"])
        forbidden_claim_count = len(record["forbidden_claims"])

        if result.required_points_met_count > required_point_count:
            raise ValueError("필수 정보 충족 개수가 전체 개수를 초과했습니다.")

        if result.forbidden_claims_detected_count > forbidden_claim_count:
            raise ValueError("금지 주장 발생 개수가 전체 개수를 초과했습니다.")

        evaluation = result.model_dump()
        evaluation["required_point_count"] = required_point_count
        evaluation["forbidden_claim_count"] = forbidden_claim_count

        return {
            "run_id": record["run_id"],
            "sample_id": record["sample_id"],
            "prompt_method": record["prompt_method"],
            "prompt_structure": record["prompt_structure"],
            "status": "ok",
            "evaluation_provider": EVALUATION_PROVIDER,
            "evaluation_model": evaluation_model,
            "evaluation_latency_ms": round(
                (time.perf_counter() - start) * 1000
            ),
            "evaluation": evaluation,
            "error": None,
        }

    except Exception as error:
        # 한 건의 평가 실패로 전체 실험이 중단되지 않도록 오류를 기록
        return {
            "run_id": record["run_id"],
            "sample_id": record["sample_id"],
            "prompt_method": record["prompt_method"],
            "prompt_structure": record["prompt_structure"],
            "status": "error",
            "evaluation_provider": EVALUATION_PROVIDER,
            "evaluation_model": evaluation_model,
            "evaluation_latency_ms": round(
                (time.perf_counter() - start) * 1000
            ),
            "evaluation": None,
            "error": f"{type(error).__name__}: {error}",
        }
        
        
def save_evaluation_results(
    output_path: Path,
    results: list[dict],
    evaluation_model: str,
    expected_response_count: int,
    source_response_files: list[Path],
) -> None:
    """자동 평가 결과를 JSON 파일로 저장한다."""

    status_counts = {}

    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    output_data = {
        "created_at": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
        "evaluation_provider": EVALUATION_PROVIDER,
        "evaluation_model": evaluation_model,
        "evaluation_limitation": EVALUATION_LIMITATION,
        "source_run_ids": [path.parent.name for path in source_response_files],
        "source_response_files": [path.name for path in source_response_files],
        "expected_response_count": expected_response_count,
        "evaluated_response_count": len(results),
        "completed": len(results) == expected_response_count,
        "status_counts": status_counts,
        "results": results,
    }

    # 한글을 유지하고 사람이 읽기 쉬운 형태로 저장
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    

async def evaluate_all_responses(
    llm: BaseChatModel,
    responses: list[dict],
    evaluation_model: str,
    output_path: Path,
    source_response_files: list[Path],
) -> list[dict]:
    """모든 생성 답변을 순서대로 평가하고 저장한다."""

    results = []
    total_count = len(responses)

    for index, record in enumerate(responses, start=1):
        # 생성 답변 한 건을 자동 평가
        result = await evaluate_response(
            llm,
            record,
            evaluation_model,
        )
        results.append(result)

        # 실행이 중단되어도 완료된 평가 결과는 유지
        save_evaluation_results(
            output_path,
            results,
            evaluation_model,
            total_count,
            source_response_files,
        )

        # 각 응답의 실제 평가 성공 또는 실패 상태 표시
        print(
            f"[{index}/{total_count}] "
            f"{record['sample_id']} "
            f"{record['prompt_structure']} "
            f"상태: {result['status']}"
        )

    return results


async def main() -> None:
    """저장된 A/B 응답을 Local LLM으로 자동 평가한다."""

    # 평가할 A/B 응답 파일 확인
    response_files = find_response_files()

    if not response_files:
        raise FileNotFoundError("평가할 프롬프트 응답 파일이 없습니다.")

    responses = load_responses(response_files)

    # 필수님이 구현한 공통 LLM 연결 기능 사용
    config = LLMConfig.from_env()
    llm = make_llm(EVALUATION_PROVIDER, config)
    evaluation_model = model_name(EVALUATION_PROVIDER, config)

    # 기존 결과를 덮어쓰지 않도록 실행 시각을 파일명에 포함
    evaluation_time = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    evaluation_run_dir = RUNS_DIR / (
        f"prompt_ab_quality_{EVALUATION_PROVIDER}_{evaluation_time}"
    )
    evaluation_run_dir.mkdir(parents=True, exist_ok=False)
    output_path = evaluation_run_dir / (
        f"prompt_ab_quality_evaluation_"
        f"{EVALUATION_PROVIDER}_{evaluation_time}.json"
    )

    results = await evaluate_all_responses(
        llm,
        responses,
        evaluation_model,
        output_path,
        response_files,
    )

    status_counts = {}

    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"평가 대상 응답 수: {len(responses)}")
    print(f"평가 상태별 결과 수: {status_counts}")
    print(f"평가 결과 저장 경로: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

