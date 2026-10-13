from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# 팀 프로젝트의 src 경로를 Python에 등록
import _common

# 팀에서 구현한 LLM 연결 기능을 그대로 재사용
from theseal.llm.client import make_llm, model_name
from theseal.llm.config import LLMConfig

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs" / "llm" / "prompt_comparison"

# 추가 API 비용 없이 현재 Local LLM으로 생성 답변을 자동 평가
EVALUATION_PROVIDER = "local"

# 답변 생성 모델과 평가 모델이 같아 자기평가 편향 가능성이 있음
EVALUATION_LIMITATION = (
    "답변 생성 모델과 평가 모델이 동일하여 자기평가 편향 가능성이 있음"
)


class QualityEvaluationOutput(BaseModel):
    """LLM이 평가한 답변 품질 결과."""

    # required_points 각 항목의 충족 여부
    required_points_met: list[bool] = Field(
        description="각 필수 정보가 답변에 포함되었는지 순서대로 표시"
    )

    # forbidden_claims 각 항목의 발생 여부
    forbidden_claims_detected: list[bool] = Field(
        description="각 금지 주장이 답변에 포함되었는지 순서대로 표시"
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


def find_response_files() -> list[Path]:
    """프롬프트 실험 응답 JSON 파일을 찾는다."""

    # A→B와 B→A 실험 폴더의 응답 JSON을 파일명 순서로 반환
    return sorted(
        RUNS_DIR.glob("prompt_*_*/prompt_*_responses_*.json")
    )
    
    
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

        # 평가 결과의 항목 수가 원본 평가 기준과 같은지 확인
        if len(result.required_points_met) != len(record["required_points"]):
            raise ValueError("필수 정보 평가 결과의 개수가 일치하지 않습니다.")

        if len(result.forbidden_claims_detected) != len(
            record["forbidden_claims"]
        ):
            raise ValueError("금지 주장 평가 결과의 개수가 일치하지 않습니다.")

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
            "evaluation": result.model_dump(),
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
) -> None:
    """자동 평가 결과를 JSON 파일로 저장한다."""

    output_data = {
        "evaluation_provider": EVALUATION_PROVIDER,
        "evaluation_model": evaluation_model,
        "evaluation_limitation": EVALUATION_LIMITATION,
        "evaluated_response_count": len(results),
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
        )

        print(
            f"[{index}/{total_count}] "
            f"{record['sample_id']} "
            f"{record['prompt_structure']} 평가 완료"
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
    evaluation_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RUNS_DIR / (
        f"prompt_quality_evaluation_"
        f"{EVALUATION_PROVIDER}_{evaluation_time}.json"
    )

    results = await evaluate_all_responses(
        llm,
        responses,
        evaluation_model,
        output_path,
    )

    print(f"평가 대상 응답 수: {len(responses)}")
    print(f"평가 완료 결과 수: {len(results)}")
    print(f"평가 결과 저장 경로: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

