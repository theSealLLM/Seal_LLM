from __future__ import annotations

import json
from pathlib import Path

# 팀 프로젝트의 최상위 경로
ROOT = Path(__file__).resolve().parents[2]

# 프롬프트 실험 및 품질 평가 결과 경로
RUNS_DIR = ROOT / "runs" / "llm" / "prompt_comparison"


def find_latest_evaluation_file() -> Path:
    """가장 최근에 완료된 품질 평가 JSON 파일을 찾는다."""

    # 이전 위치와 수정된 실행별 폴더를 모두 확인
    evaluation_files = list(
        RUNS_DIR.glob("prompt_quality_evaluation_*.json")
    ) + list(
        RUNS_DIR.glob(
            "prompt_ab_quality_*/prompt_ab_quality_evaluation_*.json"
        )
    )

    if not evaluation_files:
        raise FileNotFoundError("품질 평가 결과 파일이 없습니다.")

    completed_files = []

    for evaluation_file in evaluation_files:
        evaluation_data = json.loads(
            evaluation_file.read_text(encoding="utf-8")
        )
        results = evaluation_data.get("results", [])
        completed = evaluation_data.get(
            "completed",
            evaluation_data.get("evaluated_response_count") == len(results)
            and bool(results),
        )

        all_successful = bool(results) and all(
            result.get("status") == "ok" for result in results
        )

        if completed and all_successful:
            completed_files.append(evaluation_file)

    if not completed_files:
        raise FileNotFoundError("완료된 품질 평가 결과가 없습니다.")

    # 중단된 파일을 제외하고 가장 최근 완료 파일 반환
    return max(completed_files, key=lambda path: path.stat().st_mtime)


def load_evaluation_results(evaluation_file: Path) -> dict:
    """품질 평가 JSON 파일을 불러온다."""

    # JSON 문자열을 파이썬 딕셔너리로 변환
    return json.loads(
        evaluation_file.read_text(encoding="utf-8")
    )
    

def summarize_prompt_quality(
    results: list[dict],
    prompt_structure: str,
) -> dict:
    """한 가지 프롬프트 방식의 품질 지표를 계산한다."""

    # 평가에 성공한 현재 프롬프트 방식의 결과만 선택
    selected_results = [
        result
        for result in results
        if result["status"] == "ok"
        and result["prompt_structure"] == prompt_structure
    ]

    if not selected_results:
        raise ValueError(
            f"{prompt_structure}의 정상 평가 결과가 없습니다."
        )

    # 전체 필수 정보 수와 충족한 필수 정보 수를 각각 합산
    required_point_count = sum(
        result["evaluation"]["required_point_count"]
        for result in selected_results
    )
    required_points_met_count = sum(
        result["evaluation"]["required_points_met_count"]
        for result in selected_results
    )

    # 금지 주장이 하나 이상 발생한 응답 수
    forbidden_claim_response_count = sum(
        result["evaluation"]["forbidden_claims_detected_count"] > 0
        for result in selected_results
    )

    # 정책을 준수한 응답 수
    policy_compliance_count = sum(
        result["evaluation"]["policy_compliant"]
        for result in selected_results
    )

    response_count = len(selected_results)

    return {
        "experiment_group": selected_results[0]["prompt_method"],
        "prompt_structure": prompt_structure,
        "response_count": response_count,
        "required_point_fulfillment_percent": round(
            required_points_met_count
            / required_point_count
            * 100,
            2,
        ),
        "forbidden_claim_occurrence_percent": round(
            forbidden_claim_response_count
            / response_count
            * 100,
            2,
        ),
        "policy_compliance_percent": round(
            policy_compliance_count
            / response_count
            * 100,
            2,
        ),
        "average_relevance_score": round(
            sum(
                result["evaluation"]["relevance_score"]
                for result in selected_results
            )
            / response_count,
            2,
        ),
        "average_accuracy_score": round(
            sum(
                result["evaluation"]["accuracy_score"]
                for result in selected_results
            )
            / response_count,
            2,
        ),
    }
    
    
def build_quality_summary(
    evaluation_file: Path,
    evaluation_data: dict,
) -> dict:
    """A/B 프롬프트 품질 집계 결과를 구성한다."""

    results = evaluation_data["results"]

    # 평가 성공과 실패 건수를 상태별로 집계
    status_counts = {}

    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    # 원본 평가 파일과 연결되는 집계 결과 구성
    return {
        "summary_id": evaluation_file.stem.replace(
            "evaluation",
            "summary",
        ),
        "source_evaluation_file": evaluation_file.name,
        "evaluation_provider": evaluation_data["evaluation_provider"],
        "evaluation_model": evaluation_data["evaluation_model"],
        "evaluation_limitation": evaluation_data[
            "evaluation_limitation"
        ],
        "total_evaluation_count": len(results),
        "status_counts": status_counts,
        "prompt_methods": {
            # A 방식: 라벨별 독립 프롬프트
            "label_specific": summarize_prompt_quality(
                results,
                "label_specific",
            ),
            # B 방식: 공통 지침과 유형별 정책 조합
            "common_policy": summarize_prompt_quality(
                results,
                "common_policy",
            ),
        },
    }
    
    
def save_quality_summary(
    evaluation_file: Path,
    summary: dict,
) -> Path:
    """품질 집계 결과를 JSON 파일로 저장한다."""

    # evaluation을 summary로 바꿔 원본 평가 파일과 연결
    output_path = evaluation_file.with_name(
        f"{summary['summary_id']}.json"
    )

    # 한글을 유지하고 사람이 읽기 쉬운 형태로 저장
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    """최신 자동 평가 결과의 A/B 품질 지표를 집계한다."""

    # 가장 최근 품질 평가 결과 선택
    evaluation_file = find_latest_evaluation_file()

    # 평가 결과 JSON 불러오기
    evaluation_data = load_evaluation_results(evaluation_file)

    # A/B 프롬프트 품질 지표 계산
    summary = build_quality_summary(
        evaluation_file,
        evaluation_data,
    )

    # 집계 결과를 별도 JSON 파일로 저장
    output_path = save_quality_summary(
        evaluation_file,
        summary,
    )

    print(f"평가 결과 파일: {evaluation_file}")
    print(f"집계한 평가 수: {summary['total_evaluation_count']}")
    print(f"품질 집계 저장 경로: {output_path}")


if __name__ == "__main__":
    main()
