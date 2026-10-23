"""동일한 Local LLM으로 두 프롬프트 방식을 비교한다"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib  import Path
from zoneinfo import ZoneInfo

import _common

from theseal.answer.answer_generator import generate
from theseal.common.client import make_llm
from theseal.common.config import LLMConfig
from theseal.prompt import(
    build_label_specific_prompt,
    build_common_policy_prompt,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "inquiries" / "generation_development.jsonl"
RUNS_DIR = ROOT / "runs" / "llm" / "prompt_comparison"
TIMEZONE = ZoneInfo("Asia/Seoul")

PROMPT_BUILDERS = [
    ("A", "label_specific", build_label_specific_prompt),
    ("B", "common_policy", build_common_policy_prompt),
]


def load_development_cases(path: Path) -> list[dict]:
    """생성 프롬프트 개발용 JSONL 데이터를 읽는다"""
    
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    

def parse_args() -> argparse.Namespace:
    """터미널에서 프롬프트 실행 순서를 읽는다."""
    
    parser = argparse.ArgumentParser()
    
    # --order 생략 시 A → B, ba 지정 시 B → A
    parser.add_argument(
        "--order",
        choices=["ab", "ba"],
        default="ab"
    )
    
    # 선택값은 반환 객체의 order 속성으로 접근
    return parser.parse_args()
    
    
def create_run_directory(base_dir: Path, execution_order: str) -> tuple[str, Path]:
    """실험 결과가 덮어쓰이지 않도록 실행별 폴더를 만든다."""
    
    # 실험명(prompt_ab)과 실행 순서(order_ab/order_ba)를 구분
    timestamp = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    run_id = f"prompt_ab_order_{execution_order}_{timestamp}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    
    return run_id, run_dir


async def generate_prompt_responses_for_case(
    case: dict,
    llm,
    config: LLMConfig,
    prompt_builders: list,
) -> list[dict]:
    """문의 1건에 A/B 프롬프트를 적용하고 Local LLM 응답 기록을 반환한다."""

    # 같은 문의에서 생성된 A/B 응답 기록을 순서대로 저장
    records = []

    # 실행 옵션에서 선택한 ab 또는 ba 순서로 실행
    for prompt_method, prompt_structure, prompt_builder in prompt_builders:
        # LLM 응답 시간을 제외한 프롬프트 생성 시간 측정 시작
        prompt_start = time.perf_counter()

        # 정답 라벨을 이용해 문의 유형에 맞는 정책 선택
        # 평가 기준인 required_points와 forbidden_claims는 모델에 전달하지 않음
        prompt_spec = prompt_builder(
            case["text"],
            case["expected_label"],
        )

        # 프롬프트 생성에 걸린 시간을 밀리초 단위로 계산
        prompt_generation_ms = round(
            (time.perf_counter() - prompt_start) * 1000,
            3,
        )

        # A/B 모두 동일한 Local LLM과 동일한 생성 설정으로 응답 생성
        answer_result = await generate(
            prompt_spec,
            provider="local",
            config=config,
            llm=llm,
            tools=[],
        )

        # 프롬프트, 응답, 평가 기준을 하나의 실험 기록으로 구성
        records.append(
            {
                "sample_id": case["id"],
                "input_text": case["text"],
                "expected_label": case["expected_label"],
                "label_source": "generation_development.expected_label",
                "prompt_method": prompt_method,
                "prompt_structure": prompt_structure,
                "prompt_generation_ms": prompt_generation_ms,
                "prompt_spec": asdict(prompt_spec),
                "answer_result": asdict(answer_result),
                "required_points": case["required_points"],
                "forbidden_claims": case["forbidden_claims"],
                "rubric_note": case.get("rubric_note"),
            }
        )

    # 문의 1건에서 생성된 A/B 기록 2개 반환
    return records
    
    
def save_json(path: Path, data) -> None:
    """딕셔너리 또는 목록을 한글이 유지되는 JSON 파일로 저장한다."""

    # ensure_ascii=False: 한글을 유니코드 코드가 아닌 실제 한글로 저장
    # indent=2: 사람이 읽기 쉽도록 들여쓰기 적용
    json_text = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )

    # UTF-8 형식으로 지정한 경로에 JSON 저장
    path.write_text(
        json_text,
        encoding="utf-8",
    )
    
        
async def main() -> None:
    """개발 데이터 전체에 A/B 프롬프트를 적용하고 실험 결과를 저장한다."""
    
    # 터미널에서 지정한 실행 순서 읽기
    args = parse_args()
    
    # 기본 목록은 A → B 유지, ba 선택 시 실행 목록만 뒤집기
    if args.order == "ab":
        prompt_builders = PROMPT_BUILDERS
    else: 
        prompt_builders = list(reversed(PROMPT_BUILDERS))


    # .env와 팀 기본값에서 Local LLM 설정 읽기
    config = LLMConfig.from_env()

    # 모든 A/B 실험에서 같은 Local LLM 객체 사용
    llm = make_llm(
        "local",
        config,
        with_tools=False,
    )

    # 생성 프롬프트 개발 데이터 15건 읽기
    cases = load_development_cases(DATA_PATH)

    # 실행 시각을 기준으로 새로운 결과 폴더 생성
    run_id, run_dir = create_run_directory(RUNS_DIR, args.order)

    # 이번 실험에서 사용한 조건 기록
    experiment_settings = {
        "run_id": run_id,
        "created_at": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
        "experiment": "prompt_ab_comparison",
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "data_split": "generation_development",
        "number_of_cases": len(cases),
        "number_of_expected_responses": len(cases) * len(prompt_builders),
        "execution_order": args.order,
        "provider": "local",
        "model": config.ollama_model,
        "with_tools": False,
        "label_source": "generation_development.expected_label",
        "prompt_methods": [
            {
                "method": prompt_method,
                "structure": prompt_structure,
            }
            for prompt_method, prompt_structure, _ in prompt_builders
        ],
        "evaluation_data_sent_to_llm": False,
    }

    # 모델 호출 전에 실험 조건을 먼저 저장
    timestamp = run_id.removeprefix(
        f"prompt_ab_order_{args.order}_"
    )
    save_json(
        run_dir
        / f"prompt_ab_order_{args.order}_experiment_settings_{timestamp}.json",
        experiment_settings,
    )

    # 전체 문의의 A/B 응답 기록을 순서대로 저장
    records = []

    for case_number, case in enumerate(cases, start=1):
        # 문의 1건에서 A 응답과 B 응답 생성
        case_records = await generate_prompt_responses_for_case(
            case,
            llm,
            config,
            prompt_builders,
        )

        # 전체 결과 목록에 선택한 실행 순서대로 추가
        records.extend(case_records)

        # 현재 진행 상황을 터미널에 출력
        print(
            f"[{case_number}/{len(cases)}] "
            f"{case['id']} 응답 {len(case_records)}건 생성"
        )

    # 실제 프롬프트, 모델 응답, 평가 기준을 JSON으로 저장
    save_json(
        run_dir
        / f"prompt_ab_order_{args.order}_responses_{timestamp}.json",
        records,
    )

    # 저장 결과와 생성 개수를 터미널에 출력
    print(f"\n실험 실행 ID: {run_id}")
    print(f"개발 데이터 수: {len(cases)}")
    print(f"생성 응답 수: {len(records)}")
    print(f"결과 저장 경로: {run_dir}")
            
      
if __name__ == "__main__":
    # 비동기 main 함수를 이벤트 루프에서 실행
    asyncio.run(main())  
