"""팀원용 예제: 학습해 둔 분류기를 불러와 문장을 분류한다.

실행
    uv run python scripts/classifier/classifier_example.py
    uv run python scripts/classifier/classifier_example.py "주문한 물건이 아직 안 왔어요"
    uv run python scripts/classifier/classifier_example.py --model-dir runs/classifier/inquiries/baseline/model

준비물 (모델 파일은 git에 없으므로 둘 중 하나가 필요하다)
    - 직접 학습: train_*.py → compare.py 를 돌리면 runs/classifier/<task>/final_model/ 이 생긴다.
    - 받아오기:  uv run python scripts/classifier/share_model.py pull --final --repo-id <계정>/<저장소>

내 코드에서 쓸 때는 아래 세 줄이면 된다 (src가 import 경로에 있어야 함).
    from theseal.classifier.classifier import Classifier
    clf = Classifier.from_final()                  # 확정 모델, 기본 task는 inquiries
    clf.predict("문장")                            # {"label", "confidence", "scores"}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401  src를 import 경로에 넣기 위해 필요

from theseal.classifier.classifier import Classifier
from theseal.classifier.config import DEFAULT_TASK, TASKS, display_path

# 예제 문장 (학습/평가 데이터에 없는 문장)
SAMPLES = {
    "inquiries": [
        "택배 조회가 사흘째 같은 지점에 멈춰 있어요. 언제 받을 수 있을까요?",
        "사이즈가 맞지 않아서 반품하고 싶은데 환불은 언제 들어오나요?",
        "휴대폰 번호를 바꿔서 로그인 인증 문자를 받을 수가 없어요.",
    ],
    "documents": [
        "다음 주 월요일부터 3층 회의실은 공사로 사용할 수 없습니다. 2층 회의실을 이용해 주세요.",
        "1. 전원 버튼을 3초간 누른다. 2. 초록 불이 켜지면 시작 버튼을 누른다.",
        "참석자들은 행사 날짜를 다음 달로 미루기로 했다. 장소 섭외는 김 담당이 맡는다.",
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="분류기 사용 예제")
    p.add_argument("texts", nargs="*", help="분류할 문장 (생략하면 예제 문장 사용)")
    p.add_argument("--task", default=DEFAULT_TASK, choices=TASKS,
                   help=f"확정 모델을 고를 주제 (기본: {DEFAULT_TASK})")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="확정 모델 대신 쓸 모델 폴더 (예: runs/classifier/inquiries/baseline/model)")
    p.add_argument("--device", default=None, help="사전학습 모델용 장치 (생략하면 자동: cuda > mps > cpu)")
    return p.parse_args(argv)


def load(args: argparse.Namespace) -> Classifier:
    try:
        # 1) 특정 모델 폴더를 지정한 경우
        if args.model_dir:
            return Classifier(args.model_dir, device=args.device)
        # 2) compare.py가 확정한 모델 (runs/classifier/<task>/final_model/)
        return Classifier.from_final(args.task, device=args.device)
    except FileNotFoundError as e:
        raise SystemExit(
            f"[중단] 모델을 찾지 못했습니다: {e}\n"
            "compare.py로 모델을 확정하거나, share_model.py pull --final 로 받아 오세요."
        ) from None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    clf = load(args)
    print(f"[모델] {display_path(clf.model_dir)} "
          f"(kind={clf.meta['kind']}, name={clf.meta['name']}) labels={clf.labels}\n")

    texts = args.texts or SAMPLES[clf.meta.get("task", args.task)]

    # 문장 하나: predict(str) → dict
    first = clf.predict(texts[0])
    print(f"predict(): {texts[0]}")
    print(f"  → {first['label']} (confidence {first['confidence']:.3f})")
    print(f"  scores: { {k: round(v, 3) for k, v in first['scores'].items()} }\n")

    # 여러 문장: predict_batch(list[str]) → list[dict]
    print("predict_batch():")
    for text, result in zip(texts, clf.predict_batch(texts)):
        print(f"  {result['label']:<14} {result['confidence']:.3f}  {text}")


if __name__ == "__main__":
    main()
