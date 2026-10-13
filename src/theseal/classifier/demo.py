"""분류기 체험용 예제: 문장을 입력하면 최종 분류 모델의 결과를 보여 준다.

    uv run src/theseal/classifier/demo.py

'질문하세요:'가 나오면 문장을 입력한다. q, exit 또는 Ctrl+C로 끝낸다.
모델은 config.py의 FINAL_MODEL(기본 task: inquiries)이 가리키는 것을 쓴다.
가중치는 git에 없으므로 학습한 컴퓨터에서 실행하거나 share_model.py pull --final로 받아 온다.

출력은 Classifier.predict의 반환값이다.
    label  : int          라벨 번호 (labels.json 순서)
    scores : list[float]  라벨별 확률 (같은 순서, 합은 1)
"""

from __future__ import annotations

import sys
from pathlib import Path

# 이 파일을 직접 실행하면 이 폴더(src/theseal/classifier)가 import 경로 맨 앞에 들어가
# config.py 같은 파일이 최상위 모듈로 잡힐 수 있다. 이 폴더는 빼고 src를 넣는다.
HERE = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == HERE:
    sys.path.pop(0)
SRC_DIR = HERE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from theseal.classifier.classifier import Classifier
from theseal.classifier.config import FinalModelNotSetError, display_path

QUIT_WORDS = {"q", "quit", "exit", "종료"}


def load() -> Classifier:
    print("[모델 불러오는 중] ...")
    try:
        return Classifier.from_final()
    except FinalModelNotSetError as e:
        raise SystemExit(f"[중단] {e}") from None
    except FileNotFoundError as e:
        raise SystemExit(
            f"[중단] 모델을 찾지 못했습니다: {e}\n"
            "학습한 컴퓨터에서 실행하거나 share_model.py pull --final 로 받아 오세요."
        ) from None


def main() -> None:
    clf = load()
    print(f"[모델] {display_path(clf.model_dir)} ({clf.meta['kind']})")
    print("[라벨] " + ", ".join(f"{i}={name}" for i, name in enumerate(clf.labels)))
    print("q, exit 또는 Ctrl+C로 끝냅니다.\n")

    while True:
        try:
            text = input("질문하세요: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in QUIT_WORDS:
            break

        label, scores = clf.predict(text)
        print(f"label  = {label} ({clf.labels[label]})")
        print(f"scores = {[round(s, 5) for s in scores]}\n")

    print("종료합니다.")


if __name__ == "__main__":
    main()
