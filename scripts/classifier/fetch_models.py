"""분류기 후보 모델(config.py의 MODEL_CANDIDATES)을 번호로 골라 src/theseal/classifier/models/ 아래로 내려받는다.

uv run scripts/classifier/fetch_models.py

실행하면 후보 목록이 번호로 나오고, 받을 번호를 반드시 골라야 내려받는다 (여러 개는 쉼표, 전부는 a).
이미 받은 모델은 건너뛴다. 다시 받으려면 해당 폴더를 지우고 실행한다.
"""

import sys
from pathlib import Path

import _common  # noqa: F401  src를 import 경로에 넣기 위해 필요
from huggingface_hub import HfApi, snapshot_download

from theseal.classifier.config import MODEL_CANDIDATES, MODELS_DIR, display_path, local_model_dir

# 콘솔 코드페이지가 한글을 못 받는 환경(cp1252 등)에서 print가 죽지 않도록
sys.stdout.reconfigure(errors="replace")

# 후보 목록은 config.py의 MODEL_CANDIDATES 하나만 쓴다 (학습 스크립트 번호 메뉴와 같은 목록)
CANDIDATES = MODEL_CANDIDATES


def is_downloaded(local_dir: Path) -> bool:
    has_config = (local_dir / "config.json").exists()
    has_weights = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
    return has_config and has_weights


def choose_models() -> list[str]:
    """후보를 번호로 나열하고, 받을 모델을 고를 때까지 묻는다 (기본값 없음)."""
    print(f"\n내려받을 분류기 후보 모델을 고르세요. 저장 위치: {display_path(MODELS_DIR)}/")
    for i, repo_id in enumerate(CANDIDATES, start=1):
        done = "받음" if is_downloaded(local_model_dir(repo_id)) else ""
        print(f"  {i}) {repo_id:<45} {done}")
    print("  a) 전부")

    while True:
        try:
            raw = input("번호 (여러 개는 쉼표로, 예: 1,3): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n[중단] 모델을 고르지 않았습니다.") from None
        if raw in ("a", "all"):
            return list(CANDIDATES)
        picks = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
        if picks and all(p.isdigit() and 1 <= int(p) <= len(CANDIDATES) for p in picks):
            return [CANDIDATES[int(p) - 1] for p in dict.fromkeys(picks)]
        print(f"  1~{len(CANDIDATES)} 사이 번호나 a를 입력하세요.")


def download(api: HfApi, repo_id: str) -> None:
    local_dir = local_model_dir(repo_id)
    if is_downloaded(local_dir):
        print(f"== {repo_id}: 이미 있음 ({display_path(local_dir)}) → 건너뜀")
        return

    info = api.model_info(repo_id)
    license_ = info.card_data.get("license") if info.card_data else None

    print(f"== {repo_id}")
    print("  출처    :", f"https://huggingface.co/{repo_id}")
    print("  버전(sha):", info.sha)
    print("  라이선스 :", license_)
    print("  수정일   :", info.last_modified)

    snapshot_download(
        repo_id=repo_id,
        revision=info.sha,
        local_dir=str(local_dir),
        # *.py: 저장소 코드 모델(NeoBERT 등)의 model.py, *.model: sentencepiece 토크나이저
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "*.py", "*.model"],
    )
    print(f"  다운로드 완료 → {display_path(local_dir)}")


def main() -> None:
    api = HfApi()
    for repo_id in choose_models():
        download(api, repo_id)


if __name__ == "__main__":
    main()
