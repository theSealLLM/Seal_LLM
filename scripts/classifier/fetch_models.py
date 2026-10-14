import sys
from pathlib import Path

import _common  # noqa: F401  src를 import 경로에 넣기 위해 필요
from huggingface_hub import HfApi, snapshot_download

from theseal.classifier.config import MODEL_CANDIDATES, local_model_dir

# 콘솔 코드페이지가 한글을 못 받는 환경(cp1252 등)에서 print가 죽지 않도록
sys.stdout.reconfigure(errors="replace")

# 후보 목록은 config.py의 MODEL_CANDIDATES 하나만 쓴다 (학습 스크립트 번호 메뉴와 같은 목록)
CANDIDATES = MODEL_CANDIDATES


def is_downloaded(local_dir: Path) -> bool:
    has_config = (local_dir / "config.json").exists()
    has_weights = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
    return has_config and has_weights


api = HfApi()
for repo_id in CANDIDATES:
    local_dir = local_model_dir(repo_id)

    if is_downloaded(local_dir):
        print(f"== {repo_id}: 이미 있음 ({local_dir}) → 건너뜀")
        continue

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
    print(f"  다운로드 완료 → {local_dir}")