import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# 콘솔 코드페이지가 한글을 못 받는 환경(cp1252 등)에서 print가 죽지 않도록
sys.stdout.reconfigure(errors="replace")

CANDIDATES = [
    "klue/bert-base",
    "klue/roberta-base",
    "monologg/koelectra-small-v3-discriminator",
]


def is_downloaded(local_dir: Path) -> bool:
    has_config = (local_dir / "config.json").exists()
    has_weights = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
    return has_config and has_weights


api = HfApi()
for repo_id in CANDIDATES:
    local_dir = Path("models") / repo_id.replace("/", "__")

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
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
    )
    print(f"  다운로드 완료 → {local_dir}")