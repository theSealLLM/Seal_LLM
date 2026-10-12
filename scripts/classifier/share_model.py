"""⑤ 학습한 모델 폴더를 HF Hub 프라이빗 저장소로 올리고 내려받는다.

가중치(model.safetensors)는 수백 MB라 git에 올릴 수 없다(GitHub 파일 한도 100MB).
팀원이 직접 학습하지 않고 파인튜닝 결과를 그대로 쓰려면 이 스크립트로 주고받는다.

올리기: uv run python scripts/classifier/share_model.py push --name koelectra-small-v3-discriminator --repo-id <계정>/<저장소>
받기:   uv run python scripts/classifier/share_model.py pull --name koelectra-small-v3-discriminator --repo-id <계정>/<저장소>

확정 모델(final_model/)은 --name 대신 --final을 쓴다.
저장소 하나에 task별/이름별로 폴더를 나눠 담으므로 모델마다 저장소를 만들 필요는 없다.
토큰은 HF_TOKEN 환경변수나 `uv run hf auth login`으로 준비한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common
from huggingface_hub import HfApi, snapshot_download

from theseal.classifier.config import FINAL_NAME, META_FILENAME, display_path
from theseal.classifier.predictor import read_meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="학습한 모델을 HF Hub로 올리고 내려받기")
    p.add_argument("command", choices=["push", "pull"], help="push: 올리기, pull: 받기")
    _common.add_common_args(p)
    p.add_argument("--repo-id", required=True, help="HF Hub 저장소 (<계정 또는 조직>/<이름>)")
    p.add_argument("--name", default=None, help="runs/classifier/<task>/<이름> 의 이름")
    p.add_argument("--final", action="store_true", help=f"--name 대신 확정 모델({FINAL_NAME}/)을 다룬다")
    p.add_argument("--public", action="store_true", help="push할 때 공개 저장소로 만든다 (기본: 비공개)")
    return p.parse_args(argv)


def resolve(paths, args) -> tuple[Path, str, Path]:
    """(로컬 모델 폴더, 저장소 안 경로, 받을 때 기준 폴더)를 정한다.

    저장소 안 경로를 <task>/... 로 잡아 두면, 기준 폴더(runs/classifier)에 그대로 풀었을 때
    로컬 폴더 구조와 맞아떨어진다.
    """
    base_dir = paths.runs_dir.parent  # runs/classifier
    if args.final:
        return paths.final_dir, f"{args.task}/{FINAL_NAME}", base_dir
    if not args.name:
        raise SystemExit("[중단] --name 또는 --final 중 하나는 있어야 합니다.")
    return paths.model_dir(args.name), f"{args.task}/{args.name}/model", base_dir


def push(api: HfApi, args, local_dir: Path, path_in_repo: str) -> None:
    if not (local_dir / META_FILENAME).exists():
        raise SystemExit(f"[중단] 모델 폴더가 아닙니다: {display_path(local_dir)} "
                         f"({META_FILENAME}이 없습니다). 먼저 학습하세요.")
    meta = read_meta(local_dir)
    size_mb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"[올리기] {display_path(local_dir)} ({size_mb:,.1f}MB) "
          f"→ {args.repo_id}/{path_in_repo}")
    print(f"  kind={meta['kind']} task={meta['task']} labels={meta['labels']}")

    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=args.repo_id,
        path_in_repo=path_in_repo,
        repo_type="model",
        commit_message=f"{args.task}/{meta['name']} ({meta['kind']})",
    )
    print(f"완료: https://huggingface.co/{args.repo_id}/tree/main/{path_in_repo}")


def pull(args, local_dir: Path, path_in_repo: str, base_dir: Path) -> None:
    if (local_dir / META_FILENAME).exists():
        print(f"[덮어씀] 이미 모델이 있습니다: {display_path(local_dir)}")
    print(f"[받기] {args.repo_id}/{path_in_repo} → {display_path(local_dir)}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        allow_patterns=[f"{path_in_repo}/*"],
        local_dir=str(base_dir),
    )
    if not (local_dir / META_FILENAME).exists():
        raise SystemExit(f"[중단] 받은 뒤에도 {META_FILENAME}이 없습니다. "
                         f"--name과 --task가 올린 쪽과 같은지 확인하세요.")
    meta = read_meta(local_dir)
    print(f"완료: kind={meta['kind']} task={meta['task']} labels={meta['labels']}")
    print(f"불러오기: Predictor({display_path(local_dir)!r})")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = _common.paths_from_args(args)
    local_dir, path_in_repo, base_dir = resolve(paths, args)

    if args.command == "push":
        push(HfApi(), args, local_dir, path_in_repo)
    else:
        pull(args, local_dir, path_in_repo, base_dir)


if __name__ == "__main__":
    main()
