"""E2E 입력 데이터와 결과 파일(JSONL · JSON) 읽기 · 쓰기.

입력은 CSV(data/endtoend/questions.csv)와 JSONL(data/inquiries/generation_*.jsonl)을 모두 받는다.
    필수: id, text
    선택: expected_label, category, source, required_points, forbidden_claims, rubric_note (JSONL에만 있음)
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from theseal.common.config import LABELS


def load_cases(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".csv":
        # data/README: CSV에는 UTF-8 BOM이 있어 utf-8-sig로 읽는다
        with path.open(encoding="utf-8-sig", newline="") as f:
            cases = [dict(row) for row in csv.DictReader(f)]
    elif path.suffix == ".jsonl":
        cases = read_jsonl(path)
    else:
        raise ValueError(f"CSV나 JSONL만 읽습니다: {path}")

    for case in cases:
        if not case.get("id") or not str(case.get("text") or "").strip():
            raise ValueError(f"id와 text가 있어야 합니다: {case}")
        case["text"] = case["text"].strip()
        label = (case.get("expected_label") or "").strip() or None
        if label is not None and label not in LABELS:
            raise ValueError(f"{case['id']}: 알 수 없는 expected_label {label!r} (가능: {', '.join(LABELS)})")
        case["expected_label"] = label

    duplicated = [i for i, n in Counter(c["id"] for c in cases).items() if n > 1]
    if duplicated:
        raise ValueError(f"id가 겹칩니다: {duplicated[:5]}")
    return cases


def select_cases(cases: list[dict], per_label: int | None = None, limit: int | None = None) -> list[dict]:
    """per_label: 정답 라벨마다 앞에서 N건 (동작 확인용). limit: 그다음 앞에서 N건."""
    if per_label:
        seen: Counter = Counter()
        picked = []
        for case in cases:
            if seen[case["expected_label"]] < per_label:
                picked.append(case)
                seen[case["expected_label"]] += 1
        cases = picked
    return cases[:limit] if limit else cases


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for n, line in enumerate(lines, 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if n == len(lines):  # 쓰는 도중 끊긴 마지막 줄은 버린다 (그 건은 다음 실행에서 다시 돈다)
                break
            raise
    return rows


def latest_rows(path: Path) -> dict[str, dict]:
    """id별 마지막 줄. 같은 id를 다시 돌리면 뒤에 덧붙이므로 마지막 줄이 최신이다."""
    return {row["id"]: row for row in read_jsonl(path)}


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_jsonl(path: Path, order: list[str]) -> None:
    """id마다 마지막 줄만 남기고 order 순서로 다시 쓴다 (다시 돌린 건이 쌓여 파일이 커지지 않게)."""
    rows = latest_rows(path)
    if not rows:
        return
    rank = {row_id: n for n, row_id in enumerate(order)}
    ordered = sorted(rows.values(), key=lambda r: rank.get(r["id"], len(rank)))
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
    tmp.replace(path)
