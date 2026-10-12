"""CSV/라벨 로드와 검증.

- CSV는 UTF-8 BOM이 있어 utf-8-sig로 읽는다.
- 모델 입력은 text 하나뿐이다. label과 관리용 컬럼(id, group_id, source)은 입력에 넣지 않는다.
- 라벨↔숫자 변환은 labels.json의 labels 순서를 학습과 예측에서 똑같이 쓴다.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_COLUMNS = ("id", "text", "label", "group_id", "source")


class DataError(ValueError):
    """데이터 파일 형식이나 내용이 기대와 다를 때."""


@dataclass(frozen=True)
class LabelSet:
    labels: tuple[str, ...]
    descriptions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels):
            raise DataError(f"라벨 목록에 중복이 있습니다: {self.labels}")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def label2id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))

    def encode(self, labels: list[str]) -> list[int]:
        mapping = self.label2id
        unknown = sorted({label for label in labels if label not in mapping})
        if unknown:
            raise DataError(f"labels.json에 없는 라벨: {unknown}")
        return [mapping[label] for label in labels]

    def decode(self, ids) -> list[str]:
        return [self.labels[int(i)] for i in ids]


@dataclass(frozen=True)
class Split:
    name: str
    ids: list[str]
    texts: list[str]
    labels: list[str]
    group_ids: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    def label_counts(self) -> dict[str, int]:
        return dict(Counter(self.labels))


def load_labels(data_dir: Path) -> LabelSet:
    path = Path(data_dir) / "labels.json"
    if not path.exists():
        raise DataError(f"라벨 파일이 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not raw.get("labels"):
        raise DataError(f"{path}에 labels 목록이 없습니다.")
    return LabelSet(labels=tuple(raw["labels"]), descriptions=dict(raw.get("descriptions", {})))


def load_split(data_dir: Path, split: str, label_set: LabelSet) -> Split:
    path = Path(data_dir) / f"{split}.csv"
    if not path.exists():
        raise DataError(f"데이터 파일이 없습니다: {path}")

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DataError(f"{path}에 필요한 컬럼이 없습니다: {missing}")
        rows = list(reader)

    if not rows:
        raise DataError(f"{path}가 비어 있습니다.")

    ids, texts, labels, group_ids = [], [], [], []
    for line_no, row in enumerate(rows, start=2):
        text = (row["text"] or "").strip()
        if not text:
            raise DataError(f"{path}:{line_no} text가 비어 있습니다 (id={row['id']}).")
        ids.append(row["id"])
        texts.append(text)
        labels.append(row["label"].strip())
        group_ids.append(row["group_id"])

    duplicated = [i for i, n in Counter(ids).items() if n > 1]
    if duplicated:
        raise DataError(f"{path}에 중복 id가 있습니다: {duplicated[:5]}")
    label_set.encode(labels)  # 모르는 라벨이 있으면 여기서 실패

    return Split(name=split, ids=ids, texts=texts, labels=labels, group_ids=group_ids)


def check_group_leakage(splits: list[Split]) -> None:
    """같은 group_id가 서로 다른 분할에 들어가 있으면 실패한다."""
    seen: dict[str, str] = {}
    for split in splits:
        for gid in set(split.group_ids):
            if gid in seen and seen[gid] != split.name:
                raise DataError(
                    f"group_id {gid!r}가 {seen[gid]}와 {split.name}에 함께 있습니다. "
                    "같은 상황 묶음은 한 분할에만 있어야 합니다."
                )
            seen[gid] = split.name


def load_task(data_dir: Path, splits: tuple[str, ...]) -> tuple[LabelSet, dict[str, Split]]:
    """라벨과 요청한 분할만 읽고 분할 간 group_id 누수를 확인한다.

    학습/비교 단계에서는 splits에 test를 넣지 않는다.
    """
    label_set = load_labels(data_dir)
    loaded = {name: load_split(data_dir, name, label_set) for name in splits}
    check_group_leakage(list(loaded.values()))
    if "train" in loaded:
        missing = [label for label in label_set.labels if label not in loaded["train"].labels]
        if missing:
            raise DataError(f"train에 한 번도 나오지 않는 라벨이 있습니다: {missing}")
    return label_set, loaded
