"""② 경로 규칙: 기준선과 비교 규칙만 정의한다. p1 · margin 계산은 router.py의 Router가 한다.

    p1     = 최대 확률 (얼마나 자신 있나)
    margin = p1 - p2 (1등이 2등을 얼마나 따돌렸나)

기준선
    tau      확신       p1이 이 값 이상이면 확신
    tau_low  아예 모름  p1이 이 값 미만이면 unknown
    delta    접전       margin이 이 값 미만이면 확신이 아님

판정 순서
    1. p1 < tau_low                 → unknown    (아예 모름)
    2. p1 >= tau, margin >= delta   → confident  (확신)
    3. 나머지                       → ambiguous  (확신 아님: 1·2등 접전이거나 p1 < tau)
"""

from __future__ import annotations

from dataclasses import dataclass

CONFIDENT = "confident"
AMBIGUOUS = "ambiguous"
UNKNOWN = "unknown"
REASONS = {
    UNKNOWN: "p1 < tau_low (아예 모름)",
    CONFIDENT: "p1 >= tau, margin >= delta (확신)",
    AMBIGUOUS: "margin < delta 또는 p1 < tau (확신 아님)",
}


@dataclass(frozen=True)
class RouteRule:
    """경로 규칙 기준선 세 개. 서비스가 쓰는 값은 router.py의 RULE (모든 모델에 고정)."""

    tau: float
    """확신: p1이 이 값 이상 (margin도 delta 이상)"""

    tau_low: float
    """아예 모름: p1이 이 값 미만이면 unknown"""

    delta: float
    """접전: margin이 이 값 미만이면 확신이 아님"""

    def __post_init__(self) -> None:
        if not self.tau_low <= self.tau:
            raise ValueError(f"tau_low({self.tau_low})는 tau({self.tau}) 이하여야 합니다")

    def route(self, p1: float, margin: float) -> str:
        """p1 · margin을 기준선과 비교해 경로(confident · ambiguous · unknown)를 고른다."""
        if p1 < self.tau_low:
            return UNKNOWN
        if p1 >= self.tau and margin >= self.delta:
            return CONFIDENT
        return AMBIGUOUS
