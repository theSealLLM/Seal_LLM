"""② 경로 규칙. 분류기가 준 라벨별 확률에서 두 숫자만 보고 경로를 정한다.

    p1     = 최대 확률 (얼마나 자신 있나)
    margin = p1 - p2 (1등이 2등을 얼마나 따돌렸나)

기준선
    tau      확신       p1이 이 값 이상이면 확신
    tau_low  아예 모름  p1이 이 값 미만이면 unknown
    delta    접전       margin이 이 값 미만이면 접전

판정 순서
    1. p1 < tau_low                 → unknown         (아예 모름)
    2. margin < delta               → ambiguous       (접전: 1·2등이 비슷)
    3. p1 >= tau                    → confident       (확신)
    4. 나머지 (tau_low <= p1 < tau) → low_confidence  (확신 부족이지만 1등이 앞섬)

확률 보정 (temperature scaling, 모델 재학습 없음)
    확률 = softmax(원점수 / T). 파인튜닝 모델은 범위 밖 글에도 0.99 이상을 주므로 T > 1로 펴고,
    TF-IDF처럼 확률이 0.33~0.5에 몰린 모델은 T < 1로 벌린다.
    분류기(Classifier)는 원점수를 내지 않으므로 log(확률)을 원점수로 쓴다. softmax는 모든 원점수에
    같은 값을 더해도 결과가 같아서 log(확률)로 계산해도 보정 결과가 같다.

LLM을 부르지 않고 분류기 종류도 모른다. 라벨별 확률(scores) dict만 받는다.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

ROUTES = ("confident", "low_confidence", "ambiguous", "unknown")
REASONS = {
    "unknown": "p1 < tau_low (아예 모름)",
    "ambiguous": "margin < delta (1·2등 접전)",
    "confident": "p1 >= tau, margin >= delta (확신)",
    "low_confidence": "tau_low <= p1 < tau, margin >= delta (확신 부족)",
}
MIN_PROB = 1e-12  # log(0)을 피하기 위한 하한


class RouteDecision(BaseModel):
    route: str  # confident | low_confidence | ambiguous | unknown
    label: str  # 최종 라벨 (shipping · refund · account · unknown)
    top1: str
    top2: str
    p1: float  # 보정 후 확률
    p2: float
    margin: float
    reason: str
    params: dict  # 실제로 쓴 temperature · tau · tau_low · delta · uncertain_action


def logits_from_scores(scores: dict[str, float]) -> dict[str, float]:
    """분류기 확률(scores)을 원점수처럼 쓸 수 있게 log를 씌운다."""
    return {k: math.log(max(p, MIN_PROB)) for k, p in scores.items()}


def softmax(logits: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    z = {k: v / temperature for k, v in logits.items()}
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    total = sum(e.values())
    return {k: v / total for k, v in e.items()}


def calibrate(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """확률을 T로 보정한다 (T=1이면 그대로)."""
    if temperature == 1.0:
        return dict(scores)
    return softmax(logits_from_scores(scores), temperature)


def decide(
    scores: dict[str, float],
    *,
    temperature: float = 1.0,
    tau: float = 0.60,
    tau_low: float = 0.45,
    delta: float = 0.15,
    uncertain_action: str = "top1",
) -> RouteDecision:
    """scores: 라벨별 확률 {라벨 이름: 확률} (router.to_scores로 Classifier.predict 결과를 바꾼 것).
    uncertain_action: ambiguous · low_confidence일 때 최종 라벨. top1이면 1등 라벨, unknown이면 unknown"""
    if len(scores) < 2:
        raise ValueError(f"라벨이 2개 이상이어야 합니다: {list(scores)}")
    if not tau_low <= tau:
        raise ValueError(f"tau_low({tau_low})는 tau({tau}) 이하여야 합니다")
    probs = calibrate(scores, temperature)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    (top1, p1), (top2, p2) = ranked[0], ranked[1]
    margin = p1 - p2

    if p1 < tau_low:
        route = "unknown"
    elif margin < delta:
        route = "ambiguous"
    elif p1 >= tau:
        route = "confident"
    else:
        route = "low_confidence"

    if route == "confident":
        label = top1
    elif route == "unknown":
        label = "unknown"
    else:
        label = top1 if uncertain_action == "top1" else "unknown"

    return RouteDecision(
        route=route,
        label=label,
        top1=top1,
        top2=top2,
        p1=round(p1, 4),
        p2=round(p2, 4),
        margin=round(margin, 4),
        reason=REASONS[route],
        params={"temperature": temperature, "tau": tau, "tau_low": tau_low, "delta": delta,
                "uncertain_action": uncertain_action},
    )
