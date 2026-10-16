"""리포트: 항목(사실 여부 · 중요 정보 포함 · 정책 위반)별로 우리 서비스와 OpenAI의 평균 · 전체 점수를 비교한다.

    평균      문의 하나당 점수의 평균 (항목 만점: 사실 10 · 정보 10 · 정책 9)
    전체      비교한 문의 전부의 점수 합
    차이      우리 − OpenAI. 0에 가까울수록 우리 서비스가 OpenAI만큼 답한다는 뜻 (양수면 우리가 더 높음)
두 답변이 모두 채점된 문의만 비교한다. 라벨 정확도 · 안내 문구 비율 · 지연은 채점과 무관하므로 서비스 답변 전체로 센다.
"""

from __future__ import annotations

import statistics
from collections import Counter

from theseal.common.config import LABELS

from .rubric import CATEGORIES, INDICATORS, MAX_SCORE

ALL = "total"  # 세 항목 합계
WORST_CASES = 10
MODULES = ("classifier", "routing", "prompt/answer", "answer")


def pct(n: float, d: float) -> float | None:
    return round(n / d * 100, 1) if d else None


def pairs_of(cases: list[dict], rows: dict, refs: dict) -> list[dict]:
    """두 답변이 모두 채점된 문의: {case, ours, openai, module}. rows: results.jsonl {id: 줄}"""
    return [{"case": c, "ours": rows[c["id"]]["score"], "openai": refs[c["id"]]["score"],
             "module": rows[c["id"]]["module"]}
            for c in cases if c["id"] in rows and rows[c["id"]].get("score") and c["id"] in refs]


def points(score: dict, cat: str) -> int:
    return score["total"] if cat == ALL else score["points"][cat]


def category_scores(pairs: list[dict]) -> dict:
    """{항목: {max, ours_mean, openai_mean, gap_mean, ours_total, openai_total, gap_total, ratio}}"""
    n = len(pairs)
    out = {}
    for cat in (*CATEGORIES, ALL):
        ours = sum(points(p["ours"], cat) for p in pairs)
        openai = sum(points(p["openai"], cat) for p in pairs)
        out[cat] = {
            "max": sum(MAX_SCORE.values()) if cat == ALL else MAX_SCORE[cat],
            "ours_mean": round(ours / n, 2) if n else None,
            "openai_mean": round(openai / n, 2) if n else None,
            "gap_mean": round((ours - openai) / n, 2) if n else None,
            "ours_total": ours,
            "openai_total": openai,
            "gap_total": ours - openai,
            "ratio": pct(ours, openai),  # OpenAI 전체 점수 대비 우리 전체 점수 (%)
        }
    return out


def indicator_scores(pairs: list[dict]) -> list[dict]:
    n = len(pairs)
    rows = []
    for k, ind in enumerate(INDICATORS):
        ours = [p["ours"]["verdicts"][k] for p in pairs]
        openai = [p["openai"]["verdicts"][k] for p in pairs]
        rows.append({
            "id": ind.id, "category": ind.category, "name": ind.name,
            "ours_rate": pct(sum(v["point"] for v in ours), n),
            "openai_rate": pct(sum(v["point"] for v in openai), n),
            "gap": sum(v["point"] for v in ours) - sum(v["point"] for v in openai),
            "ours_applicable": sum(v["applicable"] for v in ours),
            "openai_applicable": sum(v["applicable"] for v in openai),
        })
    return rows


def service_metrics(services: list[dict]) -> dict:
    labeled = [s for s in services if s["trace"]["expected_label"]]
    latencies = sorted(s["e2e_latency_ms"] for s in services)
    return {
        "n_tested": len(services),
        "n_service_error": sum(s["status"] != "ok" for s in services),
        "label_accuracy": pct(sum(s["trace"]["final_label"] == s["trace"]["expected_label"] for s in labeled),
                              len(labeled)),
        "fallback_rate": pct(sum(s["trace"]["fallback"] for s in services), len(services)),
        "latency_ms_median": round(statistics.median(latencies)) if latencies else None,
        "latency_ms_p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
        # 정답 라벨 → 최종 라벨 (분류기 + 분류 후처리 결과)
        "routing": {label: dict(Counter(s["trace"]["final_label"] for s in labeled
                                        if s["trace"]["expected_label"] == label))
                    for label in LABELS if any(s["trace"]["expected_label"] == label for s in labeled)},
    }


def summarize(cases: list[dict], rows: dict, refs: dict) -> dict:
    """rows: results.jsonl {id: 줄}, refs: 검증용 데이터 {id: 줄}"""
    pairs = pairs_of(cases, rows, refs)
    labels = [label for label in LABELS if any(p["case"]["expected_label"] == label for p in pairs)]
    lower = [p for p in pairs if p["ours"]["total"] < p["openai"]["total"]]
    return {
        "n_compared": len(pairs),
        "categories": category_scores(pairs),
        "by_label": {label: {"n": sum(p["case"]["expected_label"] == label for p in pairs),
                             "categories": category_scores([p for p in pairs if p["case"]["expected_label"] == label])}
                     for label in labels},
        "indicators": indicator_scores(pairs),
        "service": service_metrics([rows[c["id"]] for c in cases if c["id"] in rows]),
        # OpenAI보다 점수가 낮은 문의를 어느 모듈부터 볼지
        "lower_by_module": {m: sum(p["module"] == m for p in lower) for m in MODULES},
        "n_lower": len(lower),
        "n_higher": sum(p["ours"]["total"] > p["openai"]["total"] for p in pairs),
        "unanswerable_ids": [p["case"]["id"] for p in pairs if not p["openai"]["answerable"]],
    }


# ---------------------------------------------------------------- report.md

def _num(value) -> str:
    return "-" if value is None else f"{value:g}"


def _gap(value) -> str:
    return "-" if value is None else f"{value:+g}"


def _clip(text: str | None, n: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "…"


CATEGORY_HEADER = (
    "| 항목 | 만점(건당) | 우리 평균 | OpenAI 평균 | 평균 차이 | 우리 전체 | OpenAI 전체 | 전체 차이 | OpenAI 대비 |",
    "|---|---|---|---|---|---|---|---|---|",
)


def _category_rows(cats: dict) -> list[str]:
    return [f"| {'합계' if cat == ALL else CATEGORIES[cat]} | {v['max']} | {_num(v['ours_mean'])} | "
            f"{_num(v['openai_mean'])} | {_gap(v['gap_mean'])} | {v['ours_total']} | {v['openai_total']} | "
            f"{_gap(v['gap_total'])} | {_num(v['ratio'])}% |" for cat, v in cats.items()]


def report_md(report: dict, cases: list[dict], rows: dict, refs: dict) -> str:
    meta, s = report["meta"], report["summary"]
    sv = s["service"]
    same = s["n_compared"] - s["n_lower"] - s["n_higher"]
    lines = [
        f"# E2E 평가 리포트 · {meta['run_id']}",
        "",
        f"- 데이터: `{meta['data']}` ({meta['n_cases']}건 중 두 답변을 모두 채점한 {s['n_compared']}건 비교)",
        (f"- 우리 서비스: {meta['service']['provider']} `{meta['service']['model']}` · 분류 모델 "
         f"`{meta['service']['classifier']}` · 경로 규칙 {meta['service']['route_rule_enabled']} · "
         f"LLM 검증 {meta['service']['verify_mode']}"),
        f"- OpenAI 답변: `{meta['reference']['model']}` (정답 라벨의 서비스 프롬프트) · `{meta['reference']['path']}`",
        f"- 채점: `{meta['judge']['model']}` ({meta['judge']['version']}) · 지표마다 만족 1점, 해당 없는 지표도 1점",
        "- 차이 = 우리 − OpenAI. 0에 가까울수록 좋다 (양수면 우리가 더 높음)",
        "",
        "## 항목별 점수",
        "",
        *CATEGORY_HEADER,
        *_category_rows(s["categories"]),
        "",
        f"OpenAI보다 낮은 문의 {s['n_lower']}건 · 같은 문의 {same}건 · 높은 문의 {s['n_higher']}건",
        "",
        "## 라벨별 점수",
    ]
    for label, v in s["by_label"].items():
        lines += ["", f"### {label} ({v['n']}건)", "", *CATEGORY_HEADER, *_category_rows(v["categories"])]

    lines += [
        "",
        "## 상세 분석",
        "",
        "### 지표별 만족률",
        "",
        "| 지표 | 항목 | 우리 | OpenAI | 차이(점) | 해당 건수 (우리 / OpenAI) |",
        "|---|---|---|---|---|---|",
        *[f"| {r['id']} {r['name']} | {CATEGORIES[r['category']]} | {_num(r['ours_rate'])}% | "
          f"{_num(r['openai_rate'])}% | {_gap(r['gap'])} | {r['ours_applicable']} / {r['openai_applicable']} |"
          for r in s["indicators"]],
        "",
        "### 분류 · 라우팅",
        "",
        (f"- 라벨 정확도 {_num(sv['label_accuracy'])}% · 안내 문구(fallback) 비율 {_num(sv['fallback_rate'])}% · "
         f"서비스 오류 {sv['n_service_error']}건 · 지연 중앙값 {sv['latency_ms_median']}ms "
         f"(p95 {sv['latency_ms_p95']}ms)"),
        "",
        "| 정답 \\ 최종 | " + " | ".join(LABELS) + " |",
        "|---" * (len(LABELS) + 1) + "|",
        *[f"| {gold} | " + " | ".join(str(counts.get(label, 0)) for label in LABELS) + " |"
          for gold, counts in sv["routing"].items()],
        "",
        "### OpenAI보다 낮은 문의를 먼저 볼 모듈",
        "",
        "| 모듈 | 문의 수 |",
        "|---|---|",
        *[f"| {m} | {n} |" for m, n in s["lower_by_module"].items()],
    ]

    pairs = sorted(pairs_of(cases, rows, refs), key=lambda p: p["ours"]["total"] - p["openai"]["total"])
    worst = [p for p in pairs if p["ours"]["total"] < p["openai"]["total"]][:WORST_CASES]
    if worst:
        lines += ["", f"### 차이가 큰 문의 (상위 {len(worst)}건)"]
        for p in worst:
            c, row = p["case"], rows[p["case"]["id"]]
            lines += [
                "",
                (f"#### {c['id']} · 우리 {p['ours']['total']}점 / OpenAI {p['openai']['total']}점 · "
                 f"{c['expected_label']} → {row['trace']['final_label']} · {p['module']}"),
                "",
                f"- 문의: {c['text']}",
                f"- 우리: {_clip(row['answer'], 240)}",
                f"- OpenAI: {_clip(refs[c['id']]['answer'], 240)}",
            ]
            for ours, openai in zip(p["ours"]["verdicts"], p["openai"]["verdicts"], strict=True):
                if ours["point"] < openai["point"]:
                    quote = f" — “{_clip(ours['evidence'])}”" if ours["evidence"] else ""
                    lines.append(f"  - {ours['id']} 0점: {_clip(ours['reason'])}{quote}")

    lines += [
        "",
        "### 점검",
        "",
        (f"- 문의만으로 뜻을 알 수 없다고 채점된 문의 (OpenAI 답변 채점 기준): {len(s['unanswerable_ids'])}건 "
         f"{', '.join(s['unanswerable_ids'][:20])}"),
        f"- 검증용 데이터가 없어 비교하지 못한 문의: {len(report['no_reference_ids'])}건",
        f"- 실패해 빠진 문의 (다시 실행하면 채워짐): {len(report['missing_ids'])}건",
        "",
    ]
    return "\n".join(lines)
