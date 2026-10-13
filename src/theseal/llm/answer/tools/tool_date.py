"""날짜 · 요일 계산 툴. 로컬 LLM은 오늘 날짜를 모르므로 파이썬이 계산한다.

- 날짜 형식은 YYYY-MM-DD, 시간대는 설정의 TIMEZONE (기본 Asia/Seoul)
- 잘못된 날짜는 ValueError를 던지고 tool_caller가 잡아서 기록한다
- 테스트에서 기준일을 고정할 수 있게 "오늘"은 today() 하나로만 얻는다
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from ...config import get_config

WEEKDAYS_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


def today() -> date:
    return datetime.now(ZoneInfo(get_config().timezone)).date()


def _parse(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"날짜 형식이 올바르지 않습니다: {value!r} (YYYY-MM-DD)") from None


def _weekday(d: date) -> str:
    return WEEKDAYS_KO[d.weekday()]


@tool
def get_today() -> dict:
    """오늘 날짜와 요일을 알려 준다. 오늘 날짜가 필요하면 추측하지 말고 반드시 이 도구를 쓴다."""
    d = today()
    return {"date": d.isoformat(), "weekday": _weekday(d)}


@tool
def add_days(date: str, days: int) -> dict:
    """날짜(YYYY-MM-DD)에 일수를 더한 날짜와 요일을 알려 준다. days가 음수면 이전 날짜."""
    d = _parse(date) + timedelta(days=days)
    return {"date": d.isoformat(), "weekday": _weekday(d)}


@tool
def days_between(start: str, end: str) -> dict:
    """두 날짜(YYYY-MM-DD) 사이의 일수(end - start)를 알려 준다."""
    return {"days": (_parse(end) - _parse(start)).days}


@tool
def get_weekday(date: str) -> dict:
    """날짜(YYYY-MM-DD)의 요일을 알려 준다."""
    return {"weekday": _weekday(_parse(date))}


@tool
def get_date_of_weekday(weekday: str, week_offset: int = 0) -> dict:
    """'이번 주 금요일', '다음 주 월요일', '지난주 일요일'처럼 주와 요일로 말한 날짜를 알려 준다.
    weekday는 '월요일'~'일요일'(또는 '월'~'일'). week_offset은 이번 주 0, 다음 주 1, 다다음 주 2, 지난주 -1.
    한 주는 월요일에 시작한다."""
    name = weekday.strip()
    if not name.endswith("요일"):
        name += "요일"
    if name not in WEEKDAYS_KO:
        raise ValueError(f"요일을 알 수 없습니다: {weekday!r} (월요일~일요일)")
    t = today()
    monday = t - timedelta(days=t.weekday())
    d = monday + timedelta(weeks=week_offset, days=WEEKDAYS_KO.index(name))
    return {"date": d.isoformat(), "weekday": _weekday(d)}


DATE_TOOLS = [get_today, add_days, days_between, get_weekday, get_date_of_weekday]
