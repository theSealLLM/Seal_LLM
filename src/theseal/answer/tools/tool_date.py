"""날짜 · 요일 계산 툴. 로컬 LLM은 오늘 날짜를 모르므로 파이썬이 계산한다.

- 날짜 형식은 YYYY-MM-DD, 시간대는 common.config의 TIMEZONE (Asia/Seoul)
- 잘못된 날짜는 ValueError를 던지고 tool_caller가 잡아서 기록한다
- "오늘"이 필요 없는 툴(add_days · days_between · get_weekday)은 모듈 레벨 @tool,
  "오늘"이 필요한 툴(get_today · get_date_of_weekday)은 시계를 들고 있는 DateTools가 만든다.
  테스트는 DateTools(today=…)로 기준일을 고정한다
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import date, datetime, timedelta

from langchain_core.tools import BaseTool, StructuredTool, tool

from theseal.common.config import TIMEZONE

WEEKDAYS_KO = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")

# LLM에 보내는 툴 설명. 예전 @tool 함수의 docstring 그대로다 (줄 앞 공백 4칸까지 같아야 LLM 입력이 바뀌지 않는다)
GET_TODAY_DESCRIPTION = "오늘 날짜와 요일을 알려 준다. 오늘 날짜가 필요하면 추측하지 말고 반드시 이 도구를 쓴다."
GET_DATE_OF_WEEKDAY_DESCRIPTION = (
    "'이번 주 금요일', '다음 주 월요일', '지난주 일요일'처럼 주와 요일로 말한 날짜를 알려 준다.\n"
    "    weekday는 '월요일'~'일요일'(또는 '월'~'일'). week_offset은 이번 주 0, 다음 주 1, 다다음 주 2, 지난주 -1.\n"
    "    한 주는 월요일에 시작한다."
)


def today_in_timezone() -> date:
    """서비스 시간대(TIMEZONE) 기준 오늘 날짜를 돌려준다. DateTools의 기본 시계다."""
    return datetime.now(TIMEZONE).date()


def _parse(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"날짜 형식이 올바르지 않습니다: {value!r} (YYYY-MM-DD)") from None


def _weekday(d: date) -> str:
    return WEEKDAYS_KO[d.weekday()]


@tool
def add_days(date: str, days: int) -> dict[str, str]:
    """날짜(YYYY-MM-DD)에 일수를 더한 날짜와 요일을 알려 준다. days가 음수면 이전 날짜."""
    d = _parse(date) + timedelta(days=days)
    return {"date": d.isoformat(), "weekday": _weekday(d)}


@tool
def days_between(start: str, end: str) -> dict[str, int]:
    """두 날짜(YYYY-MM-DD) 사이의 일수(end - start)를 알려 준다."""
    return {"days": (_parse(end) - _parse(start)).days}


@tool
def get_weekday(date: str) -> dict[str, str]:
    """날짜(YYYY-MM-DD)의 요일을 알려 준다."""
    return {"weekday": _weekday(_parse(date))}


@dataclass(frozen=True)
class DateTools:
    """"오늘"을 정하는 시계. tools()가 날짜 툴 5개를 LLM에 붙일 순서대로 돌려준다.

        DateTools().tools()                              # 서비스: TIMEZONE 기준 오늘
        DateTools(today=lambda: date(2026, 10, 12))      # 테스트: 기준일 고정
    """

    _: KW_ONLY

    today: Callable[[], date] = today_in_timezone
    """오늘 날짜를 돌려주는 함수"""

    def tools(self) -> list[BaseTool]:
        """날짜 툴 5개 (get_today · add_days · days_between · get_weekday · get_date_of_weekday)."""
        get_today = StructuredTool.from_function(self.get_today, name="get_today",
                                                 description=GET_TODAY_DESCRIPTION)
        get_date_of_weekday = StructuredTool.from_function(
            self.get_date_of_weekday, name="get_date_of_weekday", description=GET_DATE_OF_WEEKDAY_DESCRIPTION)
        return [get_today, add_days, days_between, get_weekday, get_date_of_weekday]

    def get_today(self) -> dict[str, str]:
        """오늘 날짜와 요일."""
        d = self.today()
        return {"date": d.isoformat(), "weekday": _weekday(d)}

    def get_date_of_weekday(self, weekday: str, week_offset: int = 0) -> dict[str, str]:
        """주(week_offset)와 요일로 말한 날짜와 요일. 모르는 요일 이름은 ValueError."""
        name = weekday.strip()
        if not name.endswith("요일"):
            name += "요일"
        if name not in WEEKDAYS_KO:
            raise ValueError(f"요일을 알 수 없습니다: {weekday!r} (월요일~일요일)")
        t = self.today()
        monday = t - timedelta(days=t.weekday())
        d = monday + timedelta(weeks=week_offset, days=WEEKDAYS_KO.index(name))
        return {"date": d.isoformat(), "weekday": _weekday(d)}
