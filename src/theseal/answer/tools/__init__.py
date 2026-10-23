"""툴 레지스트리. 라벨별 허용 목록(AnswerConfig.tools_by_label)에 적힌 툴 그룹을, 키가 없는 것은 빼고 돌려준다.

날짜 툴은 DateTools(시계)가, 키가 필요한 날씨 · 검색 툴은 설정(AnswerConfig)에서 만든다.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import BaseTool

from theseal.answer.config import AnswerConfig
from theseal.answer.tools.api_weather import WeatherClient
from theseal.answer.tools.api_web_search import WebSearchClient
from theseal.answer.tools.tool_date import DateTools


def _date_tools(config: AnswerConfig) -> list[BaseTool]:
    del config  # 날짜 툴은 설정을 쓰지 않는다. 그룹 만드는 함수의 시그니처 맞춤
    return DateTools().tools()


def _weather_tools(config: AnswerConfig) -> list[BaseTool]:
    return [WeatherClient.from_config(config).tool()] if config.kma_auth_key else []


def _web_search_tools(config: AnswerConfig) -> list[BaseTool]:
    return [WebSearchClient.from_config(config).tool()] if config.tavily_api_key else []


# 툴 그룹 이름 → 그 그룹의 툴을 만드는 함수. AnswerConfig.tools_by_label이 이 이름으로 라벨에 툴을 붙인다.
# 키가 없으면 빈 목록을 돌려준다 (등록하지 않는다)
TOOL_GROUPS: dict[str, Callable[[AnswerConfig], list[BaseTool]]] = {
    "date_tools": _date_tools,
    "weather": _weather_tools,
    "web_search": _web_search_tools,
}


def get_tools_for(label: str, config: AnswerConfig) -> list[BaseTool]:
    """라벨에 켜진 툴 목록을 돌려준다. 비어 있으면 답변 생성은 툴 없이 LLM을 1회 호출한다.
    TOOL_GROUPS에 없는 그룹 이름은 건너뛴다."""
    tools: list[BaseTool] = []
    for name in config.tools_by_label.get(label, ()):
        build = TOOL_GROUPS.get(name)
        if build is not None:
            tools.extend(build(config))
    return tools
