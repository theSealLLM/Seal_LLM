"""툴 레지스트리. 전체 스위치(TOOL_*)와 라벨별 허용 목록(TOOLS_BY_LABEL) 두 단계로 거른다."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ...config import LLMConfig, get_config
from .api_weather import get_weather
from .api_web_search import web_search
from .tool_date import DATE_TOOLS


def enabled_groups(config: LLMConfig) -> dict[str, list[BaseTool]]:
    groups: dict[str, list[BaseTool]] = {}
    if config.tool_date_tools:
        groups["date_tools"] = DATE_TOOLS
    if config.tool_weather and config.kma_auth_key:  # 키가 없으면 켜져 있어도 등록하지 않는다
        groups["weather"] = [get_weather]
    if config.tool_web_search and config.tavily_api_key:  # 키가 없으면 켜져 있어도 등록하지 않는다
        groups["web_search"] = [web_search]
    return groups


def get_tools_for(label: str, config: LLMConfig | None = None) -> list[BaseTool]:
    """라벨에 켜진 툴 목록. 비어 있으면 답변 생성은 툴 없이 LLM을 1회 호출한다."""
    config = config or get_config()
    groups = enabled_groups(config)
    tools: list[BaseTool] = []
    for name in config.tools_by_label.get(label, []):
        tools.extend(groups.get(name, []))
    return tools
