"""툴 레지스트리. 라벨별 허용 목록(TOOLS_BY_LABEL)에서, 키가 없는 툴을 빼고 돌려준다."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ...common.config import LLMConfig, get_config
from .api_weather import get_weather
from .api_web_search import web_search
from .tool_date import DATE_TOOLS

# 툴 그룹 이름 → 툴. LLMConfig.tools_by_label이 이 이름으로 라벨에 툴을 붙인다
TOOL_GROUPS: dict[str, list[BaseTool]] = {
    "date_tools": DATE_TOOLS,
    "weather": [get_weather],
    "web_search": [web_search],
}


def enabled_groups(config: LLMConfig) -> dict[str, list[BaseTool]]:
    """쓸 수 있는 툴 그룹. 날씨 · 검색은 키가 없으면 등록하지 않는다."""
    missing_key = {"weather": not config.kma_auth_key, "web_search": not config.tavily_api_key}
    return {name: tools for name, tools in TOOL_GROUPS.items() if not missing_key.get(name)}


def get_tools_for(label: str, config: LLMConfig | None = None) -> list[BaseTool]:
    """라벨에 켜진 툴 목록. 비어 있으면 답변 생성은 툴 없이 LLM을 1회 호출한다."""
    config = config or get_config()
    groups = enabled_groups(config)
    tools: list[BaseTool] = []
    for name in config.tools_by_label.get(label, []):
        tools.extend(groups.get(name, []))
    return tools
