"""Tavily 웹 검색 툴. 예외 처리 경로라 캐시하지 않고, 결과가 호출마다 달라도 된다.

외부 API로 검색어가 나간다. 검색어에 개인정보를 넣지 않는 규칙은 프롬프트와 툴 설명에만 있다.
키는 AnswerConfig의 tavily_api_key(.env의 TAVILY_API_KEY). 툴은 WebSearchClient.from_config(config).tool()로 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, StructuredTool
from tavily import TavilyClient

from theseal.answer.config import AnswerConfig

MAX_RESULTS = 3
MAX_CONTENT_CHARS = 500

WEB_SEARCH_TOOL_NAME = "web_search"
# LLM에 보내는 툴 설명. 예전 @tool 함수의 docstring 그대로다 (줄 앞 공백 4칸까지 같아야 LLM 입력이 바뀌지 않는다)
WEB_SEARCH_TOOL_DESCRIPTION = (
    "사실을 확인해야 할 때 웹을 검색한다. 최신 정보(뉴스·날씨·환율)와 일반 지식(인물·역사·과학·지리) 모두 해당한다.\n"
    "    검색어에는 고객의 이름·전화번호·주소·주문번호 같은 개인정보를 넣지 않는다."
)


@dataclass(frozen=True)
class WebSearchClient:
    """Tavily로 웹을 검색하는 클라이언트. tool()이 LLM에 붙일 web_search 툴을 만든다.

        tool = WebSearchClient.from_config(answer_config).tool()
        tool.invoke({"query": "대한민국 수도"})  →  [{"title": …, "url": …, "content": …}, …]
    """

    api_key: str = field(repr=False)
    """Tavily API 키. repr에 남기지 않는다"""

    @classmethod
    def from_config(cls, config: AnswerConfig) -> WebSearchClient:
        """키를 AnswerConfig(TAVILY_API_KEY)에서 읽는다."""
        return cls(api_key=config.tavily_api_key)

    def tool(self) -> BaseTool:
        """LLM에 붙일 web_search 툴. 실행하면 이 클라이언트의 search를 부른다."""
        return StructuredTool.from_function(self.search, name=WEB_SEARCH_TOOL_NAME,
                                            description=WEB_SEARCH_TOOL_DESCRIPTION)

    def search(self, query: str) -> list[dict[str, str]]:
        """검색 결과 최대 MAX_RESULTS건을 돌려준다. 본문은 MAX_CONTENT_CHARS자까지만 남긴다."""
        resp = TavilyClient(api_key=self.api_key).search(query=query, max_results=MAX_RESULTS)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:MAX_CONTENT_CHARS],
            }
            for r in resp.get("results", [])[:MAX_RESULTS]
        ]
