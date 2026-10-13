"""Tavily 웹 검색 툴. 예외 처리 경로라 캐시하지 않고, 결과가 호출마다 달라도 된다.

외부 API로 검색어가 나간다. 검색어에 개인정보를 넣지 않는 규칙은 프롬프트와 툴 설명에만 있다.
"""

from __future__ import annotations

from langchain_core.tools import tool

from ...config import get_config

MAX_RESULTS = 3
MAX_CONTENT_CHARS = 500


@tool
def web_search(query: str) -> list[dict]:
    """사실을 확인해야 할 때 웹을 검색한다. 최신 정보(뉴스·날씨·환율)와 일반 지식(인물·역사·과학·지리) 모두 해당한다.
    검색어에는 고객의 이름·전화번호·주소·주문번호 같은 개인정보를 넣지 않는다."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=get_config().tavily_api_key)
    resp = client.search(query=query, max_results=MAX_RESULTS)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": (r.get("content") or "")[:MAX_CONTENT_CHARS],
        }
        for r in resp.get("results", [])[:MAX_RESULTS]
    ]
