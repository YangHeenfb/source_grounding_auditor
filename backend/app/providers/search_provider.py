from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


class SearchProvider(Protocol):
    """Extension point for discovered sources.

    Important product rule: discovered search results are not real upstream edges unless
    the analyzed source explicitly links to or names them. They should be marked as
    discovered_source in evidence edges.
    """

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._current_url = ""
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = set(attrs_dict.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._in_title = True
            self._current_url = _unwrap_duckduckgo_url(attrs_dict.get("href", ""))
            self._current_title = []
            self._current_snippet = []
        elif "result__snippet" in classes:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            title = " ".join("".join(self._current_title).split())
            if title and self._current_url:
                self.results.append(SearchResult(title=title, url=self._current_url, snippet=""))
        elif self._in_snippet and tag in {"a", "div"}:
            self._in_snippet = False
            snippet = " ".join("".join(self._current_snippet).split())
            if snippet and self.results and not self.results[-1].snippet:
                self.results[-1].snippet = snippet

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)


class DuckDuckGoSearchProvider:
    """Small no-key web search provider for local MVP use."""

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        query = " ".join((query or "").split())
        if not query:
            return []
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "SourceGroundingAuditor/0.1"},
            ) as client:
                response = client.get(url)
                response.raise_for_status()
        except httpx.HTTPError:
            return []

        parser = _DuckDuckGoHTMLParser()
        parser.feed(response.text)
        unique: dict[str, SearchResult] = {}
        for result in parser.results:
            if result.url.startswith("http"):
                unique.setdefault(result.url.rstrip("/"), result)
            if len(unique) >= max_results:
                break
        return list(unique.values())


def _unwrap_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") or parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return url
