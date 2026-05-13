from __future__ import annotations

import html
import re
from typing import Iterable, Optional
from urllib.parse import urlparse

import httpx

from .providers.search_provider import SearchResult
from .schemas import AccessStatus, ProvidedSource, Source, SourceType
from .source_classifier import classify_source_type

TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(raw_html: str) -> tuple[str, str]:
    title = ""
    title_match = TITLE_RE.search(raw_html)
    if title_match:
        title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip())
    body = SCRIPT_STYLE_RE.sub(" ", raw_html)
    body = TAG_RE.sub(" ", body)
    body = html.unescape(body)
    body = re.sub(r"\s+", " ", body).strip()
    return title, body


def preview(text: str, max_chars: int = 600) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


class SourceFetcher:
    def __init__(self, enable_url_fetch: bool = True, timeout: float = 8.0):
        self.enable_url_fetch = enable_url_fetch
        self.timeout = timeout

    def _source_from_provided(self, provided: ProvidedSource, source_id: str) -> Source:
        title = provided.title or (provided.url or "Provided source")
        source_type = provided.source_type
        if source_type == SourceType.UNKNOWN:
            source_type = classify_source_type(provided.url, title, provided.extracted_text)
        return Source(
            source_id=source_id,
            url=provided.url,
            title=title,
            publisher_or_author=provided.publisher_or_author or "",
            publication_date=provided.publication_date,
            access_status=provided.access_status,
            source_type=source_type,
            extracted_text=provided.extracted_text,
            extracted_text_preview=preview(provided.extracted_text),
        )

    def make_opaque_source(self, source_id: str, mention: str) -> Source:
        text = mention or "Opaque or unnamed source mention"
        return Source(
            source_id=source_id,
            url=None,
            title=text,
            publisher_or_author="",
            access_status=AccessStatus.UNAVAILABLE,
            source_type=SourceType.ANONYMOUS_OR_OPAQUE,
            extracted_text=text,
            extracted_text_preview=text,
        )

    def source_from_search_result(self, result: SearchResult, source_id: str) -> Source:
        if self.enable_url_fetch:
            fetched = self.fetch_url(result.url, source_id, [])
            if fetched.extracted_text:
                if not fetched.extracted_text_preview and result.snippet:
                    fetched.extracted_text_preview = preview(result.snippet)
                return fetched

        extracted = " ".join(part for part in [result.title, result.snippet] if part).strip()
        source_type = classify_source_type(result.url, result.title, extracted)
        return Source(
            source_id=source_id,
            url=result.url,
            title=result.title or result.url,
            access_status=AccessStatus.ACCESSIBLE if extracted else AccessStatus.UNAVAILABLE,
            source_type=source_type,
            extracted_text=extracted,
            extracted_text_preview=preview(extracted or "Discovered by web search, but no snippet was available."),
        )

    def fetch_url(self, url: str, source_id: str, provided_sources: list[ProvidedSource] | None = None) -> Source:
        provided_sources = provided_sources or []
        for provided in provided_sources:
            if provided.url and provided.url.rstrip("/") == url.rstrip("/"):
                return self._source_from_provided(provided, source_id)

        title = urlparse(url).netloc or url
        if not self.enable_url_fetch:
            source_type = classify_source_type(url, title, "")
            return Source(
                source_id=source_id,
                url=url,
                title=title,
                access_status=AccessStatus.UNAVAILABLE,
                source_type=source_type,
                extracted_text="",
                extracted_text_preview="URL detected. Fetching is disabled for this request.",
            )

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True, headers={"User-Agent": "SourceGroundingAuditor/0.1"}) as client:
                response = client.get(url)
                if response.status_code in {401, 402, 403}:
                    status = AccessStatus.PAYWALLED
                elif response.status_code >= 400:
                    status = AccessStatus.FAILED
                else:
                    status = AccessStatus.ACCESSIBLE
                content_type = response.headers.get("content-type", "")
                raw = response.text if "text" in content_type or "html" in content_type or not content_type else response.text
                fetched_title, text = html_to_text(raw)
                title = fetched_title or title
                source_type = classify_source_type(url, title, text)
                return Source(
                    source_id=source_id,
                    url=url,
                    title=title,
                    access_status=status,
                    source_type=source_type,
                    extracted_text=text,
                    extracted_text_preview=preview(text),
                )
        except Exception as exc:  # pragma: no cover - network dependent
            source_type = classify_source_type(url, title, "")
            return Source(
                source_id=source_id,
                url=url,
                title=title,
                access_status=AccessStatus.FAILED,
                source_type=source_type,
                extracted_text="",
                extracted_text_preview=f"Fetch failed: {exc}",
            )
