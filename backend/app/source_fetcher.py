from __future__ import annotations

import html
import re
import subprocess
from typing import Iterable, Optional
from urllib.parse import urlparse

import httpx

from .providers.search_provider import SearchResult
from .schemas import AccessStatus, ProvidedSource, Source, SourceType
from .official_domain_verifier import verify_official_domain
from .source_classifier import classify_source_type
from .source_entity_resolver import extract_html_metadata, resolve_source_entity

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
        return self._enrich_source(
            Source(
                source_id=source_id,
                url=provided.url,
                title=title,
                publisher_or_author=provided.publisher_or_author or "",
                publication_date=provided.publication_date,
                access_status=provided.access_status,
                source_type=source_type,
                extracted_text=provided.extracted_text,
                extracted_text_preview=preview(provided.extracted_text),
                fetch_method="provided",
            ),
            html_metadata={},
            provided_source_type=provided.source_type,
        )

    def _enrich_source(
        self,
        source: Source,
        *,
        html_metadata: dict | None = None,
        provided_source_type: SourceType | None = None,
    ) -> Source:
        resolution = resolve_source_entity(
            url=source.url,
            title=source.title,
            html_metadata=html_metadata or {},
            source_text=source.extracted_text or source.extracted_text_preview,
            publisher_or_author=source.publisher_or_author,
        )
        verification = verify_official_domain(
            resolution,
            metadata={"provided_source_type": provided_source_type.value if provided_source_type else None},
        )
        source.source_entity = resolution.source_entity
        source.registrable_domain = resolution.registrable_domain
        source.publisher_name = resolution.publisher_name
        source.organization_type = resolution.organization_type
        source.entity_aliases = resolution.entity_aliases
        source.metadata_basis = resolution.metadata_basis
        source.officialness_status = verification.officialness_status
        source.officialness_basis = verification.basis
        return source

    def make_opaque_source(self, source_id: str, mention: str) -> Source:
        text = mention or "Opaque or unnamed source mention"
        return self._enrich_source(
            Source(
                source_id=source_id,
                url=None,
                title=text,
                publisher_or_author="",
                access_status=AccessStatus.UNAVAILABLE,
                source_type=SourceType.ANONYMOUS_OR_OPAQUE,
                extracted_text=text,
                extracted_text_preview=text,
                fetch_method="opaque_source",
            )
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
        return self._enrich_source(
            Source(
                source_id=source_id,
                url=result.url,
                title=result.title or result.url,
                access_status=AccessStatus.ACCESSIBLE if extracted else AccessStatus.UNAVAILABLE,
                source_type=source_type,
                extracted_text=extracted,
                extracted_text_preview=preview(extracted or "Discovered by web search, but no snippet was available."),
                fetch_method="search_result",
            )
        )

    def fetch_url(self, url: str, source_id: str, provided_sources: list[ProvidedSource] | None = None) -> Source:
        provided_sources = provided_sources or []
        for provided in provided_sources:
            if provided.url and provided.url.rstrip("/") == url.rstrip("/"):
                return self._source_from_provided(provided, source_id)

        title = urlparse(url).netloc or url
        if not self.enable_url_fetch:
            source_type = classify_source_type(url, title, "")
            return self._enrich_source(
                Source(
                    source_id=source_id,
                    url=url,
                    title=title,
                    access_status=AccessStatus.UNAVAILABLE,
                    source_type=source_type,
                    extracted_text="",
                    extracted_text_preview="URL detected. Fetching is disabled for this request.",
                    fetch_method="fetch_disabled",
                )
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
                metadata = extract_html_metadata(raw)
                fetched_title, text = html_to_text(raw)
                title = fetched_title or title
                source_type = classify_source_type(url, title, text)
                return self._enrich_source(
                    Source(
                        source_id=source_id,
                        url=url,
                        title=title,
                        access_status=status,
                        source_type=source_type,
                        extracted_text=text,
                        extracted_text_preview=preview(text),
                        fetch_method="httpx",
                    ),
                    html_metadata=metadata,
                )
        except Exception as exc:  # pragma: no cover - network dependent
            if _is_tls_error(exc):
                fallback = self._curl_fallback(url, source_id, title, exc)
                if fallback is not None:
                    return fallback
            source_type = classify_source_type(url, title, "")
            return self._enrich_source(
                Source(
                    source_id=source_id,
                    url=url,
                    title=title,
                    access_status=AccessStatus.FAILED,
                    source_type=source_type,
                    extracted_text="",
                    extracted_text_preview=f"Fetch failed: {exc}",
                    fetch_method="httpx_error",
                )
            )

    def _curl_fallback(self, url: str, source_id: str, title: str, original_exc: Exception) -> Source | None:
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-L",
                    "--max-time",
                    str(max(1, int(self.timeout))),
                    "-A",
                    "Mozilla/5.0",
                    "-sS",
                    "-w",
                    "\n%{http_code}",
                    url,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        body, _, status_text = result.stdout.rpartition("\n")
        try:
            status_code = int(status_text.strip())
        except ValueError:
            status_code = 0
        if status_code in {401, 402, 403}:
            status = AccessStatus.PAYWALLED
        elif status_code >= 400 or status_code == 0:
            status = AccessStatus.FAILED
        else:
            status = AccessStatus.ACCESSIBLE
        metadata = extract_html_metadata(body)
        fetched_title, text = html_to_text(body)
        source_title = fetched_title or title
        source_type = classify_source_type(url, source_title, text)
        return self._enrich_source(
            Source(
                source_id=source_id,
                url=url,
                title=source_title,
                access_status=status,
                source_type=source_type,
                extracted_text=text,
                extracted_text_preview=preview(text) if text else f"Fetch failed after TLS fallback: {original_exc}",
                fetch_method="curl_fallback",
            ),
            html_metadata=metadata,
        )


def _is_tls_error(exc: Exception) -> bool:
    text = repr(exc).lower()
    return any(
        needle in text
        for needle in [
            "ssl",
            "tls",
            "handshake",
            "certificate",
            "cert_verify",
            "sslv3_alert",
        ]
    )
