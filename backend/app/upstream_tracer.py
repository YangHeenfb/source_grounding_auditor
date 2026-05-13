from __future__ import annotations

from typing import Dict, List

from .citation_parser import parse_citations
from .schemas import EdgeBasis, EdgeType, EvidenceEdge, Source
from .source_fetcher import SourceFetcher


class UpstreamTracer:
    """Trace only explicit upstream links found inside a source body.

    This class does not infer lineage from semantic similarity. It only creates upstream
    source nodes when the source text itself contains visible URLs or references.
    """

    def __init__(self, fetcher: SourceFetcher):
        self.fetcher = fetcher

    def trace(self, source: Source, *, existing_sources: dict[str, Source], max_depth: int = 2) -> list[Source]:
        if max_depth < 2 or not source.extracted_text:
            return []
        citations = parse_citations(source.extracted_text)
        added: list[Source] = []
        for citation in citations[:5]:
            if not citation.url:
                continue
            if any(s.url and s.url.rstrip("/") == citation.url.rstrip("/") for s in existing_sources.values()):
                continue
            source_id = f"s{len(existing_sources)+1:03d}"
            upstream = self.fetcher.fetch_url(citation.url, source_id, [])
            existing_sources[source_id] = upstream
            source.upstream_source_ids.append(source_id)
            added.append(upstream)
        return added
