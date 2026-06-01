from __future__ import annotations

import re
from dataclasses import dataclass

from .citation_parser import (
    FOOTNOTE_REF_RE,
    MARKDOWN_LINK_RE,
    URL_RE,
    ReferenceDescription,
    citation_anchored_left_boundary,
    is_sentence_boundary_punctuation,
)
from .schemas import CitationEdge, CitationUnit, CitedStatement, EdgeBasis, LocatedCitation, ParsedCitation

SOURCE_REGISTRY_HEADING_RE = re.compile(
    r"^\s*(?:来源指针|资料来源|参考资料|references|sources|source pointer)\s*[:：]?\s*$",
    re.IGNORECASE,
)
REFERENCE_LINE_RE = re.compile(r"^\s*(?:\[(\d+)\]|\[?\^?(\d+)\]?|\d+\.|-)\s*[:.]?\s+")
REFERENCE_STYLE_LINK_RE = re.compile(r"\[([^\]]+)\]\[(?:\^)?(\d+)\]")


@dataclass(frozen=True)
class SourceRegistryEntry:
    label: str
    raw_text: str
    url: str | None = None
    title: str = ""
    description: str = ""
    registry_type: str = "unknown"


@dataclass(frozen=True)
class _CitationMarker:
    marker_text: str
    start: int
    end: int
    label: str | None = None
    source_url: str | None = None
    source_title: str | None = None
    source_registry_entry: str | None = None
    capture_method: str = "citation_marker"
    confidence: str = "medium"


def build_source_registry(
    citations: list[ParsedCitation],
    reference_descriptions: list[ReferenceDescription],
) -> dict[str, SourceRegistryEntry]:
    markdown_link_definitions = build_markdown_link_definitions(citations)
    source_pointer_entries = build_source_pointer_entries(
        reference_descriptions,
        markdown_link_definitions=markdown_link_definitions,
    )
    return {**markdown_link_definitions, **source_pointer_entries}


def build_markdown_link_definitions(citations: list[ParsedCitation]) -> dict[str, SourceRegistryEntry]:
    return _markdown_link_definitions(citations)


def build_source_pointer_entries(
    reference_descriptions: list[ReferenceDescription],
    *,
    markdown_link_definitions: dict[str, SourceRegistryEntry] | None = None,
) -> dict[str, SourceRegistryEntry]:
    return _source_pointer_entries(reference_descriptions, markdown_link_definitions or {})


def build_cited_statements(
    input_text: str,
    citations: list[ParsedCitation],
    reference_descriptions: list[ReferenceDescription],
) -> list[CitedStatement]:
    registry = build_source_registry(citations, reference_descriptions)
    registry_ranges = _source_registry_ranges(input_text)
    markers = _visible_citation_markers(input_text, citations, registry, registry_ranges)
    clusters = _citation_clusters(input_text, markers)
    statements: list[CitedStatement] = []
    previous_boundary = 0

    for cluster in clusters:
        cluster_start = cluster[0].start
        cluster_end = cluster[-1].end
        content_end = _skip_markdown_closing_markers_left(input_text, cluster_start, floor=previous_boundary)
        left = citation_anchored_left_boundary(input_text, content_end, floor=previous_boundary)
        left = _skip_leading_space(input_text, left, content_end)
        right = _right_boundary_after_cluster(input_text, cluster_end)
        cited_text = _cited_text_for_cluster(input_text, left, content_end, cluster_end, right)
        if not cited_text:
            previous_boundary = _skip_leading_space(input_text, right, len(input_text))
            continue

        statement_id = f"stmt_{len(statements)+1:03d}"
        edges = [
            CitationEdge(
                citation_id=f"{statement_id}_edge_{edge_index:03d}",
                label=marker.label,
                marker_text=marker.marker_text,
                source_url=marker.source_url,
                source_title=marker.source_title,
                source_registry_entry=marker.source_registry_entry,
                marker_start=marker.start,
                marker_end=marker.end,
                capture_method=marker.capture_method,
                confidence=marker.confidence,
            )
            for edge_index, marker in enumerate(cluster, start=1)
        ]
        statements.append(
            CitedStatement(
                statement_id=statement_id,
                cited_text=cited_text,
                char_start=left,
                char_end=right if right > cluster_end else content_end,
                citation_edges=edges,
            )
        )
        previous_boundary = _skip_leading_space(input_text, right, len(input_text))

    return statements


def build_cited_statements_from_located_citations(
    located_citations: list[LocatedCitation],
) -> list[CitedStatement]:
    grouped: dict[tuple[str, int, int], list[LocatedCitation]] = {}
    order: list[tuple[str, int, int]] = []
    for located in located_citations:
        cited_text = re.sub(r"\s+", " ", located.cited_text_span or "").strip()
        if not cited_text:
            continue
        key = (cited_text, located.char_start, located.char_end)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(located)

    statements: list[CitedStatement] = []
    for key in order:
        cited_text, char_start, char_end = key
        statement_id = f"stmt_{len(statements)+1:03d}"
        edges = [
            CitationEdge(
                citation_id=located.citation_id or f"{statement_id}_edge_{edge_index:03d}",
                label=located.source_label,
                marker_text=located.marker_text or (
                    f"[{located.source_label}]" if located.source_label else located.source_url or ""
                ),
                source_url=located.source_url,
                source_title=located.source_title,
                source_registry_entry=located.source_title or located.source_url or located.marker_text or None,
                marker_start=max(0, char_end),
                marker_end=max(0, char_end),
                capture_method=located.capture_method.value,
                confidence=located.confidence or "medium",
            )
            for edge_index, located in enumerate(grouped[key], start=1)
        ]
        statements.append(
            CitedStatement(
                statement_id=statement_id,
                cited_text=cited_text,
                char_start=char_start,
                char_end=char_end,
                citation_edges=edges,
            )
        )
    return statements


def build_citation_units(
    input_text: str,
    citations: list[ParsedCitation],
    reference_descriptions: list[ReferenceDescription],
) -> list[CitationUnit]:
    return citation_units_from_cited_statements(
        build_cited_statements(input_text, citations, reference_descriptions)
    )


def citation_units_from_cited_statements(statements: list[CitedStatement]) -> list[CitationUnit]:
    units: list[CitationUnit] = []
    for statement in statements:
        for edge in statement.citation_edges:
            units.append(
                CitationUnit(
                    cited_text=statement.cited_text,
                    citation_label=edge.label,
                    source_url=edge.source_url,
                    source_title=edge.source_title or "",
                    source_registry_entry=edge.source_registry_entry or "",
                    char_start=statement.char_start,
                    char_end=statement.char_end,
                    cited_statement_id=statement.statement_id,
                    citation_edge_id=edge.citation_id,
                    citation_edges=statement.citation_edges,
                )
            )
    return units


def parsed_citation_for_unit(unit: CitationUnit, citation_id: str = "cit_unit_001") -> ParsedCitation:
    raw = f"[{unit.citation_label}]" if unit.citation_label else (unit.source_url or "")
    return ParsedCitation(
        citation_id=citation_id,
        raw_text=raw,
        url=unit.source_url,
        label=unit.citation_label,
        span_start=unit.char_start,
        span_end=unit.char_end,
    )


def parsed_citation_for_edge(edge: CitationEdge, citation_id: str | None = None) -> ParsedCitation:
    return ParsedCitation(
        citation_id=citation_id or edge.citation_id,
        raw_text=edge.marker_text,
        url=edge.source_url,
        label=edge.label,
        span_start=edge.marker_start,
        span_end=edge.marker_end,
    )


def _visible_citation_markers(
    text: str,
    citations: list[ParsedCitation],
    registry: dict[str, SourceRegistryEntry],
    registry_ranges: list[tuple[int, int]],
) -> list[_CitationMarker]:
    markers: list[_CitationMarker] = []
    occupied_spans: list[tuple[int, int]] = []

    for match in MARKDOWN_LINK_RE.finditer(text):
        if _overlaps_any(match.start(), match.end(), registry_ranges):
            continue
        label, url = match.group(1).strip(), match.group(2).strip()
        occupied_spans.append(match.span())
        markers.append(
            _CitationMarker(
                marker_text=match.group(0),
                start=match.start(),
                end=match.end(),
                label=label or None,
                source_url=url,
                source_title=label,
                source_registry_entry=match.group(0),
                capture_method="markdown_link",
                confidence="high",
            )
        )

    for match in FOOTNOTE_REF_RE.finditer(text):
        if _overlaps_any(match.start(), match.end(), registry_ranges):
            continue
        if _span_inside_any(match.start(), match.end(), occupied_spans):
            continue
        label = match.group(1)
        entry = registry.get(label)
        markers.append(
            _CitationMarker(
                marker_text=match.group(0),
                start=match.start(),
                end=match.end(),
                label=label,
                source_url=entry.url if entry else None,
                source_title=entry.title if entry else None,
                source_registry_entry=entry.raw_text if entry else None,
                capture_method="footnote_label",
                confidence="high" if entry and entry.url else "medium",
            )
        )

    parsed_raw_url_spans = {
        (citation.span_start, citation.span_end): citation
        for citation in citations
        if citation.url and not citation.label
    }
    for match in URL_RE.finditer(text):
        if _overlaps_any(match.start(), match.end(), registry_ranges):
            continue
        if _span_inside_any(match.start(), match.end(), occupied_spans):
            continue
        citation = parsed_raw_url_spans.get((match.start(), match.end()))
        url = citation.url if citation else match.group(0).strip().rstrip(".,;:)]}")
        markers.append(
            _CitationMarker(
                marker_text=match.group(0),
                start=match.start(),
                end=match.end(),
                source_url=url,
                source_registry_entry=match.group(0),
                capture_method="raw_url",
                confidence="medium",
            )
        )

    markers.sort(key=lambda marker: (marker.start, marker.end))
    deduped: list[_CitationMarker] = []
    seen: set[tuple[int, int, str]] = set()
    for marker in markers:
        key = (marker.start, marker.end, marker.marker_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(marker)
    return deduped


def _citation_clusters(text: str, markers: list[_CitationMarker]) -> list[list[_CitationMarker]]:
    clusters: list[list[_CitationMarker]] = []
    current: list[_CitationMarker] = []
    for marker in markers:
        if not current:
            current = [marker]
            continue
        gap = text[current[-1].end : marker.start]
        if gap.strip() == "":
            current.append(marker)
        else:
            clusters.append(current)
            current = [marker]
    if current:
        clusters.append(current)
    return clusters


def _cited_text_for_cluster(text: str, left: int, cluster_start: int, cluster_end: int, right: int) -> str:
    prefix = text[left:cluster_start].strip()
    if not prefix:
        return ""
    suffix = ""
    if right > cluster_end:
        suffix = text[cluster_end:right].strip()
    return f"{prefix}{suffix}".strip()


def _skip_markdown_closing_markers_left(text: str, index: int, *, floor: int = 0) -> int:
    probe = index
    while probe > floor and text[probe - 1].isspace() and text[probe - 1] != "\n":
        probe -= 1
    for marker in ("**", "__", "*", "_"):
        if probe - len(marker) >= floor and text[probe - len(marker) : probe] == marker:
            return probe - len(marker)
    return index


def _right_boundary_after_cluster(text: str, cluster_end: int) -> int:
    right = cluster_end
    probe = cluster_end
    while probe < len(text) and text[probe].isspace() and text[probe] != "\n":
        probe += 1
    if probe < len(text) and is_sentence_boundary_punctuation(text, probe):
        right = probe + 1
    return right


def _skip_leading_space(text: str, start: int, limit: int) -> int:
    while start < limit and text[start].isspace():
        start += 1
    return start


def _markdown_link_definitions(citations: list[ParsedCitation]) -> dict[str, SourceRegistryEntry]:
    registry: dict[str, SourceRegistryEntry] = {}
    for citation in citations:
        if not citation.label or not citation.url:
            continue
        if not _is_markdown_link_definition(citation.raw_text):
            continue
        title = _title_from_registry_line(citation.raw_text, citation.url)
        registry[citation.label] = SourceRegistryEntry(
            label=citation.label,
            raw_text=citation.raw_text,
            url=citation.url,
            title=title,
            description=title or citation.raw_text,
            registry_type="markdown_link_definition",
        )
    return registry


def _source_pointer_entries(
    reference_descriptions: list[ReferenceDescription],
    markdown_link_definitions: dict[str, SourceRegistryEntry],
) -> dict[str, SourceRegistryEntry]:
    registry: dict[str, SourceRegistryEntry] = {}
    for reference in reference_descriptions:
        url, linked_title = _source_pointer_url(reference.raw_text, markdown_link_definitions)
        registry[reference.label] = SourceRegistryEntry(
            label=reference.label,
            raw_text=reference.raw_text,
            url=url,
            description=reference.description,
            title=reference.description or linked_title,
            registry_type="source_pointer_entry",
        )
    return registry


def _is_markdown_link_definition(raw_text: str) -> bool:
    return bool(re.match(r"^\s*\[?\^?\d+\]?\s*:\s*https?://", raw_text or "", flags=re.IGNORECASE))


def _source_pointer_url(
    raw_text: str,
    markdown_link_definitions: dict[str, SourceRegistryEntry],
) -> tuple[str | None, str]:
    inline_match = MARKDOWN_LINK_RE.search(raw_text or "")
    if inline_match:
        return inline_match.group(2).strip(), inline_match.group(1).strip()
    reference_style_match = REFERENCE_STYLE_LINK_RE.search(raw_text or "")
    if reference_style_match:
        linked_title = reference_style_match.group(1).strip()
        linked_label = reference_style_match.group(2)
        linked_entry = markdown_link_definitions.get(linked_label)
        if linked_entry and linked_entry.url:
            return linked_entry.url, linked_title or linked_entry.title
        return None, linked_title
    url_match = URL_RE.search(raw_text or "")
    if url_match:
        return url_match.group(0).strip().rstrip(".,;:)]}"), ""
    return None, ""


def _source_registry_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start = offset
        line_end = offset + len(line)
        if SOURCE_REGISTRY_HEADING_RE.match(stripped):
            current_start = line_start
        elif current_start is not None:
            if stripped and not _is_reference_line(stripped):
                ranges.append((current_start, line_start))
                current_start = None
        elif _is_reference_line(stripped):
            ranges.append((line_start, line_end))
        offset = line_end
    if current_start is not None:
        ranges.append((current_start, len(text)))
    return ranges


def _is_reference_line(line: str) -> bool:
    return bool(REFERENCE_LINE_RE.match(line)) and (
        "http://" in line
        or "https://" in line
        or bool(re.match(r"^\s*\[\d+\]\s+\S", line))
    )


def _overlaps_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < range_end and range_start < end for range_start, range_end in ranges)


def _span_inside_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(range_start <= start and end <= range_end for range_start, range_end in ranges)


def _title_from_registry_line(raw_text: str, url: str | None) -> str:
    if not raw_text:
        return ""
    title_match = re.search(r'"([^"]+)"|“([^”]+)”', raw_text)
    if title_match:
        return (title_match.group(1) or title_match.group(2) or "").strip()
    if url and url in raw_text:
        rest = raw_text.split(url, 1)[1]
        return rest.strip(" -–—:：\t\r\n\"'")
    return ""
