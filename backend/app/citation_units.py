from __future__ import annotations

import re
from dataclasses import dataclass

from .citation_parser import (
    FOOTNOTE_REF_RE,
    MARKDOWN_LINK_RE,
    URL_RE,
    ReferenceDescription,
)
from .schemas import CitationUnit, ParsedCitation

SOURCE_REGISTRY_HEADING_RE = re.compile(
    r"^\s*(?:来源指针|资料来源|参考资料|references|sources|source pointer)\s*[:：]?\s*$",
    re.IGNORECASE,
)
REFERENCE_LINE_RE = re.compile(r"^\s*(?:\[(\d+)\]|\[?\^?(\d+)\]?|\d+\.|-)\s*[:.]?\s+")
SENTENCE_RE = re.compile(r"\S(?:.*?)(?:[。！？.!?]+(?=\s|$)|(?=\n)|$)", re.DOTALL)


@dataclass(frozen=True)
class SourceRegistryEntry:
    label: str
    raw_text: str
    url: str | None = None
    title: str = ""
    description: str = ""


def build_citation_units(
    input_text: str,
    citations: list[ParsedCitation],
    reference_descriptions: list[ReferenceDescription],
) -> list[CitationUnit]:
    registry = _source_registry(citations, reference_descriptions)
    registry_ranges = _source_registry_ranges(input_text)
    units: list[CitationUnit] = []
    seen: set[tuple[int, int, str, str | None]] = set()

    for start, end, sentence in _sentence_spans(input_text):
        if not sentence.strip() or _overlaps_any(start, end, registry_ranges):
            continue
        labels = FOOTNOTE_REF_RE.findall(sentence)
        for label in labels:
            entry = registry.get(label)
            unit = CitationUnit(
                cited_text=sentence.strip(),
                citation_label=label,
                source_url=entry.url if entry else None,
                source_title=entry.title if entry else "",
                source_registry_entry=entry.raw_text if entry else "",
                char_start=start,
                char_end=end,
            )
            key = (unit.char_start, unit.char_end, unit.cited_text, unit.citation_label)
            if key not in seen:
                seen.add(key)
                units.append(unit)

        for match in MARKDOWN_LINK_RE.finditer(sentence):
            label, url = match.group(1), match.group(2)
            unit = CitationUnit(
                cited_text=sentence.strip(),
                citation_label=label,
                source_url=url.strip(),
                source_title=label.strip(),
                source_registry_entry=match.group(0),
                char_start=start,
                char_end=end,
            )
            key = (unit.char_start, unit.char_end, unit.cited_text, unit.source_url)
            if key not in seen:
                seen.add(key)
                units.append(unit)

        for match in URL_RE.finditer(sentence):
            if any(link_match.start() <= match.start() and match.end() <= link_match.end() for link_match in MARKDOWN_LINK_RE.finditer(sentence)):
                continue
            unit = CitationUnit(
                cited_text=sentence.strip(),
                source_url=match.group(0).strip(),
                source_registry_entry=match.group(0),
                char_start=start,
                char_end=end,
            )
            key = (unit.char_start, unit.char_end, unit.cited_text, unit.source_url)
            if key not in seen:
                seen.add(key)
                units.append(unit)

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


def _source_registry(
    citations: list[ParsedCitation],
    reference_descriptions: list[ReferenceDescription],
) -> dict[str, SourceRegistryEntry]:
    registry: dict[str, SourceRegistryEntry] = {}
    for citation in citations:
        if not citation.label:
            continue
        title = _title_from_registry_line(citation.raw_text, citation.url)
        registry[citation.label] = SourceRegistryEntry(
            label=citation.label,
            raw_text=citation.raw_text,
            url=citation.url,
            title=title,
            description=title or citation.raw_text,
        )
    for reference in reference_descriptions:
        registry.setdefault(
            reference.label,
            SourceRegistryEntry(
                label=reference.label,
                raw_text=reference.raw_text,
                description=reference.description,
                title=reference.description,
            ),
        )
    return registry


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


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in SENTENCE_RE.finditer(text):
        sentence = match.group(0)
        if sentence.strip():
            spans.append((match.start(), match.end(), sentence))
    return spans


def _is_reference_line(line: str) -> bool:
    return bool(REFERENCE_LINE_RE.match(line)) and (
        "http://" in line
        or "https://" in line
        or bool(re.match(r"^\s*\[\d+\]\s+\S", line))
    )


def _overlaps_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < range_end and range_start < end for range_start, range_end in ranges)


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
