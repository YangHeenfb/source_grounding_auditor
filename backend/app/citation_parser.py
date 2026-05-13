from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .schemas import EdgeBasis, ParsedCitation

URL_RE = re.compile(r"https?://[^\s\])}>\"']+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", re.IGNORECASE)
FOOTNOTE_DEF_RE = re.compile(r"(?m)^\s*\[?\^?(\d+)\]?\s*[:.]\s*(https?://\S+)(?:\s+(.*))?$")
FOOTNOTE_REF_RE = re.compile(r"\[(?:\^)?(\d+)\]")
REFERENCE_LINE_RE = re.compile(r"(?m)^\s*(?:\[(\d+)\]|(\d+)\.|-)\s*(.+?)(https?://\S+)\s*$")
REFERENCE_DESCRIPTION_LINE_RE = re.compile(r"(?m)^\s*\[(\d+)\]\s*(?![:.]?\s*https?://)(.+?)\s*$")


@dataclass
class ReferenceDescription:
    label: str
    raw_text: str
    description: str
    span_start: int
    span_end: int


def _clean_url(url: str) -> str:
    return url.strip().rstrip(".,;:)]}")


def parse_citations(text: str) -> List[ParsedCitation]:
    """Parse explicit citations without inferring new source edges.

    The parser deliberately extracts only visible citations: markdown links, raw URLs,
    footnote definitions, and reference-list URLs. It does not create guessed sources.
    """

    citations: list[ParsedCitation] = []
    seen: set[tuple[str, int, int, str]] = set()

    def add(raw: str, url: Optional[str], label: Optional[str], kind: EdgeBasis, start: int, end: int) -> None:
        normalized_url = _clean_url(url) if url else None
        key = (normalized_url or raw, start, end, kind.value)
        if key in seen:
            return
        seen.add(key)
        citations.append(
            ParsedCitation(
                citation_id=f"cit_{len(citations)+1:03d}",
                raw_text=raw,
                url=normalized_url,
                label=label,
                kind=kind,
                span_start=start,
                span_end=end,
            )
        )

    markdown_spans: list[tuple[int, int]] = []
    for match in MARKDOWN_LINK_RE.finditer(text):
        label, url = match.group(1), match.group(2)
        markdown_spans.append(match.span())
        add(match.group(0), url, label, EdgeBasis.MARKDOWN_CITATION, match.start(), match.end())

    for match in FOOTNOTE_DEF_RE.finditer(text):
        label, url = match.group(1), match.group(2)
        add(match.group(0), url, label, EdgeBasis.FOOTNOTE, match.start(), match.end())

    for match in REFERENCE_LINE_RE.finditer(text):
        label = match.group(1) or match.group(2)
        url = match.group(4)
        kind = EdgeBasis.REFERENCE_LIST if label or match.group(0).strip().startswith("-") else EdgeBasis.EXPLICIT_LINK
        add(match.group(0), url, label, kind, match.start(), match.end())

    for match in URL_RE.finditer(text):
        # Avoid duplicating the URL inside a markdown link already captured.
        if any(start <= match.start() and match.end() <= end for start, end in markdown_spans):
            continue
        add(match.group(0), match.group(0), None, EdgeBasis.EXPLICIT_LINK, match.start(), match.end())

    citations.sort(key=lambda c: (c.span_start, c.span_end))
    return citations


def parse_reference_descriptions(text: str) -> list[ReferenceDescription]:
    """Parse bracketed reference descriptions that do not expose a URL.

    Example: "[1] Reuters 2026 年 2 月 2 日独家报道：..."
    These are not citations by themselves, but they are useful search queries when
    the caller explicitly enables web search.
    """

    references: list[ReferenceDescription] = []
    seen: set[str] = set()
    for match in REFERENCE_DESCRIPTION_LINE_RE.finditer(text):
        label = match.group(1)
        description = re.sub(r"\s+", " ", match.group(2)).strip()
        if not description or label in seen:
            continue
        seen.add(label)
        references.append(
            ReferenceDescription(
                label=label,
                raw_text=match.group(0),
                description=description,
                span_start=match.start(),
                span_end=match.end(),
            )
        )
    return references


def sentence_window_for_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Return a rough sentence/paragraph window around a text span."""

    if not text:
        return (0, 0)
    left_candidates = [text.rfind(p, 0, start) for p in [".", "!", "?", "\n"]]
    left = max(left_candidates)
    left = 0 if left == -1 else min(left + 1, len(text))
    right_positions = [text.find(p, end) for p in [".", "!", "?", "\n"]]
    right_positions = [p for p in right_positions if p != -1]
    right = min(right_positions) + 1 if right_positions else len(text)
    return (left, right)


def citations_near_text_span(full_text: str, original_span: str, citations: list[ParsedCitation]) -> list[ParsedCitation]:
    """Find citations in the same rough sentence or paragraph as a claim span."""

    if not full_text or not original_span:
        return []
    idx = full_text.find(original_span)
    if idx == -1:
        # Fall back to overlap by citation raw text appearing in the claim span.
        return [c for c in citations if c.raw_text and c.raw_text in original_span]
    left, right = sentence_window_for_span(full_text, idx, idx + len(original_span))
    near = [c for c in citations if left <= c.span_start <= right or c.raw_text in original_span]

    # Handle footnote refs in the claim sentence by matching reference definitions.
    window_text = full_text[left:right]
    labels = set(FOOTNOTE_REF_RE.findall(window_text))
    if labels:
        near.extend([c for c in citations if c.label in labels])

    unique: dict[str, ParsedCitation] = {}
    for c in near:
        unique[c.citation_id] = c
    return list(unique.values())


def reference_descriptions_near_text_span(
    full_text: str,
    original_span: str,
    references: list[ReferenceDescription],
) -> list[ReferenceDescription]:
    if not full_text or not original_span or not references:
        return []
    idx = full_text.find(original_span)
    if idx == -1:
        labels = set(FOOTNOTE_REF_RE.findall(original_span))
    else:
        left, right = sentence_window_for_span(full_text, idx, idx + len(original_span))
        labels = set(FOOTNOTE_REF_RE.findall(full_text[left:right]))
    return [reference for reference in references if reference.label in labels]


def extract_source_mentions(text: str) -> list[str]:
    """Extract lightweight source mentions from one claim/sentence."""

    patterns = [
        r"according to\s+([^,.。;:]+)",
        r"(experts?)\s+(?:say|said|argue|believe)",
        r"(sources?\s+(?:familiar with|close to|inside|say|said)[^,.。;:]*)",
        r"(insiders?)\s+(?:say|said|believe)",
        r"((?:a|the)\s+(?:study|report|paper|survey|filing|annual report|court filing))\s+(?:says|said|found|shows|showed|reported|claims|concludes)",
        r"((?:market|industry|expert)\s+commentary)",
    ]
    mentions: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            mentions.append(match.group(1).strip())
    return list(dict.fromkeys(mentions))


def is_vague_or_anonymous_mention(mention: str) -> bool:
    mention_l = mention.lower()
    vague_bits = ["experts", "sources", "insider", "familiar with", "close to", "people familiar"]
    return any(bit in mention_l for bit in vague_bits)
