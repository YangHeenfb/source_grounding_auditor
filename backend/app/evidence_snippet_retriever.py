from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSnippet:
    text: str
    score: float
    basis: list[str]


SENTENCE_RE = re.compile(r"[^。！？.!?\n]+(?:[。！？.!?]+|$)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|:1|人|名|个|所|位|million|billion|%)?", re.IGNORECASE)
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.-]{1,}")
CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][A-Za-z&.-]+(?:\s+|$)){1,5}")
CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


def retrieve_evidence_snippets(cited_text: str, source_text: str, *, top_k: int = 5) -> list[EvidenceSnippet]:
    query = _features(cited_text)
    if not source_text.strip() or not query["all"]:
        return []

    snippets: list[EvidenceSnippet] = []
    for sentence in _sentences(source_text):
        sentence_features = _features(sentence)
        score = 0.0
        basis: list[str] = []

        number_overlap = query["numbers"] & sentence_features["numbers"]
        if number_overlap:
            score += 5.0 * len(number_overlap)
            basis.append(f"number overlap: {', '.join(sorted(number_overlap))}")

        entity_overlap = query["entities"] & sentence_features["entities"]
        if entity_overlap:
            score += 3.0 * len(entity_overlap)
            basis.append(f"entity/proper-noun overlap: {', '.join(sorted(entity_overlap)[:6])}")

        token_overlap = query["tokens"] & sentence_features["tokens"]
        if token_overlap:
            score += 1.0 * len(token_overlap)
            basis.append(f"token overlap: {', '.join(sorted(token_overlap)[:8])}")

        if score > 0:
            snippets.append(EvidenceSnippet(text=sentence, score=score, basis=basis))

    snippets.sort(key=lambda item: item.score, reverse=True)
    return snippets[:top_k]


def _sentences(text: str) -> list[str]:
    pieces = [re.sub(r"\s+", " ", match.group(0)).strip() for match in SENTENCE_RE.finditer(text)]
    return [piece for piece in pieces if piece]


def _features(text: str) -> dict[str, set[str]]:
    numbers = {re.sub(r"\s+", "", item.lower()) for item in NUMBER_RE.findall(text or "")}
    latin = {item.lower().strip(".") for item in LATIN_TOKEN_RE.findall(text or "") if len(item.strip(".")) >= 2}
    entities = {
        re.sub(r"\s+", " ", item).strip().lower()
        for item in CAPITALIZED_PHRASE_RE.findall(text or "")
        if item.strip()
    }
    cjk = {item for item in CJK_TOKEN_RE.findall(text or "") if len(item) >= 2}
    stopwords = {
        "the",
        "and",
        "or",
        "of",
        "in",
        "to",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "with",
        "for",
        "page",
        "source",
    }
    tokens = {token for token in latin if token not in stopwords}
    tokens.update(cjk)
    all_features = numbers | entities | tokens
    return {
        "numbers": numbers,
        "entities": entities,
        "tokens": tokens,
        "all": all_features,
    }
