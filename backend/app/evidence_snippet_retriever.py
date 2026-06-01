from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSnippet:
    text: str
    score: float
    basis: list[str]


@dataclass(frozen=True)
class SnippetRetrievalResult:
    snippets: list[EvidenceSnippet]
    failure_reason: str = ""


SENTENCE_RE = re.compile(r"[^。！？.!?\n]+(?:[。！？.!?]+|$)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|:1|人|名|个|所|位|million|billion|%)?", re.IGNORECASE)
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.-]{1,}")
CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][A-Za-z&.-]+(?:\s+|$)){1,5}")
CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
MIN_SCORE = 1.0


def retrieve_evidence_snippets(cited_text: str, source_text: str, *, top_k: int = 5) -> list[EvidenceSnippet]:
    return retrieve_evidence_snippets_with_reason(cited_text, source_text, top_k=top_k).snippets


def retrieve_evidence_snippets_with_reason(
    cited_text: str,
    source_text: str,
    *,
    top_k: int = 5,
) -> SnippetRetrievalResult:
    query = _features(cited_text)
    cleaned_source = _clean_source_text(source_text)
    if not cleaned_source.strip():
        return SnippetRetrievalResult([], "no_source_text")
    if not query["all"]:
        return SnippetRetrievalResult([], "no_feature_overlap")

    snippets: list[EvidenceSnippet] = []
    low_score_candidates = 0
    for sentence in _sentences(cleaned_source):
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

        thematic_overlap = query["themes"] & sentence_features["themes"]
        if thematic_overlap:
            score += 3.0 * len(thematic_overlap)
            basis.append(f"theme overlap: {', '.join(sorted(thematic_overlap)[:8])}")

        if query["holdings_intent"] and sentence_features["holdings_terms"]:
            score += 4.0
            basis.append("holdings context")

        if query["fee_intent"] and sentence_features["fee_terms"]:
            score += 3.0
            basis.append("fee context")

        if score >= MIN_SCORE:
            snippets.append(EvidenceSnippet(text=sentence, score=score, basis=basis))
        elif score > 0:
            low_score_candidates += 1

    snippets.sort(key=lambda item: item.score, reverse=True)
    if snippets:
        return SnippetRetrievalResult(snippets[:top_k])
    if low_score_candidates:
        return SnippetRetrievalResult([], "candidate_snippets_low_score")
    return SnippetRetrievalResult([], "no_feature_overlap")


def _sentences(text: str) -> list[str]:
    pieces = [re.sub(r"\s+", " ", match.group(0)).strip() for match in SENTENCE_RE.finditer(text)]
    expanded: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        words = piece.split()
        if len(piece) > 700 and len(words) > 80:
            window = 80
            stride = 40
            for start in range(0, len(words), stride):
                chunk = " ".join(words[start : start + window]).strip()
                if chunk:
                    expanded.append(chunk)
                if start + window >= len(words):
                    break
        else:
            expanded.append(piece)
    return [piece for piece in expanded if piece]


def _features(text: str) -> dict[str, set[str]]:
    normalized_text = text or ""
    normalized_lower = normalized_text.lower()
    numbers = {re.sub(r"\s+", "", item.lower()) for item in NUMBER_RE.findall(normalized_text)}
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
    themes = _themes(normalized_lower)
    holdings_intent = _has_any(normalized_lower, {"前十大", "持仓", "权重", "合计", "接近", "集中度", "top holdings", "holdings", "weight"})
    fee_intent = _has_any(normalized_lower, {"费用率", "费率", "expense ratio", "management fee", "0.68"})
    holdings_terms = _has_any(normalized_lower, {"holdings", "weight", "top", "keyence", "fanuc", "abb", "nvidia", "intuitive", "yaskawa", "smc", "daifuku", "inovance"})
    fee_terms = _has_any(normalized_lower, {"expense ratio", "management fee", "gross expense", "net expense", "0.68", "费用率"})
    all_features = numbers | entities | tokens | themes
    return {
        "numbers": numbers,
        "entities": entities,
        "tokens": tokens,
        "themes": themes,
        "holdings_intent": {"holdings_intent"} if holdings_intent else set(),
        "fee_intent": {"fee_intent"} if fee_intent else set(),
        "holdings_terms": {"holdings_terms"} if holdings_terms else set(),
        "fee_terms": {"fee_terms"} if fee_terms else set(),
        "all": all_features,
    }


def _themes(text: str) -> set[str]:
    theme_map = {
        "holdings": {"前十大", "持仓", "权重", "集中度", "top holdings", "holdings", "weight", "keyence", "fanuc", "abb", "nvidia"},
        "fees": {"费用率", "费率", "expense ratio", "management fee", "0.68"},
        "country": {"国家", "美国", "日本", "中国", "瑞士", "country", "geographic", "japan", "china", "switzerland"},
        "risk": {"风险", "波动", "beta", "standard deviation", "volatility", "标准差"},
    }
    return {theme for theme, needles in theme_map.items() if _has_any(text, needles)}


def _has_any(text: str, needles: set[str]) -> bool:
    return any(needle in text for needle in needles)


def _clean_source_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    noisy_phrases = [
        "Skip to content",
        "Search",
        "Menu",
        "Privacy Policy",
        "Terms of Use",
        "Cookie",
        "Subscribe",
    ]
    for phrase in noisy_phrases:
        cleaned = cleaned.replace(phrase, " ")
    return re.sub(r"\s+", " ", cleaned).strip()
