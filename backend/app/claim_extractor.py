from __future__ import annotations

import re
from typing import Iterable, List, Protocol

from .citation_parser import extract_source_mentions
from .schemas import Claim, ClaimType, ImportanceLabel

SENTENCE_RE = re.compile(r".+?(?:[.!?。！？](?=\s|$)|$)", re.MULTILINE)
URL_RE = re.compile(r"https?://[^\s\])}>\"']+", re.IGNORECASE)
CITATION_MARK_RE = re.compile(r"\[[^\]]+\](?:\([^\)]+\))?|\[\^?\d+\]")
QUANT_RE = re.compile(
    r"(\$|€|£)?\s*\d+(?:\.\d+)?\s*(?:%|percent|percentage|billion|million|trillion|thousand|bn|mn|k|x|times|rank|#|No\.?\s*\d+)?|\b\d{4}\b",
    re.IGNORECASE,
)

ATTRIBUTION_RE = re.compile(
    r"\b(according to|said|stated|says|claims|claimed|found|shows|showed|concludes|wrote|argues|experts say|sources say|a study|the study|a report|the report|court filing)\b",
    re.IGNORECASE,
)
JUDGMENT_RE = re.compile(
    r"\b(should|would|will|could|might|likely|guaranteed|dominate|damage|harm|benefit|fail(?:ed)?|better|worse|best|worst|causes?|caused|lead(?:s)? to|because|therefore|suggests|implies|predicts?|forecast|strategy|recommend|worth|important|significant|proves?)\b",
    re.IGNORECASE,
)
NON_CLAIM_RE = re.compile(
    r"^(overall|in conclusion|to summarize|summary|here are|as follows|that said|however|first|second|third|finally|note)\b[:：]?\s*$",
    re.IGNORECASE,
)


def _strip_citations(text: str) -> str:
    # Remove citation wrappers before raw URLs, otherwise markdown links become dangling "[label]()".
    text = CITATION_MARK_RE.sub("", text)
    text = URL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" -–—\t\r\n")


def _normalize_clause(text: str) -> str:
    text = _strip_citations(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;:。.!?！？")


def _is_non_claim(text: str) -> bool:
    clean = _normalize_clause(text)
    if not clean:
        return True
    if len(clean.split()) <= 2:
        return True
    if NON_CLAIM_RE.match(clean):
        return True
    # Pure headings are usually not auditable claims.
    if len(clean.split()) <= 6 and clean.endswith(":"):
        return True
    return False


def _split_sentence_into_atomic_clauses(sentence: str) -> list[str]:
    """Heuristic atomic claim splitting.

    This intentionally stays conservative. It handles obvious multi-claim structures and
    common AI-answer patterns without pretending to do full semantic decomposition.
    """

    sentence = sentence.strip()
    if not sentence:
        return []

    # Break numbered/list items inside one line.
    sentence = re.sub(r"\s+(?:\d+\.|[•*-])\s+", ". ", sentence)

    clauses: list[str] = []

    # Special case: "X supports text, audio and image inputs" -> three facts.
    support_match = re.search(
        r"(?P<subject>\b[A-Z][A-Za-z0-9\- ]{1,60}?)\s+(?P<verb>supports?|allows?|accepts?|handles?)\s+(?P<items>[^.?!。！？]+?)\s+(?P<object>inputs?|modalities|formats|files|features)\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if support_match and "," in support_match.group("items"):
        subject = support_match.group("subject").strip()
        verb = support_match.group("verb").strip()
        items_text = support_match.group("items")
        obj = support_match.group("object")
        items = re.split(r",|\band\b|\bor\b", items_text, flags=re.IGNORECASE)
        items = [i.strip(" ,") for i in items if i.strip(" ,")]
        if 2 <= len(items) <= 6:
            for item in items:
                clauses.append(f"{subject} {verb} {item} {obj}")
            remainder = sentence.replace(support_match.group(0), "").strip(" ,.;")
            if remainder and len(remainder.split()) > 3:
                clauses.append(remainder)
            return clauses

    # Split on semicolons and obvious independent conjunctions.
    chunks = re.split(r";|\s+and\s+(?=(?:it|they|he|she|the|a|an|[A-Z][A-Za-z0-9_\-]+)\b)", sentence, flags=re.IGNORECASE)
    for chunk in chunks:
        chunk = chunk.strip(" ,;\n\t")
        if chunk:
            clauses.append(chunk)

    return clauses or [sentence]


def _claim_type_for_text(text: str) -> ClaimType:
    clean = _normalize_clause(text)
    lower = clean.lower()
    if _is_non_claim(clean):
        return ClaimType.NON_CLAIM
    has_attribution = bool(ATTRIBUTION_RE.search(clean))
    has_judgment = bool(JUDGMENT_RE.search(clean))
    if has_attribution and has_judgment:
        # If the attributed content itself is predictive, causal, or evaluative, expose it
        # as a judgment claim. The source mention remains available in source_mentions.
        return ClaimType.JUDGMENT
    if has_attribution:
        return ClaimType.ATTRIBUTION
    if has_judgment:
        return ClaimType.JUDGMENT
    # Concrete dates, named entities, and numbers are usually factual claims.
    if QUANT_RE.search(clean) or re.search(r"\b(is|are|was|were|has|have|released|launched|published|located|owns|operates)\b", clean, re.IGNORECASE):
        return ClaimType.FACTUAL
    return ClaimType.FACTUAL


def _importance_for_text(text: str, claim_type: ClaimType) -> ImportanceLabel:
    clean = _normalize_clause(text).lower()
    if claim_type == ClaimType.NON_CLAIM:
        return ImportanceLabel.BACKGROUND
    if claim_type == ClaimType.JUDGMENT or any(word in clean for word in ["therefore", "conclusion", "guaranteed", "should", "because", "dominates", "damage"]):
        return ImportanceLabel.CORE
    if claim_type == ClaimType.FACTUAL and QUANT_RE.search(clean):
        return ImportanceLabel.SUPPORTING
    return ImportanceLabel.SUPPORTING


class ClaimExtractor(Protocol):
    def extract_claims(self, input_text: str, original_question: str | None = None) -> list[Claim]:
        ...


class RuleBasedClaimExtractor:
    """Rule-based extractor used by focused parser tests.

    It is intentionally transparent and imperfect. Runtime analysis uses an LLM
    provider that returns the same schema.
    """

    def extract_claims(self, input_text: str, original_question: str | None = None) -> list[Claim]:
        claims: list[Claim] = []
        for sentence_match in SENTENCE_RE.finditer(input_text):
            sentence = sentence_match.group(0).strip()
            if not sentence:
                continue
            clauses = _split_sentence_into_atomic_clauses(sentence)
            for clause in clauses:
                normalized = _normalize_clause(clause)
                if not normalized:
                    continue
                claim_type = _claim_type_for_text(normalized)
                claim = Claim(
                    claim_id=f"c{len(claims)+1:03d}",
                    original_text_span=clause.strip(),
                    normalized_claim=normalized,
                    claim_type=claim_type,
                    has_quantitative_data=bool(QUANT_RE.search(normalized)),
                    source_mentions=extract_source_mentions(normalized),
                    importance_label=_importance_for_text(normalized, claim_type),
                )
                claims.append(claim)
        if not claims and input_text.strip():
            normalized = _normalize_clause(input_text)
            claims.append(
                Claim(
                    claim_id="c001",
                    original_text_span=input_text.strip(),
                    normalized_claim=normalized,
                    claim_type=_claim_type_for_text(normalized),
                    has_quantitative_data=bool(QUANT_RE.search(normalized)),
                    source_mentions=extract_source_mentions(normalized),
                    importance_label=ImportanceLabel.SUPPORTING,
                )
            )
        return claims
