from __future__ import annotations

import re
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field

from .evidence_snippet_retriever import EvidenceSnippet

BILINGUAL_FINANCE_TERMS = {
    "股票": ["stock", "stocks", "equity", "security"],
    "股东": ["stockholder", "stockholders", "shareholder", "shareholders"],
    "公司所有权": ["ownership", "company", "share"],
    "所有权": ["ownership", "share"],
    "资本增值": ["capital", "appreciation"],
    "分红": ["dividend", "dividends", "payments"],
    "股息": ["dividend", "dividends", "payments"],
    "利润": ["profit", "earnings"],
    "普通股": ["common", "stock", "stockholders"],
    "优先股": ["preferred", "stock", "stockholders"],
    "债权人": ["bondholders", "creditors"],
    "破产": ["bankrupt", "liquidated", "bankruptcy"],
    "清算": ["liquidated", "liquidation"],
    "剩余资产": ["proceeds", "left"],
    "成交价格": ["last-traded", "price", "execution"],
    "市价单": ["market", "order"],
    "限价单": ["limit", "order"],
    "未来收益": ["future", "payoffs", "income"],
    "折现": ["discounted", "value"],
    "风险": ["risk", "premium", "compensation"],
    "估值": ["asset", "valuations", "discounted", "payoffs"],
}


class SemanticSupportHint(str, Enum):
    DIRECT_FACT_SUPPORT = "direct_fact_support"
    FACT_PREMISE_SUPPORT = "fact_premise_support"
    PARTIAL_OR_NUANCED_SUPPORT = "partial_or_nuanced_support"
    OPINION_ONLY = "opinion_only"
    NO_SUPPORT = "no_support"


class SemanticSnippetRerankResponse(BaseModel):
    selected_snippet_indexes: list[int] = Field(default_factory=list)
    support_hint: SemanticSupportHint = SemanticSupportHint.NO_SUPPORT
    short_reason: str = ""


SEMANTIC_SNIPPET_RERANKER_SYSTEM_PROMPT = """You are selecting evidence snippets for a cited statement.

Rules:
1. Select only from candidate_snippets by index.
2. Do not invent source text.
3. Prefer snippets that directly support the cited statement.
4. If the statement is explanatory or interpretive, select snippets that support its factual premise.
5. Output only JSON matching the schema."""


def rerank_candidate_snippets(
    *,
    cited_text: str,
    source_title: str = "",
    source_pointer_description: str = "",
    candidate_snippets: list[EvidenceSnippet],
) -> SemanticSnippetRerankResponse:
    """Schema-compatible local reranker.

    Production callers can replace this with an LLM call using the same schema.
    The local implementation is deliberately conservative: it only selects
    existing candidate snippets using expanded query overlap, never generated text.
    """

    if not candidate_snippets:
        return SemanticSnippetRerankResponse(short_reason="No candidate snippets were supplied.")
    query = _token_set(" ".join([cited_text, source_title, source_pointer_description]))
    if not query:
        return SemanticSnippetRerankResponse(short_reason="No usable query features were available.")

    scored: list[tuple[int, int]] = []
    for index, snippet in enumerate(candidate_snippets):
        snippet_tokens = _token_set(snippet.text)
        overlap = query & snippet_tokens
        if overlap:
            scored.append((index, len(overlap)))
    if not scored:
        return SemanticSnippetRerankResponse(short_reason="Candidate snippets did not overlap with expanded query features.")

    selected = [index for index, _score in sorted(scored, key=lambda item: item[1], reverse=True)[:5]]
    support_hint = SemanticSupportHint.DIRECT_FACT_SUPPORT
    lowered = (cited_text or "").lower()
    if any(term in lowered for term in ["影响", "估值", "风险", "未来", "本质", "可能", "合理", "判断"]):
        support_hint = SemanticSupportHint.FACT_PREMISE_SUPPORT
    if any(term in lowered for term in ["通常", "不一定", "可能", "最新股价"]):
        support_hint = SemanticSupportHint.PARTIAL_OR_NUANCED_SUPPORT
    return SemanticSnippetRerankResponse(
        selected_snippet_indexes=selected,
        support_hint=support_hint,
        short_reason="Selected existing snippets with expanded bilingual finance feature overlap.",
    )


def _token_set(text: str) -> set[str]:
    normalized = (text or "").lower()
    tokens = {token for token in re.findall(r"[a-z][a-z0-9-]{2,}", normalized)}
    for zh_term, english_terms in BILINGUAL_FINANCE_TERMS.items():
        if zh_term in text:
            tokens.update(english_terms)
    tokens.update(term for term in _finance_terms(normalized))
    return tokens


def _finance_terms(text: str) -> Iterable[str]:
    for term in [
        "stock",
        "stocks",
        "stockholder",
        "stockholders",
        "shareholder",
        "shareholders",
        "ownership",
        "company",
        "capital",
        "appreciation",
        "dividend",
        "dividends",
        "profit",
        "earnings",
        "common",
        "preferred",
        "bondholders",
        "creditors",
        "bankrupt",
        "liquidated",
        "left",
        "market",
        "order",
        "last-traded",
        "execution",
        "discounted",
        "payoffs",
        "risk",
        "return",
    ]:
        if term in text:
            yield term
