from __future__ import annotations

import re
from urllib.parse import urlparse

from .schemas import SourceType

PRIMARY_DOMAINS = [
    ".gov", "sec.gov", "data.", "stats.", "statistics", "who.int", "worldbank.org",
    "imf.org", "oecd.org", "europa.eu", "federalregister.gov", "congress.gov",
]
NEWS_DOMAINS = ["reuters", "apnews", "bbc", "nytimes", "washingtonpost", "wsj", "bloomberg", "cnn"]
SCHOLAR_DOMAINS = ["doi.org", "pubmed", "ncbi.nlm.nih.gov", "nature.com", "science.org", "nejm.org", "thelancet", "arxiv.org", "semanticscholar", "openalex"]
OPINION_TERMS = ["opinion", "editorial", "commentary", "analysis", "blog", "column", "forecast", "prediction", "take"]
PRIMARY_TERMS = [
    "annual report", "10-k", "10-q", "financial statements", "official data", "official statistics",
    "statute", "regulation", "court filing", "transcript", "press release", "dataset", "data table",
]
SYNTHESIS_TERMS = ["systematic review", "meta-analysis", "guideline", "evidence review", "consensus report"]


def classify_source_type(url: str | None = None, title: str = "", text: str = "") -> SourceType:
    """Fast-path/fallback coarse source type classification.

    This function is not the officialness authority. Claim-relative source roles
    are assigned by ClaimAwareSourceRoleClassifier after source entity resolution.
    """

    haystack = " ".join([url or "", title or "", text[:4000] or ""]).lower()
    domain = urlparse(url).netloc.lower() if url else ""

    if any(term in haystack for term in ["experts say", "sources familiar", "people familiar", "anonymous source", "insiders say"]):
        return SourceType.ANONYMOUS_OR_OPAQUE

    if any(term in haystack for term in OPINION_TERMS):
        # Some news analysis pages are still not primary factual evidence for claim support.
        return SourceType.OPINION_ANALYSIS

    if any(term in haystack for term in SYNTHESIS_TERMS):
        return SourceType.EVIDENCE_SYNTHESIS

    if any(d in domain or d in haystack for d in PRIMARY_DOMAINS) or any(term in haystack for term in PRIMARY_TERMS):
        return SourceType.PRIMARY_FACT_SOURCE

    if any(d in domain for d in SCHOLAR_DOMAINS):
        return SourceType.PRIMARY_FACT_SOURCE

    if any(d in domain for d in NEWS_DOMAINS):
        return SourceType.SECONDARY_REPORTING

    if re.search(r"\b(report|paper|study|filing|dataset|transcript)\b", haystack):
        return SourceType.PRIMARY_FACT_SOURCE

    return SourceType.UNKNOWN
