from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional

from .schemas import (
    AccessStatus,
    Claim,
    ClaimType,
    EdgeBasis,
    EdgeType,
    EvidenceEdge,
    GroundingBucket,
    RiskFlag,
    Source,
    SourceType,
    SupportRelation,
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "that", "this", "it", "its",
    "is", "are", "was", "were", "be", "been", "by", "as", "from", "at", "according", "report", "study",
    "says", "said", "shows", "found", "will", "would", "could", "should", "claim", "claims",
}
CAUSAL_TERMS = {"cause", "causes", "caused", "causing", "prove", "proves", "proved", "proven", "lead", "leads", "led", "because", "due"}
CORRELATION_TERMS = {"association", "associated", "correlation", "correlated", "linked", "observed", "relationship"}
WEAK_MODAL_TERMS = {"may", "might", "could", "likely", "suggests", "appears", "possible", "potential"}
STRONG_TERMS = {"guaranteed", "proves", "definitely", "certainly", "always", "dominates", "will"}
GENERIC_QUANT_TERMS = {"fell", "fall", "rose", "rise", "grew", "growth", "increased", "decreased", "reported", "report", "says", "percent", "percentage", "billion", "million", "revenue", "amount"}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|\d+(?:\.\d+)?", text) if t.lower() not in STOPWORDS]


def _numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def _sentences(text: str) -> list[str]:
    pieces = re.findall(r"[^.!?。！？\n]+(?:[.!?。！？]|$)", text or "")
    return [p.strip() for p in pieces if p.strip()]


def _best_evidence_span(claim_text: str, source_text: str) -> tuple[str, float]:
    claim_tokens = set(_tokens(claim_text))
    if not source_text:
        return "", 0.0
    best_sentence = ""
    best_score = 0.0
    for sentence in _sentences(source_text):
        sentence_tokens = set(_tokens(sentence))
        if not sentence_tokens:
            continue
        overlap = len(claim_tokens & sentence_tokens)
        number_bonus = len(set(_numbers(claim_text)) & set(_numbers(sentence))) * 2
        score = (overlap + number_bonus) / max(len(claim_tokens), 1)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence, best_score


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms)


def _opaque_flags(source: Source | None, claim: Claim) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    mentions = " ".join(claim.source_mentions).lower()
    if source and source.source_type == SourceType.ANONYMOUS_OR_OPAQUE:
        if any(bit in mentions or bit in (source.title or "").lower() for bit in ["sources", "insider", "familiar", "close to"]):
            flags.append(RiskFlag.ANONYMOUS_SOURCE)
        else:
            flags.append(RiskFlag.VAGUE_SOURCE)
    if any(bit in mentions for bit in ["experts", "sources", "insider", "familiar", "close to"]):
        flag = RiskFlag.ANONYMOUS_SOURCE if "sources" in mentions or "insider" in mentions else RiskFlag.VAGUE_SOURCE
        if flag not in flags:
            flags.append(flag)
    return flags


def bucket_for(source: Source | None, relation: SupportRelation) -> GroundingBucket:
    if source is None:
        return GroundingBucket.UNVERIFIABLE_OR_MISMATCH
    if source.access_status in {AccessStatus.FAILED, AccessStatus.UNAVAILABLE} and source.source_type != SourceType.ANONYMOUS_OR_OPAQUE:
        return GroundingBucket.UNVERIFIABLE_OR_MISMATCH
    if source.source_type == SourceType.ANONYMOUS_OR_OPAQUE:
        return GroundingBucket.UNVERIFIABLE_OR_MISMATCH
    if relation == SupportRelation.DIRECTLY_SUPPORTS and source.source_type in {SourceType.PRIMARY_FACT_SOURCE, SourceType.EVIDENCE_SYNTHESIS}:
        return GroundingBucket.HARD_FACT_GROUNDING
    if relation in {SupportRelation.PARTIALLY_SUPPORTS, SupportRelation.SUPPORTS_WEAKER_CLAIM, SupportRelation.DIRECTLY_SUPPORTS} and source.source_type in {
        SourceType.PRIMARY_FACT_SOURCE,
        SourceType.EVIDENCE_SYNTHESIS,
        SourceType.SECONDARY_REPORTING,
        SourceType.UNKNOWN,
    }:
        return GroundingBucket.WEAK_FACT_GROUNDING
    if relation in {SupportRelation.ATTRIBUTION_ONLY, SupportRelation.OPINION_ONLY} or source.source_type == SourceType.OPINION_ANALYSIS:
        return GroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING
    return GroundingBucket.UNVERIFIABLE_OR_MISMATCH


class SupportChecker:
    def check(
        self,
        claim: Claim,
        source: Source | None,
        *,
        edge_type: EdgeType = EdgeType.AUTHOR_CITED,
        basis: EdgeBasis = EdgeBasis.NONE,
    ) -> tuple[EvidenceEdge, list[RiskFlag]]:
        flags: list[RiskFlag] = []

        if claim.claim_type == ClaimType.NON_CLAIM:
            edge = EvidenceEdge(
                claim_id=claim.claim_id,
                source_id=source.source_id if source else None,
                edge_type=edge_type,
                basis=basis,
                support_relation=SupportRelation.BACKGROUND_ONLY,
                evidence_span="",
                reasoning_summary="Non-claim text is excluded from fact-support ratios.",
                final_bucket=GroundingBucket.UNVERIFIABLE_OR_MISMATCH,
            )
            return edge, flags

        if source is None:
            flags.append(RiskFlag.INACCESSIBLE_SOURCE)
            if claim.has_quantitative_data:
                flags.append(RiskFlag.QUANTITATIVE_CLAIM_WITHOUT_PRIMARY_DATA)
            edge = EvidenceEdge(
                claim_id=claim.claim_id,
                source_id=None,
                edge_type=edge_type,
                basis=basis,
                support_relation=SupportRelation.INACCESSIBLE,
                evidence_span="",
                reasoning_summary="No explicit or supplied source was found for this claim.",
                final_bucket=GroundingBucket.UNVERIFIABLE_OR_MISMATCH,
            )
            return edge, flags

        flags.extend(_opaque_flags(source, claim))

        if source.access_status in {AccessStatus.UNAVAILABLE, AccessStatus.FAILED} and not source.extracted_text:
            flags.append(RiskFlag.INACCESSIBLE_SOURCE)
            if claim.has_quantitative_data:
                flags.append(RiskFlag.QUANTITATIVE_CLAIM_WITHOUT_PRIMARY_DATA)
            relation = SupportRelation.INACCESSIBLE
            bucket = bucket_for(source, relation)
            edge = EvidenceEdge(
                claim_id=claim.claim_id,
                source_id=source.source_id,
                edge_type=edge_type,
                basis=basis,
                support_relation=relation,
                evidence_span="",
                reasoning_summary="A source was referenced, but its content is not available in this run.",
                final_bucket=bucket,
            )
            return edge, flags

        if source.source_type == SourceType.ANONYMOUS_OR_OPAQUE:
            relation = SupportRelation.INACCESSIBLE
            bucket = bucket_for(source, relation)
            edge = EvidenceEdge(
                claim_id=claim.claim_id,
                source_id=source.source_id,
                edge_type=edge_type,
                basis=basis,
                support_relation=relation,
                evidence_span=source.extracted_text_preview,
                reasoning_summary="The claim relies on an unnamed, vague, or opaque source mention, so it is not publicly auditable.",
                final_bucket=bucket,
            )
            return edge, flags

        source_text = source.extracted_text or source.extracted_text_preview or source.title or ""
        evidence_span, overlap_score = _best_evidence_span(claim.normalized_claim, source_text)
        claim_lower = claim.normalized_claim.lower()
        evidence_lower = evidence_span.lower()

        relation = SupportRelation.BACKGROUND_ONLY
        summary = "The source is topically related but does not clearly support the claim."

        claim_nums = set(_numbers(claim.normalized_claim))
        source_nums = set(_numbers(evidence_span or source_text))
        claim_salient = {t for t in _tokens(claim.normalized_claim) if not t.replace('.', '', 1).isdigit() and t not in GENERIC_QUANT_TERMS}
        source_salient = {t for t in _tokens(evidence_span or source_text) if not t.replace('.', '', 1).isdigit() and t not in GENERIC_QUANT_TERMS}
        salient_overlap = len(claim_salient & source_salient) / max(len(claim_salient), 1)
        if claim_nums and not (claim_nums & source_nums):
            relation = SupportRelation.NO_SUPPORT
            flags.append(RiskFlag.SOURCE_CLAIM_MISMATCH)
            if claim.has_quantitative_data:
                flags.append(RiskFlag.QUANTITATIVE_CLAIM_WITHOUT_PRIMARY_DATA)
            summary = "The claim contains quantitative information that does not appear in the selected source text."
        elif claim_nums and claim_salient and salient_overlap < 0.6:
            relation = SupportRelation.NO_SUPPORT
            flags.append(RiskFlag.SOURCE_CLAIM_MISMATCH)
            summary = "The source contains a matching number but appears to attach it to a different subject or object."
        elif source.source_type == SourceType.OPINION_ANALYSIS:
            relation = SupportRelation.OPINION_ONLY
            flags.append(RiskFlag.OPINION_USED_AS_FACT)
            if _contains_any(claim_lower, STRONG_TERMS) and _contains_any(evidence_lower or source_text.lower(), WEAK_MODAL_TERMS):
                flags.append(RiskFlag.SOURCE_ONLY_SUPPORTS_WEAKER_CLAIM)
            summary = "The source is classified as opinion or analysis rather than direct factual evidence."
        elif _contains_any(claim_lower, CAUSAL_TERMS) and _contains_any(evidence_lower or source_text.lower(), CORRELATION_TERMS):
            relation = SupportRelation.SUPPORTS_WEAKER_CLAIM
            flags.append(RiskFlag.CORRELATION_PRESENTED_AS_CAUSATION)
            flags.append(RiskFlag.SOURCE_ONLY_SUPPORTS_WEAKER_CLAIM)
            summary = "The source appears to discuss association or correlation, while the claim uses stronger causal language."
        elif _contains_any(claim_lower, STRONG_TERMS) and _contains_any(evidence_lower or source_text.lower(), WEAK_MODAL_TERMS):
            relation = SupportRelation.SUPPORTS_WEAKER_CLAIM
            flags.append(RiskFlag.SOURCE_ONLY_SUPPORTS_WEAKER_CLAIM)
            summary = "The source uses weaker or probabilistic language than the claim."
        elif overlap_score <= 0.12:
            relation = SupportRelation.NO_SUPPORT
            flags.append(RiskFlag.SOURCE_CLAIM_MISMATCH)
            summary = "The source text does not contain enough overlapping evidence to support the claim."
        elif claim.claim_type == ClaimType.ATTRIBUTION and overlap_score >= 0.8:
            relation = SupportRelation.ATTRIBUTION_ONLY
            summary = "The source can support that a person, report, study, or institution made a statement, but this is not the same as proving the statement true."
        elif claim.claim_type == ClaimType.ATTRIBUTION:
            relation = SupportRelation.NO_SUPPORT
            flags.append(RiskFlag.SOURCE_CLAIM_MISMATCH)
            summary = "The source exists, but it does not clearly match the attribution claim."
        elif overlap_score >= 0.68:
            relation = SupportRelation.DIRECTLY_SUPPORTS
            summary = "The source text contains the core terms or numbers needed to support the claim."
        elif overlap_score >= 0.35:
            relation = SupportRelation.PARTIALLY_SUPPORTS
            summary = "The source text supports part of the claim, but the match is incomplete or indirect."

        if source.source_type == SourceType.SECONDARY_REPORTING and relation in {SupportRelation.DIRECTLY_SUPPORTS, SupportRelation.PARTIALLY_SUPPORTS}:
            flags.append(RiskFlag.SECONDARY_SOURCE_ONLY)
            if relation == SupportRelation.DIRECTLY_SUPPORTS:
                # A secondary article can directly state something, but it is still weak fact grounding.
                relation = SupportRelation.PARTIALLY_SUPPORTS
                summary = "The source states the claim, but it is secondary reporting rather than an upstream factual source."

        bucket = bucket_for(source, relation)
        edge = EvidenceEdge(
            claim_id=claim.claim_id,
            source_id=source.source_id,
            edge_type=edge_type,
            basis=basis,
            support_relation=relation,
            evidence_span=evidence_span,
            reasoning_summary=summary,
            final_bucket=bucket,
        )
        return edge, list(dict.fromkeys(flags))
