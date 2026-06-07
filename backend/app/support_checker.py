from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from .evidence_snippet_retriever import retrieve_evidence_snippets_with_reason
from .providers.llm_provider import LLMProvider, LLMProviderError, LLMProviderTimeoutError
from .providers.llm_provider import CancellationToken
from .semantic_snippet_reranker import rerank_candidate_snippets
from .schemas import (
    AccessStatus,
    Claim,
    ClaimType,
    ClaimReviewCategory,
    EdgeBasis,
    EdgeType,
    EvidenceEdge,
    EvidenceExcerpt,
    FinalGroundingBucket,
    OfficialnessStatus,
    RiskFlag,
    Source,
    SourceRole,
    SourceType,
    SupportRelation,
)

MAX_SOURCE_TEXT_CHARS = int(os.environ.get("SOURCE_GROUNDING_MAX_SOURCE_TEXT_CHARS") or "6000")


@dataclass
class SupportCheckInput:
    claim: Claim
    source: Source | None
    edge_type: EdgeType = EdgeType.AUTHOR_CITED
    basis: EdgeBasis = EdgeBasis.NONE


def bucket_for(source: Source | None, relation: SupportRelation) -> FinalGroundingBucket:
    """Non-semantic fallback used only when source text is unavailable."""

    if relation == SupportRelation.INACCESSIBLE:
        return FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH
    if relation == SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET:
        return FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH
    if relation == SupportRelation.BACKGROUND_ONLY:
        return FinalGroundingBucket.EXCLUDED_OR_CONTEXT
    if source is None:
        return FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH
    if relation == SupportRelation.DIRECTLY_SUPPORTS and source.source_type == SourceType.PRIMARY_FACT_SOURCE:
        return FinalGroundingBucket.HARD_FACT_GROUNDING
    if relation in {
        SupportRelation.DIRECTLY_SUPPORTS,
        SupportRelation.PARTIALLY_SUPPORTS,
        SupportRelation.SUPPORTS_WEAKER_CLAIM,
    }:
        return FinalGroundingBucket.WEAK_FACT_GROUNDING
    if relation in {SupportRelation.ATTRIBUTION_ONLY, SupportRelation.OPINION_ONLY}:
        return FinalGroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING
    return FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH


class SupportChecker:
    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider

    def check(
        self,
        claim: Claim,
        source: Source | None,
        *,
        edge_type: EdgeType = EdgeType.AUTHOR_CITED,
        basis: EdgeBasis = EdgeBasis.NONE,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[EvidenceEdge, list[RiskFlag]]:
        return self.check_many(
            [
                SupportCheckInput(
                    claim=claim,
                    source=source,
                    edge_type=edge_type,
                    basis=basis,
                )
            ],
            cancellation_token=cancellation_token,
        )[0]

    def check_many(
        self,
        checks: list[SupportCheckInput],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[tuple[EvidenceEdge, list[RiskFlag]]]:
        results: list[tuple[EvidenceEdge, list[RiskFlag]] | None] = [None] * len(checks)
        llm_payloads: list[dict[str, Any]] = []
        llm_indexes: list[int] = []
        for index, item in enumerate(checks):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            fallback = self._fallback_check(
                item.claim,
                item.source,
                edge_type=item.edge_type,
                basis=item.basis,
            )
            if fallback is not None:
                results[index] = fallback
                continue
            if self.provider is None:
                raise RuntimeError("SupportChecker requires an LLMProvider when source body is available.")
            source_bundle = self._source_bundle(item.source, item.claim)
            if source_bundle.get("snippet_retrieval_status") == "no_relevant_snippet":
                failure_reason = source_bundle.get("snippet_failure_reason") or "no_relevant_snippet"
                updated = self._mark_no_relevant_snippet(
                    item.claim,
                    source_id=source_bundle.get("source_id"),
                    reasoning=(
                        "Source body was available, but no relevant evidence snippet was retrieved "
                        f"for this cited text. Snippet failure reason: {failure_reason}."
                    ),
                )
                results[index] = (
                    self._edge_from_claim(updated, item.source, edge_type=item.edge_type, basis=item.basis),
                    list(updated.risk_flags),
                )
                continue
            llm_indexes.append(index)
            llm_payloads.append(
                {
                    "check_id": f"sc{index+1:04d}",
                    "claim": item.claim,
                    "source_bundle": source_bundle,
                }
            )

        if llm_payloads:
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            try:
                updated_claims = _run_async(
                    self.provider.check_claim_supports(
                        llm_payloads,
                        cancellation_token=cancellation_token,
                    )
                )
            except LLMProviderTimeoutError as exc:
                updated_claims = [
                    self._fallback_from_retrieved_snippets(
                        item["claim"],
                        item["source_bundle"],
                        reasoning=f"Source support check timed out in this run: {exc}",
                    )
                    for item in llm_payloads
                ]
            except LLMProviderError as exc:
                updated_claims = [
                    self._fallback_from_retrieved_snippets(
                        item["claim"],
                        item["source_bundle"],
                        reasoning=f"Source support check could not be completed by the LLM provider in this run: {exc}",
                    )
                    for item in llm_payloads
                ]
            for index, updated, payload in zip(llm_indexes, updated_claims, llm_payloads):
                item = checks[index]
                updated = self._attach_evidence_excerpts(updated, payload["source_bundle"])
                results[index] = (
                    self._edge_from_claim(updated, item.source, edge_type=item.edge_type, basis=item.basis),
                    list(updated.risk_flags),
                )

        return [result for result in results if result is not None]

    def _fallback_check(
        self,
        claim: Claim,
        source: Source | None,
        *,
        edge_type: EdgeType,
        basis: EdgeBasis,
    ) -> tuple[EvidenceEdge, list[RiskFlag]] | None:
        if (
            claim.review_category
            and claim.review_category.value == "audit_limited"
            and claim.support_relation == SupportRelation.INACCESSIBLE
        ):
            return self._edge_from_claim(claim, source, edge_type=edge_type, basis=basis), list(claim.risk_flags)
        if claim.claim_type == ClaimType.NON_CLAIM:
            updated = claim.model_copy(deep=True)
            updated.support_relation = SupportRelation.BACKGROUND_ONLY
            updated.final_bucket = FinalGroundingBucket.EXCLUDED_OR_CONTEXT
            return self._edge_from_claim(
                updated,
                source,
                edge_type=edge_type,
                basis=basis,
                reasoning="Non-claim or contextual text is excluded from source support checking.",
            ), list(updated.risk_flags)

        if source is None:
            updated = self._mark_inaccessible(
                claim,
                source_id=None,
                reasoning="No explicit, discovered, or supplied source body was available for this claim.",
            )
            return self._edge_from_claim(updated, None, edge_type=edge_type, basis=basis), list(updated.risk_flags)

        source_text = source.extracted_text or ""
        if (
            not source_text.strip()
            or source.access_status in {AccessStatus.UNAVAILABLE, AccessStatus.FAILED, AccessStatus.PAYWALLED}
            or source.source_type == SourceType.ANONYMOUS_OR_OPAQUE
        ):
            updated = self._mark_inaccessible(
                claim,
                source_id=source.source_id,
                reasoning="A source was referenced, but source text was not available for support checking in this run.",
            )
            preview = source.extracted_text_preview or source.title or ""
            updated.evidence_quote = preview[:300]
            return self._edge_from_claim(updated, source, edge_type=edge_type, basis=basis), list(updated.risk_flags)

        return None

    def _source_bundle(self, source: Source | None, claim: Claim) -> dict[str, Any]:
        if source is None:
            return {}
        source_text = source.extracted_text or ""
        source_pointer_description = claim.source_registry_entry or ""
        retrieval = retrieve_evidence_snippets_with_reason(
            " ".join(part for part in [claim.original_text_span, claim.normalized_claim] if part),
            source_text,
            source_pointer_description=source_pointer_description,
            source_title=source.title,
            source_url=source.url or "",
        )
        snippets = retrieval.snippets
        snippet_status = retrieval.status
        semantic_rerank = None
        if snippet_status == "semantic_rerank_needed" and snippets:
            semantic_rerank = rerank_candidate_snippets(
                cited_text=" ".join(part for part in [claim.original_text_span, claim.normalized_claim] if part),
                source_title=source.title,
                source_pointer_description=source_pointer_description,
                candidate_snippets=snippets,
            )
            if semantic_rerank.selected_snippet_indexes:
                snippets = [
                    snippets[index]
                    for index in semantic_rerank.selected_snippet_indexes
                    if 0 <= index < len(snippets)
                ]
                snippet_status = "semantic_match"
        snippet_text = "\n".join(snippet.text for snippet in snippets)
        return {
            "source_id": source.source_id,
            "url": source.url,
            "title": source.title,
            "publisher_or_author": source.publisher_or_author,
            "publication_date": source.publication_date,
            "access_status": source.access_status.value,
            "source_type": source.source_type.value,
            "source_entity": source.source_entity,
            "registrable_domain": source.registrable_domain,
            "organization_type": source.organization_type.value,
            "entity_aliases": source.entity_aliases,
            "officialness_status": source.officialness_status.value,
            "officialness_basis": source.officialness_basis,
            "evidence_snippets": [
                {"text": snippet.text, "score": snippet.score, "basis": snippet.basis}
                for snippet in snippets
            ],
            "snippet_retrieval_status": snippet_status if snippets else "no_relevant_snippet",
            "snippet_failure_reason": retrieval.failure_reason,
            "snippet_retrieval_query": retrieval.retrieval_query,
            "semantic_rerank": semantic_rerank.model_dump(mode="json") if semantic_rerank else None,
            "extracted_text": snippet_text[:MAX_SOURCE_TEXT_CHARS],
            "extracted_text_preview": source.extracted_text_preview,
        }

    def _attach_evidence_excerpts(self, claim: Claim, source_bundle: dict[str, Any]) -> Claim:
        snippets = source_bundle.get("evidence_snippets") or []
        if not snippets:
            return claim
        updated = claim.model_copy(deep=True)
        semantic = source_bundle.get("semantic_rerank") or {}
        support_hint = semantic.get("support_hint") or ""
        selection_method = "semantic_reranker" if semantic else "lexical_retrieval"
        excerpts: list[EvidenceExcerpt] = []
        for index, snippet in enumerate(snippets[:3], start=1):
            text = str(snippet.get("text") or "").strip()
            if not text:
                continue
            excerpts.append(
                EvidenceExcerpt(
                    excerpt_id=f"{updated.claim_id}:excerpt:{index}",
                    source_id=source_bundle.get("source_id"),
                    source_title=source_bundle.get("title") or "",
                    source_url=source_bundle.get("url"),
                    text=text,
                    char_start=snippet.get("char_start"),
                    char_end=snippet.get("char_end"),
                    excerpt_role=_excerpt_role_for(updated, support_hint=support_hint),
                    selection_method=selection_method,
                    confidence="high" if index == 1 and selection_method == "semantic_reranker" else "medium",
                    explanation=_excerpt_explanation(updated, support_hint=support_hint),
                )
            )
        updated.evidence_excerpts = excerpts
        updated.best_evidence_excerpt = excerpts[0] if excerpts else None
        return updated

    def _fallback_from_retrieved_snippets(
        self,
        claim: Claim,
        source_bundle: dict[str, Any],
        *,
        reasoning: str,
    ) -> Claim:
        snippets = source_bundle.get("evidence_snippets") or []
        if not snippets:
            return self._mark_inaccessible(
                claim,
                source_id=source_bundle.get("source_id"),
                reasoning=reasoning,
            )

        updated = self._attach_evidence_excerpts(claim, source_bundle)
        first_snippet = str(snippets[0].get("text") or "").strip()
        updated.evidence_quote = first_snippet[:300]
        support_hint = ((source_bundle.get("semantic_rerank") or {}).get("support_hint") or "").strip()
        if support_hint in {"fact_premise_support", "partial_or_nuanced_support"} or _looks_interpretive(updated):
            updated.support_relation = SupportRelation.PARTIALLY_SUPPORTS
            updated.final_bucket = FinalGroundingBucket.WEAK_FACT_GROUNDING
        elif support_hint == "opinion_only":
            updated.support_relation = SupportRelation.OPINION_ONLY
            updated.final_bucket = FinalGroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING
        else:
            updated.support_relation = SupportRelation.DIRECTLY_SUPPORTS
            updated.final_bucket = (
                FinalGroundingBucket.HARD_FACT_GROUNDING
                if _source_bundle_is_fact_like(source_bundle)
                else FinalGroundingBucket.WEAK_FACT_GROUNDING
            )
        updated.review_category = ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK
        updated.risk_flags = [
            flag for flag in updated.risk_flags if flag != RiskFlag.INACCESSIBLE_SOURCE
        ]
        updated.reasoning_summary = (
            f"{reasoning} Falling back to retrieved source excerpts instead of marking the citation "
            "as inaccessible, because source body and candidate evidence snippets were available."
        )
        if updated.best_evidence_excerpt:
            updated.best_evidence_excerpt.excerpt_role = _excerpt_role_for(
                updated,
                support_hint=support_hint,
            )
            updated.best_evidence_excerpt.explanation = _excerpt_explanation(
                updated,
                support_hint=support_hint,
            )
        return updated

    def _mark_inaccessible(self, claim: Claim, *, source_id: str | None, reasoning: str) -> Claim:
        updated = claim.model_copy(deep=True)
        updated.support_relation = SupportRelation.INACCESSIBLE
        updated.final_bucket = FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH
        updated.reasoning_summary = reasoning
        flags = list(updated.risk_flags)
        if RiskFlag.INACCESSIBLE_SOURCE not in flags:
            flags.append(RiskFlag.INACCESSIBLE_SOURCE)
        updated.risk_flags = list(dict.fromkeys(flags))
        return updated

    def _mark_no_relevant_snippet(self, claim: Claim, *, source_id: str | None, reasoning: str) -> Claim:
        updated = claim.model_copy(deep=True)
        updated.support_relation = SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET
        updated.final_bucket = FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH
        updated.reasoning_summary = reasoning
        flags = list(updated.risk_flags)
        if RiskFlag.INACCESSIBLE_SOURCE not in flags:
            flags.append(RiskFlag.INACCESSIBLE_SOURCE)
        updated.risk_flags = list(dict.fromkeys(flags))
        updated.evidence_needed = list(
            dict.fromkeys(updated.evidence_needed + ["Relevant source excerpt for the cited claim."])
        )
        return updated

    def _edge_from_claim(
        self,
        claim: Claim,
        source: Source | None,
        *,
        edge_type: EdgeType,
        basis: EdgeBasis,
        reasoning: str | None = None,
    ) -> EvidenceEdge:
        quote = claim.evidence_quote or ""
        return EvidenceEdge(
            claim_id=claim.claim_id,
            source_id=source.source_id if source else None,
            edge_type=edge_type,
            basis=basis,
            support_relation=claim.support_relation or SupportRelation.NOT_CHECKED,
            evidence_span=quote,
            evidence_quote=quote,
            reasoning_summary=reasoning or claim.reasoning_summary,
            final_bucket=claim.final_bucket or FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
            source_role=claim.source_role or SourceRole.UNKNOWN,
            best_evidence_excerpt=claim.best_evidence_excerpt,
            evidence_excerpts=list(claim.evidence_excerpts),
        )


def _run_async(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("Synchronous support checking cannot run inside an already running event loop.") from None


def _excerpt_role_for(claim: Claim, *, support_hint: str = "") -> str:
    if support_hint in {
        "direct_fact_support",
        "fact_premise_support",
        "partial_or_nuanced_support",
        "opinion_only",
        "no_support",
    }:
        return {
            "direct_fact_support": "direct_support",
            "fact_premise_support": "fact_premise_support",
            "partial_or_nuanced_support": "partial_or_nuanced_support",
            "opinion_only": "opinion_statement",
            "no_support": "closest_available_context",
        }[support_hint]
    if claim.support_relation == SupportRelation.DIRECTLY_SUPPORTS:
        return "direct_support"
    if claim.support_relation in {SupportRelation.PARTIALLY_SUPPORTS, SupportRelation.SUPPORTS_WEAKER_CLAIM}:
        return "partial_or_nuanced_support"
    if claim.support_relation == SupportRelation.OPINION_ONLY:
        return "opinion_statement"
    if claim.support_relation == SupportRelation.CONTRADICTS:
        return "contradicting_excerpt"
    return "closest_available_context"


def _excerpt_explanation(claim: Claim, *, support_hint: str = "") -> str:
    if support_hint == "fact_premise_support":
        return "Semantic reranker selected this source excerpt as factual premise support."
    if support_hint == "partial_or_nuanced_support":
        return "Semantic reranker selected this source excerpt as partial or nuanced support."
    if claim.support_relation == SupportRelation.DIRECTLY_SUPPORTS:
        return "Selected source excerpt used for direct support checking."
    if claim.support_relation in {SupportRelation.PARTIALLY_SUPPORTS, SupportRelation.SUPPORTS_WEAKER_CLAIM}:
        return "Selected source excerpt used for partial or weaker support checking."
    return "Closest retrieved source excerpt used for support checking."


def _source_bundle_is_fact_like(source_bundle: dict[str, Any]) -> bool:
    source_type = source_bundle.get("source_type")
    officialness = source_bundle.get("officialness_status")
    return source_type in {
        SourceType.PRIMARY_FACT_SOURCE.value,
        SourceType.EVIDENCE_SYNTHESIS.value,
    } or officialness in {
        OfficialnessStatus.VERIFIED_FIRST_PARTY.value,
        OfficialnessStatus.PROBABLE_FIRST_PARTY.value,
        OfficialnessStatus.VERIFIED_AFFILIATED_SOURCE.value,
    }


def _looks_interpretive(claim: Claim) -> bool:
    text = " ".join([claim.original_span or "", claim.original_text_span or "", claim.normalized_claim or ""]).lower()
    return any(
        term in text
        for term in [
            "适合",
            "建议",
            "推荐",
            "首选",
            "偏新手",
            "灵活试用",
            "更适合",
            "不建议",
            "影响",
            "风险",
            "判断",
            "opinion",
            "recommend",
            "suitable",
            "risk",
        ]
    )
