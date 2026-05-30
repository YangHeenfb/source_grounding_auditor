from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from .providers.llm_provider import LLMProvider, LLMProviderError, LLMProviderTimeoutError
from .providers.llm_provider import CancellationToken
from .schemas import (
    AccessStatus,
    Claim,
    ClaimType,
    EdgeBasis,
    EdgeType,
    EvidenceEdge,
    FinalGroundingBucket,
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
            llm_indexes.append(index)
            llm_payloads.append(
                {
                    "check_id": f"sc{index+1:04d}",
                    "claim": item.claim,
                    "source_bundle": self._source_bundle(item.source),
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
                    self._mark_inaccessible(
                        item["claim"],
                        source_id=item["source_bundle"].get("source_id"),
                        reasoning=f"Source support check timed out in this run: {exc}",
                    )
                    for item in llm_payloads
                ]
            except LLMProviderError as exc:
                updated_claims = [
                    self._mark_inaccessible(
                        item["claim"],
                        source_id=item["source_bundle"].get("source_id"),
                        reasoning=f"Source support check could not be completed by the LLM provider in this run: {exc}",
                    )
                    for item in llm_payloads
                ]
            for index, updated in zip(llm_indexes, updated_claims):
                item = checks[index]
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

    def _source_bundle(self, source: Source | None) -> dict[str, Any]:
        if source is None:
            return {}
        source_text = source.extracted_text or ""
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
            "extracted_text": source_text[:MAX_SOURCE_TEXT_CHARS],
            "extracted_text_preview": source.extracted_text_preview,
        }

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
        )


def _run_async(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("Synchronous support checking cannot run inside an already running event loop.") from None
