from __future__ import annotations

from collections import Counter
from typing import Iterable, List

from .schemas import (
    AnalysisSummary,
    Claim,
    ClaimType,
    ContentMix,
    DisplayCitationResult,
    DisplayStatus,
    GroundingBucket,
    GroundingMix,
    KeyRates,
    RiskFlag,
    SourceRoleForClaim,
    SourceOpacity,
    SupportRelation,
    SupportRelationMix,
    SupportScope,
)


def _ratio(count: int, denom: int) -> float:
    return round(count / denom, 4) if denom else 0.0


class RatioReporter:
    def build_summary(
        self,
        claims: list[Claim],
        *,
        display_citations: list[DisplayCitationResult] | None = None,
    ) -> AnalysisSummary:
        total = len(claims)
        auditable = [c for c in claims if c.claim_type != ClaimType.NON_CLAIM]
        non_claim_count = total - len(auditable)
        denom = len(auditable)

        type_counts = Counter(c.claim_type.value for c in auditable)
        content_mix = ContentMix(
            factual=_ratio(type_counts[ClaimType.FACTUAL.value], denom),
            attribution=_ratio(type_counts[ClaimType.ATTRIBUTION.value], denom),
            judgment=_ratio(type_counts[ClaimType.JUDGMENT.value], denom),
            has_quantitative_data=_ratio(sum(1 for c in auditable if c.has_quantitative_data), denom),
        )

        bucket_counts = Counter((c.final_bucket or GroundingBucket.UNVERIFIABLE_OR_MISMATCH).value for c in auditable)
        grounding_mix = GroundingMix(
            hard_fact_grounding=_ratio(bucket_counts[GroundingBucket.HARD_FACT_GROUNDING.value], denom),
            weak_fact_grounding=_ratio(bucket_counts[GroundingBucket.WEAK_FACT_GROUNDING.value], denom),
            attribution_or_opinion_grounding=_ratio(bucket_counts[GroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING.value], denom),
            unverifiable_or_mismatch=_ratio(bucket_counts[GroundingBucket.UNVERIFIABLE_OR_MISMATCH.value], denom),
            excluded_or_context=_ratio(bucket_counts[GroundingBucket.EXCLUDED_OR_CONTEXT.value], denom),
        )

        relation_counts = Counter((c.support_relation or SupportRelation.INACCESSIBLE).value for c in auditable)
        support_relation_mix = SupportRelationMix(
            directly_supports=_ratio(relation_counts[SupportRelation.DIRECTLY_SUPPORTS.value], denom),
            partially_supports=_ratio(relation_counts[SupportRelation.PARTIALLY_SUPPORTS.value], denom),
            supports_weaker_claim=_ratio(relation_counts[SupportRelation.SUPPORTS_WEAKER_CLAIM.value], denom),
            attribution_only=_ratio(relation_counts[SupportRelation.ATTRIBUTION_ONLY.value], denom),
            opinion_only=_ratio(relation_counts[SupportRelation.OPINION_ONLY.value], denom),
            background_only=_ratio(relation_counts[SupportRelation.BACKGROUND_ONLY.value], denom),
            no_support=_ratio(relation_counts[SupportRelation.NO_SUPPORT.value], denom),
            contradicts=_ratio(relation_counts[SupportRelation.CONTRADICTS.value], denom),
            inaccessible=_ratio(relation_counts[SupportRelation.INACCESSIBLE.value], denom),
            audit_limited_no_relevant_snippet=_ratio(
                relation_counts[SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET.value],
                denom,
            ),
            not_checked=_ratio(relation_counts[SupportRelation.NOT_CHECKED.value], denom),
        )

        opinion_packaging = sum(1 for c in auditable if RiskFlag.OPINION_USED_AS_FACT in c.risk_flags)
        opacity_flags = {
            RiskFlag.ANONYMOUS_SOURCE,
            RiskFlag.VAGUE_SOURCE,
            RiskFlag.NAMED_SECONDARY_WITH_OPAQUE_UNDERLYING,
        }
        opaque_source_labels = {
            SourceOpacity.ANONYMOUS_SOURCE,
            SourceOpacity.VAGUE_SOURCE_MENTION,
            SourceOpacity.NAMED_SECONDARY_WITH_OPAQUE_UNDERLYING,
        }
        source_opacity = sum(
            1
            for c in auditable
            if c.source_opacity in opaque_source_labels or any(flag in opacity_flags for flag in c.risk_flags)
        )
        mismatch_relations = {
            SupportRelation.NO_SUPPORT,
            SupportRelation.CONTRADICTS,
        }
        def has_usable_source_body(claim: Claim) -> bool:
            return any(
                edge.source_id
                and edge.support_relation not in {
                    SupportRelation.INACCESSIBLE,
                    SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
                }
                and bool(edge.evidence_quote or edge.evidence_span)
                for edge in claim.evidence_chain
            )

        true_mismatch = sum(
            1
            for c in auditable
            if has_usable_source_body(c)
            and (RiskFlag.SOURCE_CLAIM_MISMATCH in c.risk_flags or (c.support_relation in mismatch_relations))
        )
        audit_limited = sum(
            1
            for c in auditable
            if c.support_relation
            in {
                SupportRelation.INACCESSIBLE,
                SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
            }
            or RiskFlag.INACCESSIBLE_SOURCE in c.risk_flags
        )
        premise_support_for_analysis = sum(
            1
            for c in auditable
            if c.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS
            or any(edge.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS for edge in c.evidence_chain)
        )
        official_fact_scopes = {
            SupportScope.OWN_INSTITUTIONAL_FACT,
            SupportScope.OWN_PRODUCT_OR_PROGRAM_FACT,
            SupportScope.OWN_REPORTED_DATA,
            SupportScope.OFFICIAL_ANNOUNCEMENT,
        }
        official_source_roles = {
            SourceRoleForClaim.OFFICIAL_INSTITUTION_SOURCE,
            SourceRoleForClaim.OFFICIAL_COMPANY_SOURCE,
            SourceRoleForClaim.REGULATORY_OR_FILING_SOURCE,
            SourceRoleForClaim.SCHOLARLY_PRIMARY_SOURCE,
            SourceRoleForClaim.EVIDENCE_SYNTHESIS_SOURCE,
        }
        official_fact_support = sum(
            1
            for c in auditable
            if (
                c.support_scope in official_fact_scopes
                and c.source_role_for_claim in official_source_roles
                and c.support_relation
                in {
                    SupportRelation.DIRECTLY_SUPPORTS,
                    SupportRelation.PARTIALLY_SUPPORTS,
                    SupportRelation.SUPPORTS_WEAKER_CLAIM,
                }
            )
        )

        key_rates = KeyRates(
            verified_fact_support_rate=grounding_mix.hard_fact_grounding,
            partial_or_weak_support_rate=grounding_mix.weak_fact_grounding,
            attribution_support_rate=grounding_mix.attribution_or_opinion_grounding,
            analysis_from_sourced_premises_rate=_ratio(premise_support_for_analysis, denom),
            audit_limited_rate=_ratio(audit_limited, denom),
            true_mismatch_rate=_ratio(true_mismatch, denom),
            public_fact_support_rate=grounding_mix.hard_fact_grounding,
            loose_fact_support_rate=round(grounding_mix.hard_fact_grounding + grounding_mix.weak_fact_grounding, 4),
            opinion_packaging_rate=_ratio(opinion_packaging, denom),
            source_opacity_rate=_ratio(source_opacity, denom),
            citation_mismatch_rate=_ratio(true_mismatch, denom),
            premise_support_for_analysis_rate=_ratio(premise_support_for_analysis, denom),
            official_fact_support_rate=_ratio(official_fact_support, denom),
        )
        if display_citations is not None:
            key_rates = _key_rates_from_display_citations(
                display_citations,
                opinion_packaging_rate=_ratio(opinion_packaging, denom),
                source_opacity_rate=_ratio(source_opacity, denom),
                official_fact_support_rate=_ratio(official_fact_support, denom),
            )

        return AnalysisSummary(
            total_claims=total,
            auditable_claims=denom,
            non_claim_items=non_claim_count,
            content_mix=content_mix,
            grounding_mix=grounding_mix,
            support_relation_mix=support_relation_mix,
            key_rates=key_rates,
        )


def _key_rates_from_display_citations(
    display_citations: list[DisplayCitationResult],
    *,
    opinion_packaging_rate: float,
    source_opacity_rate: float,
    official_fact_support_rate: float,
) -> KeyRates:
    user_visible = [
        item
        for item in display_citations
        if item.display_status != DisplayStatus.EXCLUDED_OR_CONTEXT
    ]
    denom = len(user_visible)
    status_counts = Counter(item.display_status.value for item in user_visible)
    verified = _ratio(status_counts[DisplayStatus.VERIFIED_FACT_SUPPORT.value], denom)
    partial = _ratio(status_counts[DisplayStatus.PARTIAL_OR_WEAK_SUPPORT.value], denom)
    attribution = _ratio(status_counts[DisplayStatus.ATTRIBUTION_SUPPORT.value], denom)
    analysis = _ratio(status_counts[DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES.value], denom)
    audit_limited = _ratio(status_counts[DisplayStatus.AUDIT_LIMITED.value], denom)
    true_mismatch = _ratio(status_counts[DisplayStatus.TRUE_CITATION_PROBLEM.value], denom)
    return KeyRates(
        verified_fact_support_rate=verified,
        partial_or_weak_support_rate=partial,
        attribution_support_rate=attribution,
        analysis_from_sourced_premises_rate=analysis,
        audit_limited_rate=audit_limited,
        true_mismatch_rate=true_mismatch,
        public_fact_support_rate=verified,
        loose_fact_support_rate=round(verified + partial, 4),
        opinion_packaging_rate=opinion_packaging_rate,
        source_opacity_rate=source_opacity_rate,
        citation_mismatch_rate=true_mismatch,
        premise_support_for_analysis_rate=analysis,
        official_fact_support_rate=official_fact_support_rate,
    )
