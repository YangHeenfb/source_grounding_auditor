from __future__ import annotations

from collections import Counter
from typing import Iterable, List

from .schemas import (
    AnalysisSummary,
    Claim,
    ClaimType,
    ContentMix,
    GroundingBucket,
    GroundingMix,
    KeyRates,
    RiskFlag,
    SupportRelation,
    SupportRelationMix,
)


def _ratio(count: int, denom: int) -> float:
    return round(count / denom, 4) if denom else 0.0


class RatioReporter:
    def build_summary(self, claims: list[Claim]) -> AnalysisSummary:
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
        )

        opinion_packaging = sum(1 for c in auditable if RiskFlag.OPINION_USED_AS_FACT in c.risk_flags)
        opacity_flags = {RiskFlag.ANONYMOUS_SOURCE, RiskFlag.VAGUE_SOURCE, RiskFlag.INACCESSIBLE_SOURCE}
        source_opacity = sum(1 for c in auditable if any(flag in opacity_flags for flag in c.risk_flags))
        mismatch_flags = {
            RiskFlag.SOURCE_CLAIM_MISMATCH,
            RiskFlag.SOURCE_ONLY_SUPPORTS_WEAKER_CLAIM,
            RiskFlag.CORRELATION_PRESENTED_AS_CAUSATION,
            RiskFlag.QUANTITATIVE_CLAIM_WITHOUT_PRIMARY_DATA,
        }
        mismatch_relations = {
            SupportRelation.NO_SUPPORT,
            SupportRelation.CONTRADICTS,
            SupportRelation.BACKGROUND_ONLY,
            SupportRelation.SUPPORTS_WEAKER_CLAIM,
        }
        citation_mismatch = sum(
            1
            for c in auditable
            if any(flag in mismatch_flags for flag in c.risk_flags) or (c.support_relation in mismatch_relations)
        )

        key_rates = KeyRates(
            public_fact_support_rate=grounding_mix.hard_fact_grounding,
            loose_fact_support_rate=round(grounding_mix.hard_fact_grounding + grounding_mix.weak_fact_grounding, 4),
            opinion_packaging_rate=_ratio(opinion_packaging, denom),
            source_opacity_rate=_ratio(source_opacity, denom),
            citation_mismatch_rate=_ratio(citation_mismatch, denom),
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
