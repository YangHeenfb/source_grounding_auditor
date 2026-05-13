from app.ratio_reporter import RatioReporter
from app.schemas import Claim, ClaimType, GroundingBucket, ImportanceLabel, RiskFlag, SupportRelation


def c(i, claim_type, bucket, relation=SupportRelation.DIRECTLY_SUPPORTS, quant=False, flags=None):
    return Claim(
        claim_id=f"c{i:03d}",
        original_text_span="x",
        normalized_claim="x",
        claim_type=claim_type,
        has_quantitative_data=quant,
        importance_label=ImportanceLabel.SUPPORTING,
        final_bucket=bucket,
        support_relation=relation,
        risk_flags=flags or [],
    )


def test_ratio_reporter_core_rates():
    claims = [
        c(1, ClaimType.FACTUAL, GroundingBucket.HARD_FACT_GROUNDING, quant=True),
        c(2, ClaimType.FACTUAL, GroundingBucket.WEAK_FACT_GROUNDING, relation=SupportRelation.SUPPORTS_WEAKER_CLAIM),
        c(3, ClaimType.JUDGMENT, GroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING, relation=SupportRelation.OPINION_ONLY, flags=[RiskFlag.OPINION_USED_AS_FACT]),
        c(4, ClaimType.ATTRIBUTION, GroundingBucket.UNVERIFIABLE_OR_MISMATCH, relation=SupportRelation.INACCESSIBLE, flags=[RiskFlag.VAGUE_SOURCE]),
    ]
    summary = RatioReporter().build_summary(claims)
    assert summary.auditable_claims == 4
    assert summary.content_mix.factual == 0.5
    assert summary.content_mix.has_quantitative_data == 0.25
    assert summary.grounding_mix.hard_fact_grounding == 0.25
    assert summary.key_rates.loose_fact_support_rate == 0.5
    assert summary.key_rates.opinion_packaging_rate == 0.25
    assert summary.key_rates.source_opacity_rate == 0.25
