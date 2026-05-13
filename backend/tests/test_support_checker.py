from app.support_checker import SupportChecker
from app.schemas import AccessStatus, Claim, ClaimType, ImportanceLabel, Source, SourceType, SupportRelation, GroundingBucket, RiskFlag


def make_claim(text, claim_type=ClaimType.FACTUAL, quant=False):
    return Claim(
        claim_id="c001",
        original_text_span=text,
        normalized_claim=text,
        claim_type=claim_type,
        has_quantitative_data=quant,
        importance_label=ImportanceLabel.SUPPORTING,
    )


def test_primary_direct_support_is_hard_fact():
    claim = make_claim("The company reported revenue of $10 billion in 2024", quant=True)
    source = Source(
        source_id="s001",
        url="https://example.com/annual-report",
        title="2024 annual report",
        access_status=AccessStatus.ACCESSIBLE,
        source_type=SourceType.PRIMARY_FACT_SOURCE,
        extracted_text="The company reported revenue of $10 billion in 2024 in its annual report.",
        extracted_text_preview="The company reported revenue of $10 billion in 2024 in its annual report.",
    )
    edge, flags = SupportChecker().check(claim, source)
    assert edge.support_relation == SupportRelation.DIRECTLY_SUPPORTS
    assert edge.final_bucket == GroundingBucket.HARD_FACT_GROUNDING


def test_correlation_as_causation_is_weaker_support():
    claim = make_claim("The study proves that coffee causes lower mortality", claim_type=ClaimType.ATTRIBUTION)
    source = Source(
        source_id="s001",
        title="Coffee study",
        access_status=AccessStatus.ACCESSIBLE,
        source_type=SourceType.PRIMARY_FACT_SOURCE,
        extracted_text="The study found an association between coffee intake and lower mortality in an observational cohort.",
        extracted_text_preview="The study found an association between coffee intake and lower mortality in an observational cohort.",
    )
    edge, flags = SupportChecker().check(claim, source)
    assert edge.support_relation == SupportRelation.SUPPORTS_WEAKER_CLAIM
    assert RiskFlag.CORRELATION_PRESENTED_AS_CAUSATION in flags
    assert edge.final_bucket == GroundingBucket.WEAK_FACT_GROUNDING


def test_source_claim_mismatch():
    claim = make_claim("The report says unemployment fell by 15 percent", claim_type=ClaimType.ATTRIBUTION, quant=True)
    source = Source(
        source_id="s001",
        title="Report",
        access_status=AccessStatus.ACCESSIBLE,
        source_type=SourceType.PRIMARY_FACT_SOURCE,
        extracted_text="The report says inflation fell by 15 percent.",
        extracted_text_preview="The report says inflation fell by 15 percent.",
    )
    edge, flags = SupportChecker().check(claim, source)
    assert edge.support_relation == SupportRelation.NO_SUPPORT
    assert RiskFlag.SOURCE_CLAIM_MISMATCH in flags
    assert edge.final_bucket == GroundingBucket.UNVERIFIABLE_OR_MISMATCH
