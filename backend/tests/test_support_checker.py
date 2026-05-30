from app.providers.llm_provider import MockLLMProvider
from app.support_checker import SupportChecker, SupportCheckInput
from app.schemas import (
    AccessStatus,
    Claim,
    ClaimType,
    FinalGroundingBucket,
    ImportanceLabel,
    Source,
    SourceRole,
    SourceType,
    SupportRelation,
    RiskFlag,
)


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
    provider = MockLLMProvider(
        support_outputs={
            "c001": {
                "support_relation": SupportRelation.DIRECTLY_SUPPORTS.value,
                "final_bucket": FinalGroundingBucket.HARD_FACT_GROUNDING.value,
                "risk_flags": [],
                "reasoning_summary": "The source directly states the revenue claim.",
                "evidence_quote": "The company reported revenue of $10 billion in 2024 in its annual report.",
                "source_role": SourceRole.PRIMARY_FACT_SOURCE.value,
            }
        }
    )
    edge, flags = SupportChecker(provider).check(claim, source)
    assert edge.support_relation == SupportRelation.DIRECTLY_SUPPORTS
    assert edge.final_bucket == FinalGroundingBucket.HARD_FACT_GROUNDING


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
    provider = MockLLMProvider(
        support_outputs={
            "c001": {
                "support_relation": SupportRelation.SUPPORTS_WEAKER_CLAIM.value,
                "final_bucket": FinalGroundingBucket.WEAK_FACT_GROUNDING.value,
                "risk_flags": [RiskFlag.CORRELATION_PRESENTED_AS_CAUSATION.value],
                "reasoning_summary": "The source supports association, not causation.",
                "evidence_quote": "association between coffee intake and lower mortality",
                "source_role": SourceRole.PRIMARY_FACT_SOURCE.value,
            }
        }
    )
    edge, flags = SupportChecker(provider).check(claim, source)
    assert edge.support_relation == SupportRelation.SUPPORTS_WEAKER_CLAIM
    assert RiskFlag.CORRELATION_PRESENTED_AS_CAUSATION in flags
    assert edge.final_bucket == FinalGroundingBucket.WEAK_FACT_GROUNDING


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
    provider = MockLLMProvider(
        support_outputs={
            "c001": {
                "support_relation": SupportRelation.NO_SUPPORT.value,
                "final_bucket": FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH.value,
                "risk_flags": [RiskFlag.SOURCE_CLAIM_MISMATCH.value],
                "reasoning_summary": "The source attaches the number to a different subject.",
                "evidence_quote": "The report says inflation fell by 15 percent.",
                "source_role": SourceRole.PRIMARY_FACT_SOURCE.value,
            }
        }
    )
    edge, flags = SupportChecker(provider).check(claim, source)
    assert edge.support_relation == SupportRelation.NO_SUPPORT
    assert RiskFlag.SOURCE_CLAIM_MISMATCH in flags
    assert edge.final_bucket == FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH


def test_support_checker_batches_accessible_source_checks():
    calls = {"batch": 0, "single": 0}

    class BatchProvider(MockLLMProvider):
        async def check_claim_support(self, claim, source_bundle, cancellation_token=None):
            calls["single"] += 1
            return await super().check_claim_support(claim, source_bundle, cancellation_token=cancellation_token)

        async def check_claim_supports(self, checks, cancellation_token=None):
            calls["batch"] += 1
            updated = []
            for item in checks:
                claim = item["claim"].model_copy(deep=True)
                claim.support_relation = SupportRelation.DIRECTLY_SUPPORTS
                claim.final_bucket = FinalGroundingBucket.HARD_FACT_GROUNDING
                claim.risk_flags = []
                claim.reasoning_summary = "The source directly supports the claim."
                claim.evidence_quote = item["source_bundle"]["extracted_text"][:80]
                claim.source_role = SourceRole.PRIMARY_FACT_SOURCE
                updated.append(claim)
            return updated

    claim_a = make_claim("The company reported revenue of $10 billion in 2024")
    claim_b = make_claim("The company reported operating income of $2 billion in 2024")
    claim_b.claim_id = "c002"
    source = Source(
        source_id="s001",
        title="Annual report",
        access_status=AccessStatus.ACCESSIBLE,
        source_type=SourceType.PRIMARY_FACT_SOURCE,
        extracted_text="The company reported revenue of $10 billion and operating income of $2 billion in 2024.",
        extracted_text_preview="The company reported revenue of $10 billion and operating income of $2 billion in 2024.",
    )

    results = SupportChecker(BatchProvider()).check_many(
        [
            SupportCheckInput(claim=claim_a, source=source),
            SupportCheckInput(claim=claim_b, source=source),
        ]
    )

    assert len(results) == 2
    assert calls == {"batch": 1, "single": 0}
    assert all(edge.support_relation == SupportRelation.DIRECTLY_SUPPORTS for edge, _flags in results)
