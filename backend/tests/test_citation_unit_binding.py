from app.analyzer import SourceGroundingAnalyzer
from app.providers.llm_provider import MockLLMProvider
from app.schemas import (
    AnalysisRequest,
    Claim,
    ClaimReviewCategory,
    ClaimType,
    DiscourseRole,
    FinalGroundingBucket,
    ImportanceLabel,
    SourceRole,
    SourceRoleForClaim,
    SupportRelation,
    SupportScope,
)


DKU_DEGREE_TEXT = "DKU 本科完成后会拿到 DKU 的中国学位和 Duke University 的学位，并成为两校校友。[3]"
DKU_REGISTRY = (
    '[3]: https://www.dukekunshan.edu.cn/about/the-liberal-arts-in-the-21st-century/ '
    '"Undergraduate Curriculum - DKU"'
)
DKU_URL = "https://www.dukekunshan.edu.cn/about/the-liberal-arts-in-the-21st-century/"
DKU_SOURCE_EXCERPT = (
    "Upon completion of the DKU and Duke requirements, students receive a DKU Chinese "
    "graduation certificate and degree, and a Duke University degree, and become alumni "
    "of both institutions."
)


def dku_degree_claim() -> Claim:
    return Claim(
        claim_id="c001",
        original_text_span=DKU_DEGREE_TEXT,
        original_span=DKU_DEGREE_TEXT,
        normalized_claim="DKU 本科毕业生可获得 DKU 的中国学位和 Duke University 的学位，并成为两校校友",
        claim_type=ClaimType.FACTUAL,
        discourse_role=DiscourseRole.ASSERTED_CLAIM,
        has_quantitative_data=False,
        has_material_quantitative_data=False,
        importance_label=ImportanceLabel.SUPPORTING,
        review_category=ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK,
    )


def test_inline_reference_label_binds_to_official_dku_source_and_supports_institutional_fact():
    provider = MockLLMProvider(
        extraction_outputs=[dku_degree_claim()],
        support_outputs={
            "c001": {
                "support_relation": SupportRelation.DIRECTLY_SUPPORTS.value,
                "final_bucket": FinalGroundingBucket.HARD_FACT_GROUNDING.value,
                "risk_flags": [],
                "reasoning_summary": "The DKU page states the Chinese DKU degree, Duke degree, and alumni status.",
                "evidence_quote": DKU_SOURCE_EXCERPT[:300],
                "source_role": SourceRole.PRIMARY_FACT_SOURCE.value,
            }
        },
    )
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=f"{DKU_DEGREE_TEXT}\n\n来源指针\n{DKU_REGISTRY}",
            enable_url_fetch=False,
            enable_web_search=True,
            provided_sources=[
                {
                    "url": DKU_URL,
                    "title": "Undergraduate Curriculum - DKU",
                    "source_type": "primary_fact_source",
                    "extracted_text": DKU_SOURCE_EXCERPT,
                    "access_status": "accessible",
                }
            ],
        )
    )

    assert result.problematic_citations == []
    assert result.claims[0].citation_label == "3"
    assert result.claims[0].citation_source_url == DKU_URL
    assert result.claims[0].support_relation in {
        SupportRelation.DIRECTLY_SUPPORTS,
        SupportRelation.PARTIALLY_SUPPORTS,
    }
    assert result.claims[0].final_bucket in {
        FinalGroundingBucket.HARD_FACT_GROUNDING,
        FinalGroundingBucket.WEAK_FACT_GROUNDING,
    }
    assert result.claims[0].source_role_for_claim == SourceRoleForClaim.OFFICIAL_INSTITUTION_SOURCE
    assert result.claims[0].support_scope == SupportScope.OWN_INSTITUTIONAL_FACT
    assert result.metadata["search_query_count"] == 0


def test_inline_reference_label_with_failed_fetch_is_audit_limited_not_problematic_or_true_mismatch():
    provider = MockLLMProvider(extraction_outputs=[dku_degree_claim()])
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=f"{DKU_DEGREE_TEXT}\n\n来源指针\n{DKU_REGISTRY}",
            enable_url_fetch=False,
            enable_web_search=True,
        )
    )

    assert result.problematic_citations == []
    assert len(result.audit_limited_citations) == 1
    assert result.claims[0].citation_source_url == DKU_URL
    assert result.claims[0].support_relation == SupportRelation.INACCESSIBLE
    assert result.summary.key_rates.true_mismatch_rate == 0
    assert result.summary.key_rates.citation_mismatch_rate == 0
    assert result.metadata["search_query_count"] == 0
