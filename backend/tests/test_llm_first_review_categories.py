from app.analyzer import SourceGroundingAnalyzer
from app.providers.llm_provider import LLMProviderError, MockLLMProvider
from app.schemas import (
    AnalysisRequest,
    Claim,
    ClaimReviewCategory,
    ClaimType,
    DiscourseRole,
    FinalGroundingBucket,
    ImportanceLabel,
    RiskFlag,
    SourceOpacity,
    SourceRole,
    SupportRelation,
)


def make_claim(
    text,
    *,
    claim_type=ClaimType.FACTUAL,
    role=DiscourseRole.ASSERTED_CLAIM,
    category=ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK,
    opacity=SourceOpacity.NOT_APPLICABLE,
    flags=None,
    quant=False,
    material_quant=False,
    importance=ImportanceLabel.SUPPORTING,
    not_asserted=False,
):
    return Claim(
        claim_id="c001",
        original_text_span=text,
        original_span=text,
        normalized_claim=text.strip("。."),
        claim_type=claim_type,
        discourse_role=role,
        source_opacity=opacity,
        risk_flags=flags or [],
        has_quantitative_data=quant,
        has_material_quantitative_data=material_quant,
        importance_label=importance,
        review_category=category,
        not_asserted_by_author=not_asserted,
    )


def analyze_with_claims(claims, **request_kwargs):
    provider = MockLLMProvider(extraction_outputs=claims, support_outputs=request_kwargs.pop("support_outputs", None))
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)
    input_text = request_kwargs.pop("input_text", claims[0].original_text_span if claims else "x")
    if "[1]" not in input_text and "http://" not in input_text and "https://" not in input_text:
        input_text = f"{input_text} [1]\n\n[1] Mock citation source"
    elif "[1]" in input_text and "[1] Mock citation source" not in input_text and not input_text.strip().endswith("Mock citation source"):
        input_text = f"{input_text}\n\n[1] Mock citation source"
    return analyzer.analyze(
        AnalysisRequest(
            input_text=input_text,
            enable_url_fetch=False,
            enable_web_search=False,
            **request_kwargs,
        )
    )


def test_reuters_reported_attribution_not_high_risk():
    text = "Reuters 报道称 Nvidia 曾接触 Cerebras，讨论潜在收购。"
    result = analyze_with_claims(
        [
            make_claim(
                text,
                claim_type=ClaimType.ATTRIBUTION,
                role=DiscourseRole.ATTRIBUTION_REPORT,
                category=ClaimReviewCategory.ATTRIBUTION_SUPPORTED,
                opacity=SourceOpacity.NAMED_SECONDARY_WITH_OPAQUE_UNDERLYING,
            )
        ],
        input_text=f"{text} [source](https://example.com/reuters)",
        provided_sources=[
            {
                "url": "https://example.com/reuters",
                "title": "Reuters report",
                "source_type": "secondary_reporting",
                "extracted_text": "Reuters reported that Nvidia had contacted Cerebras to discuss a potential acquisition.",
                "access_status": "accessible",
            }
        ],
        support_outputs={
            "c001": {
                "support_relation": SupportRelation.ATTRIBUTION_ONLY.value,
                "final_bucket": FinalGroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING.value,
                "risk_flags": [],
                "reasoning_summary": "The cited Reuters article supports that Reuters reported the statement.",
                "evidence_quote": "Reuters reported that Nvidia had contacted Cerebras.",
                "source_role": SourceRole.SECONDARY_REPORT.value,
            }
        },
    )

    assert not result.high_risk_claims
    assert not result.problematic_citations
    assert len(result.attribution_supported_claims) == 1
    assert len(result.attribution_supported_citations) == 1


def test_attribution_dropped_asserted_fact_is_high_risk():
    text = "Nvidia 确实接触过 Cerebras，讨论潜在收购。"
    result = analyze_with_claims(
        [
            make_claim(
                text,
                claim_type=ClaimType.FACTUAL,
                role=DiscourseRole.ASSERTED_CLAIM,
                category=ClaimReviewCategory.HIGH_RISK,
                opacity=SourceOpacity.NAMED_SECONDARY_WITH_OPAQUE_UNDERLYING,
                flags=[RiskFlag.ATTRIBUTION_DROPPED],
                importance=ImportanceLabel.CORE,
            )
        ],
        input_text=text,
    )

    assert result.problematic_citations == []
    assert len(result.audit_limited_citations) == 1


def test_caveat_is_excluded_context():
    text = "没有公开披露报价、时间线、条款或董事会投票细节。"
    result = analyze_with_claims(
        [
            make_claim(
                text,
                claim_type=ClaimType.NON_CLAIM,
                role=DiscourseRole.CAVEAT_OR_LIMITATION,
                category=ClaimReviewCategory.EXCLUDED_OR_CONTEXT,
            )
        ],
        input_text=text,
    )

    assert not result.high_risk_claims
    assert len(result.excluded_or_context_claims) == 1


def test_unsupported_example_not_asserted_by_author_is_excluded():
    text = "不能把它说成“双方已经签过收购意向书”。"
    result = analyze_with_claims(
        [
            make_claim(
                text,
                claim_type=ClaimType.NON_CLAIM,
                role=DiscourseRole.UNSUPPORTED_EXAMPLE,
                category=ClaimReviewCategory.EXCLUDED_OR_CONTEXT,
                not_asserted=True,
            )
        ],
        input_text=text,
    )

    assert not result.high_risk_claims
    assert result.claims[0].not_asserted_by_author is True


def test_groq_official_announcement_is_not_token_overlap_mismatch():
    text = "Groq 官方公告称 Groq 与 Nvidia 达成非独家推理技术授权协议，Groq 创始人 Jonathan Ross 和总裁 Sunny Madra 加入 Nvidia。"
    claim = make_claim(
        text,
        claim_type=ClaimType.ATTRIBUTION,
        role=DiscourseRole.ATTRIBUTION_REPORT,
        category=ClaimReviewCategory.ATTRIBUTION_SUPPORTED,
        opacity=SourceOpacity.CLEAR_NAMED_SOURCE,
    )
    support_outputs = {
        "c001": {
            "support_relation": SupportRelation.DIRECTLY_SUPPORTS.value,
            "final_bucket": FinalGroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING.value,
            "risk_flags": [],
            "reasoning_summary": "The official announcement directly states the licensing agreement and joining executives.",
            "evidence_quote": "Groq and NVIDIA entered into a non exclusive inference technology licensing agreement.",
            "source_role": SourceRole.OFFICIAL_ANNOUNCEMENT.value,
        }
    }
    result = analyze_with_claims(
        [claim],
        input_text=f"{text} [source](https://example.com/groq)",
        provided_sources=[
            {
                "url": "https://example.com/groq",
                "title": "Groq and Nvidia enter non exclusive inference technology licensing agreement",
                "source_type": "primary_fact_source",
                "extracted_text": (
                    "Groq and NVIDIA entered into a non exclusive inference technology licensing agreement. "
                    "Groq founder Jonathan Ross, president Sunny Madra, and select team members will join NVIDIA. "
                    "Groq will continue to operate as an independent company."
                ),
                "access_status": "accessible",
            }
        ],
        support_outputs=support_outputs,
    )

    assert not result.high_risk_claims
    assert RiskFlag.SOURCE_CLAIM_MISMATCH not in result.claims[0].risk_flags


def test_core_judgment_with_background_only_support_is_high_risk():
    text = "该消息说明 Nvidia 很清楚推理市场正在变化，SRAM 重、低延迟、高速推理这条路线值得防守。"
    result = analyze_with_claims(
        [
            make_claim(
                text,
                claim_type=ClaimType.JUDGMENT,
                role=DiscourseRole.JUDGMENT_OR_ANALYSIS,
                category=ClaimReviewCategory.HIGH_RISK,
                flags=[RiskFlag.UNSUPPORTED_CAUSAL_OR_STRATEGIC_INFERENCE],
                importance=ImportanceLabel.CORE,
            )
        ],
        input_text=text,
    )

    assert result.high_risk_claims == []
    assert len(result.audit_limited_claims) == 1


def test_date_is_not_material_quantitative_risk():
    text = "2026 年 2 月 2 日 Reuters 发布了报道。"
    result = analyze_with_claims(
        [
            make_claim(
                text,
                claim_type=ClaimType.ATTRIBUTION,
                role=DiscourseRole.ATTRIBUTION_REPORT,
                category=ClaimReviewCategory.ATTRIBUTION_SUPPORTED,
                opacity=SourceOpacity.CLEAR_NAMED_SOURCE,
                quant=True,
                material_quant=False,
            )
        ],
        input_text=text,
    )

    assert not result.high_risk_claims
    assert RiskFlag.QUANTITATIVE_CLAIM_WITHOUT_PRIMARY_DATA not in result.claims[0].risk_flags


def test_inaccessible_source_alone_is_audit_limited_not_high_risk():
    text = "Nvidia 与某公司有过接触。"
    result = analyze_with_claims(
        [
            make_claim(
                text,
                claim_type=ClaimType.FACTUAL,
                role=DiscourseRole.ASSERTED_CLAIM,
                category=ClaimReviewCategory.AUDIT_LIMITED,
                importance=ImportanceLabel.SUPPORTING,
            )
        ],
        input_text=text,
    )

    assert RiskFlag.INACCESSIBLE_SOURCE in result.claims[0].risk_flags
    assert not result.high_risk_claims
    assert len(result.audit_limited_claims) == 1


def test_nvidia_cerebras_groq_regression_fixture_reduces_high_risk():
    safe_items = [
        ("Reuters 报道称 Nvidia 曾接触 Cerebras。", ClaimType.ATTRIBUTION, DiscourseRole.ATTRIBUTION_REPORT, ClaimReviewCategory.ATTRIBUTION_SUPPORTED),
        ("Reuters 写到双方讨论潜在收购。", ClaimType.ATTRIBUTION, DiscourseRole.ATTRIBUTION_REPORT, ClaimReviewCategory.ATTRIBUTION_SUPPORTED),
        ("没有公开披露报价。", ClaimType.NON_CLAIM, DiscourseRole.CAVEAT_OR_LIMITATION, ClaimReviewCategory.EXCLUDED_OR_CONTEXT),
        ("是否有正式报价不清楚。", ClaimType.NON_CLAIM, DiscourseRole.CAVEAT_OR_LIMITATION, ClaimReviewCategory.EXCLUDED_OR_CONTEXT),
        ("Groq 官方公告称 Groq 与 Nvidia 达成非独家推理技术授权协议。", ClaimType.ATTRIBUTION, DiscourseRole.ATTRIBUTION_REPORT, ClaimReviewCategory.ATTRIBUTION_SUPPORTED),
        ("来源指针：[1] Reuters。", ClaimType.NON_CLAIM, DiscourseRole.SOURCE_POINTER, ClaimReviewCategory.EXCLUDED_OR_CONTEXT),
        ("不能说成双方已经签过收购意向书。", ClaimType.NON_CLAIM, DiscourseRole.UNSUPPORTED_EXAMPLE, ClaimReviewCategory.EXCLUDED_OR_CONTEXT),
    ]
    risky_texts = [
        "这个说法大体可信。",
        "该消息可信度较高。",
        "Cerebras 拒绝后与 OpenAI 达成商业合作。",
        "Nvidia 似乎在快速补强推理芯片能力。",
        "可以把该消息当作 Cerebras 技术资产被 Nvidia 重视的积极信号。",
        "该消息说明 Nvidia 很清楚推理市场正在变化。",
        "SRAM 重、低延迟、高速推理这条路线值得防守。",
        "Cerebras 被 Nvidia 接触并不奇怪。",
    ]
    claims = []
    for text, claim_type, role, category in safe_items:
        claims.append(make_claim(text, claim_type=claim_type, role=role, category=category))
    for text in risky_texts:
        claims.append(
            make_claim(
                text,
                claim_type=ClaimType.JUDGMENT,
                role=DiscourseRole.JUDGMENT_OR_ANALYSIS,
                category=ClaimReviewCategory.HIGH_RISK,
                flags=[RiskFlag.UNSUPPORTED_CAUSAL_OR_STRATEGIC_INFERENCE],
                importance=ImportanceLabel.CORE,
            )
        )
    for idx, claim in enumerate(claims, start=1):
        claim.claim_id = f"c{idx:03d}"

    result = analyze_with_claims(claims, input_text="\n".join(c.original_text_span for c in claims))

    assert result.high_risk_claims == []
    assert len(result.high_risk_claims) < 15
    high_risk_text = "\n".join(item.normalized_claim for item in result.high_risk_claims)
    assert "Reuters 报道称" not in high_risk_text
    assert result.attribution_supported_claims or result.audit_limited_claims
    assert result.excluded_or_context_claims


def test_citation_only_mode_excludes_uncited_text_from_claims_and_ratios():
    uncited = make_claim(
        "This uncited claim should not be analyzed.",
        claim_type=ClaimType.FACTUAL,
        role=DiscourseRole.ASSERTED_CLAIM,
        category=ClaimReviewCategory.HIGH_RISK,
        importance=ImportanceLabel.CORE,
    )
    result = analyze_with_claims(
        [uncited],
        input_text="This uncited claim should not be analyzed.",
    )

    assert result.claims
    assert result.summary.ratios_basis == "based only on cited claims"

    provider = MockLLMProvider(extraction_outputs=[uncited])
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)
    uncited_result = analyzer.analyze(
        AnalysisRequest(
            input_text="This uncited claim should not be analyzed.",
            enable_url_fetch=False,
            enable_web_search=False,
        )
    )
    assert uncited_result.claims == []
    assert uncited_result.summary.total_claims == 0
    assert uncited_result.summary.ratios_basis == "based only on cited claims"
    assert uncited_result.uncited_claim_analysis_enabled is False


def test_review_provider_failure_does_not_create_problematic_citations():
    class FailingReviewProvider(MockLLMProvider):
        async def classify_review_categories(self, claims, cancellation_token=None):
            raise LLMProviderError("Codex CLI request failed (1): model refresh failed")

    claim = make_claim(
        "The cited source says the company reported revenue of $10 billion.",
        claim_type=ClaimType.FACTUAL,
        role=DiscourseRole.ASSERTED_CLAIM,
        category=ClaimReviewCategory.HIGH_RISK,
        importance=ImportanceLabel.CORE,
    )
    provider = FailingReviewProvider(extraction_outputs=[claim])
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text="The cited source says the company reported revenue of $10 billion [1].\n\n[1] Mock citation source",
            enable_url_fetch=False,
            enable_web_search=False,
        )
    )

    assert result.problematic_citations == []
    assert result.high_risk_claims == []
    assert len(result.audit_limited_citations) == 1
    assert "Review classification could not be completed" in result.claims[0].reasoning_summary
