from app import analyzer as analyzer_module
from app.analyzer import SourceGroundingAnalyzer
from app.providers.llm_provider import LLMProviderTimeoutError, MockLLMProvider
from app.schemas import (
    AnalysisRequest,
    Claim,
    ClaimReviewCategory,
    ClaimType,
    DiscourseRole,
    FinalGroundingBucket,
    ImportanceLabel,
    SourceRole,
    SupportRelation,
    DisplayStatus,
    TerminalClass,
    UnresolvedReason,
)


URLS = {
    "1": "https://example.com/apple-2024",
    "2": "https://example.com/microsoft-2024",
    "3": "https://example.com/nvidia-2025",
    "4": "https://example.com/apple-services-commentary",
    "5": "https://example.com/nvidia-valuation-opinion",
    "6": "https://example.com/missing-quarterly-report",
}


def _claim_for_statement(input_text: str) -> list[Claim]:
    is_opinion = "认为" in input_text or "不再便宜" in input_text
    return [
        Claim(
            claim_id="c001",
            original_text_span=input_text,
            original_span=input_text,
            normalized_claim=input_text,
            claim_type=ClaimType.ATTRIBUTION if is_opinion else ClaimType.FACTUAL,
            discourse_role=DiscourseRole.ATTRIBUTION_REPORT if is_opinion else DiscourseRole.ASSERTED_CLAIM,
            has_quantitative_data=any(char.isdigit() for char in input_text),
            has_material_quantitative_data=any(char.isdigit() for char in input_text),
            importance_label=ImportanceLabel.SUPPORTING,
            review_category=ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK,
        )
    ]


def _support_for_claim(claim: Claim, _source_bundle):
    if claim.citation_label in {"1", "2", "3"}:
        return {
            "support_relation": SupportRelation.DIRECTLY_SUPPORTS.value,
            "final_bucket": FinalGroundingBucket.HARD_FACT_GROUNDING.value,
            "risk_flags": [],
            "reasoning_summary": "The provided source directly states the financial metric.",
            "evidence_quote": claim.normalized_claim,
            "source_role": SourceRole.PRIMARY_FACT_SOURCE.value,
        }
    return {
        "support_relation": SupportRelation.ATTRIBUTION_ONLY.value,
        "final_bucket": FinalGroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING.value,
        "risk_flags": [],
        "reasoning_summary": "The provided source preserves the opinion attribution.",
        "evidence_quote": claim.normalized_claim,
        "source_role": SourceRole.OPINION_OR_ANALYSIS.value,
    }


class FailingSearchProvider:
    calls = 0

    def search(self, query, max_results=5):  # pragma: no cover - should not be called
        self.calls += 1
        raise AssertionError(f"Search should not be called for explicit source URLs: {query}")


class CountingProvider(MockLLMProvider):
    def __init__(self, *, extraction_error: Exception | None = None, **kwargs):
        super().__init__(**kwargs)
        self.extraction_error = extraction_error
        self.extract_calls = 0
        self.support_batch_calls = 0

    async def extract_claims(self, *args, **kwargs):
        self.extract_calls += 1
        if self.extraction_error:
            raise self.extraction_error
        return await super().extract_claims(*args, **kwargs)

    async def check_claim_supports(self, checks, cancellation_token=None):
        self.support_batch_calls += 1
        return await super().check_claim_supports(checks, cancellation_token=cancellation_token)


def _direct_fact_support(claim: Claim, _source_bundle):
    return {
        "support_relation": SupportRelation.DIRECTLY_SUPPORTS.value,
        "final_bucket": FinalGroundingBucket.HARD_FACT_GROUNDING.value,
        "risk_flags": [],
        "reasoning_summary": "The source directly supports the cited statement.",
        "evidence_quote": claim.normalized_claim[:300],
        "source_role": SourceRole.PRIMARY_FACT_SOURCE.value,
    }


def test_citation_only_mode_creates_claims_from_cited_statements_without_llm_extraction():
    text = (
        "BOTZ 全名是 Global X Robotics & Artificial Intelligence ETF。[1]\n\n"
        "来源指针\n"
        "[1] BOTZ 的基金目标来自 Global X 官方页面。([Global X ETFs][1])\n\n"
        '[1]: https://www.globalxetfs.com/funds/botz/ "BOTZ"'
    )
    provider = CountingProvider(support_outputs=_direct_fact_support)
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=text,
            enable_url_fetch=False,
            enable_web_search=False,
            provided_sources=[
                {
                    "url": "https://www.globalxetfs.com/funds/botz/",
                    "title": "BOTZ",
                    "source_type": "primary_fact_source",
                    "extracted_text": "BOTZ 全名是 Global X Robotics & Artificial Intelligence ETF。",
                    "access_status": "accessible",
                }
            ],
        )
    )

    assert provider.extract_calls == 0
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.normalized_claim == "BOTZ 全名是 Global X Robotics & Artificial Intelligence ETF。"
    assert claim.original_span == claim.normalized_claim
    assert claim.claim_type == ClaimType.FACTUAL
    assert claim.claim_type != ClaimType.NON_CLAIM
    assert claim.citation_edges
    assert claim.citation_source_url == "https://www.globalxetfs.com/funds/botz/"
    assert claim.citation_edges[0].source_url == "https://www.globalxetfs.com/funds/botz/"
    assert result.metadata["citation_capture_mode"] == "text_only"
    assert result.metadata["citation_capture_coverage"] == "medium"
    assert result.metadata["structured_citation_count"] == 0
    assert result.metadata["text_fallback_citation_count"] == 1


def test_browser_dom_citations_are_used_without_text_parser_or_llm_extraction(monkeypatch):
    def fail_parse(_text):  # pragma: no cover - should not be called
        raise AssertionError("text parser should not run when structured citations are supplied")

    monkeypatch.setattr(analyzer_module, "parse_citations", fail_parse)
    monkeypatch.setattr(analyzer_module, "parse_reference_descriptions", fail_parse)

    provider = CountingProvider(support_outputs=_direct_fact_support)
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text="This visible text has no parseable marker.",
            input_mode="browser_dom",
            dom_citations=[
                {
                    "citation_id": "dom-1",
                    "marker_text": "[A]",
                    "source_url": "https://example.com/source",
                    "source_title": "DOM Source",
                    "source_label": "A",
                    "cited_text_span": "Revenue was $10 billion.",
                    "char_start": 0,
                    "char_end": 24,
                    "capture_method": "dom_anchor",
                    "confidence": "high",
                }
            ],
            enable_url_fetch=False,
            enable_web_search=True,
            provided_sources=[
                {
                    "url": "https://example.com/source",
                    "title": "DOM Source",
                    "source_type": "primary_fact_source",
                    "extracted_text": "Revenue was $10 billion.",
                    "access_status": "accessible",
                }
            ],
        )
    )

    assert provider.extract_calls == 0
    assert result.claims[0].normalized_claim == "Revenue was $10 billion."
    assert result.claims[0].citation_label == "A"
    assert result.claims[0].citation_source_url == "https://example.com/source"
    assert result.claims[0].citation_edges[0].capture_method == "dom_anchor"
    assert result.metadata["citation_capture_mode"] == "browser_dom"
    assert result.metadata["citation_capture_coverage"] == "high"
    assert result.metadata["structured_citation_count"] == 1
    assert result.metadata["text_fallback_citation_count"] == 0
    assert result.metadata["total_located_citations"] == 1


def test_citation_annotations_take_priority_over_dom_citations():
    provider = CountingProvider(support_outputs=_direct_fact_support)
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text="The text parser and DOM citation should be ignored in favor of API annotation.",
            input_mode="mixed",
            dom_citations=[
                {
                    "citation_id": "dom-ignored",
                    "marker_text": "[DOM]",
                    "source_url": "https://example.com/dom",
                    "source_title": "DOM Source",
                    "source_label": "DOM",
                    "cited_text_span": "DOM statement.",
                    "capture_method": "dom_anchor",
                }
            ],
            citation_annotations=[
                {
                    "citation_id": "api-used",
                    "marker_text": "[API]",
                    "source_url": "https://example.com/api",
                    "source_title": "API Source",
                    "source_label": "API",
                    "cited_text_span": "API annotated statement.",
                    "capture_method": "api_annotation",
                    "confidence": "high",
                }
            ],
            enable_url_fetch=False,
            enable_web_search=False,
            provided_sources=[
                {
                    "url": "https://example.com/api",
                    "title": "API Source",
                    "source_type": "primary_fact_source",
                    "extracted_text": "API annotated statement.",
                    "access_status": "accessible",
                }
            ],
        )
    )

    assert provider.extract_calls == 0
    assert len(result.claims) == 1
    assert result.claims[0].normalized_claim == "API annotated statement."
    assert result.claims[0].citation_source_url == "https://example.com/api"
    assert result.metadata["citation_capture_mode"] == "api_annotation"


def test_claim_extraction_timeout_is_ignored_in_default_citation_terminal_audit():
    text = (
        "BOTZ 全名是 Global X Robotics & Artificial Intelligence ETF。[1]\n\n"
        "来源指针\n"
        '[1]: https://www.globalxetfs.com/funds/botz/ "BOTZ"'
    )
    provider = CountingProvider(
        extraction_error=LLMProviderTimeoutError("boom"),
        support_outputs=_direct_fact_support,
    )
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=text,
            enable_url_fetch=False,
            enable_web_search=False,
            provided_sources=[
                {
                    "url": "https://www.globalxetfs.com/funds/botz/",
                    "title": "BOTZ",
                    "source_type": "primary_fact_source",
                    "extracted_text": "BOTZ 全名是 Global X Robotics & Artificial Intelligence ETF。",
                    "access_status": "accessible",
                }
            ],
        )
    )

    assert provider.extract_calls == 0
    assert all("LLM claim extraction timed out" not in claim.normalized_claim for claim in result.claims)
    assert result.display_citations[0].display_status != DisplayStatus.EXCLUDED_OR_CONTEXT
    assert result.citation_terminal_results[0].terminal_class != TerminalClass.UNRESOLVED


def test_source_pointer_entry_precedence_binds_botz_two_to_botz_not_aiq():
    text = (
        "BOTZ 费用率是 0.68%，资产规模约 37.4 亿美元。[2] "
        "AIQ 费用率同样是 0.68%，资产规模约 108.5 亿美元。[8]\n\n"
        "来源指针\n"
        "[2] BOTZ 的成立日期、费用率、资产规模和 NAV 来自 Global X 官方页面。([Global X ETFs][1])\n"
        "[8] AIQ 的费用率和资产规模来自 Global X 官方页面。([Global X ETFs][2])\n\n"
        '[1]: https://www.globalxetfs.com/funds/botz/ "BOTZ"\n'
        '[2]: https://www.globalxetfs.com/funds/aiq/ "AIQ"'
    )
    provider = CountingProvider(support_outputs=_direct_fact_support)
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=text,
            enable_url_fetch=False,
            enable_web_search=False,
            provided_sources=[
                {
                    "url": "https://www.globalxetfs.com/funds/botz/",
                    "title": "BOTZ",
                    "source_type": "primary_fact_source",
                    "extracted_text": "BOTZ 费用率是 0.68%，资产规模约 37.4 亿美元。",
                    "access_status": "accessible",
                },
                {
                    "url": "https://www.globalxetfs.com/funds/aiq/",
                    "title": "AIQ",
                    "source_type": "primary_fact_source",
                    "extracted_text": "AIQ 费用率同样是 0.68%，资产规模约 108.5 亿美元。",
                    "access_status": "accessible",
                },
            ],
        )
    )

    by_label = {claim.citation_label: claim for claim in result.claims}
    assert by_label["2"].citation_source_url == "https://www.globalxetfs.com/funds/botz/"
    assert by_label["8"].citation_source_url == "https://www.globalxetfs.com/funds/aiq/"


def test_source_pointer_without_url_is_unresolved_not_mismatch():
    text = (
        "BOTZ 当前价格约 40.56 美元。[7]\n\n"
        "来源指针\n"
        "[7] BOTZ 的当前价格来自实时行情工具，价格约 40.56 美元，时间为 2026 年 6 月 1 日。"
    )
    provider = CountingProvider(support_outputs=_direct_fact_support)
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=text,
            enable_url_fetch=False,
            enable_web_search=False,
        )
    )

    assert len(result.claims) == 1
    assert result.claims[0].citation_source_url is None
    assert result.citation_terminal_results[0].terminal_class == TerminalClass.UNRESOLVED
    assert result.citation_terminal_results[0].terminal_reason == "no_source_url"
    assert result.problematic_citations == []
    assert result.summary.key_rates.true_mismatch_rate == 0


def test_many_cited_statements_do_not_call_extract_claims_and_support_check_is_batched():
    statements = [f"指标 {index} 为 {index}。[{index}]" for index in range(1, 14)]
    references = [f'[{index}]: https://example.com/source-{index} "Source {index}"' for index in range(1, 14)]
    text = " ".join(statements) + "\n\n来源指针\n" + "\n".join(references)
    provider = CountingProvider(support_outputs=_direct_fact_support)
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=text,
            enable_url_fetch=False,
            enable_web_search=True,
            provided_sources=[
                {
                    "url": f"https://example.com/source-{index}",
                    "title": f"Source {index}",
                    "source_type": "primary_fact_source",
                    "extracted_text": f"指标 {index} 为 {index}。",
                    "access_status": "accessible",
                }
                for index in range(1, 14)
            ],
        )
    )

    assert result.metadata["cited_statement_count"] == 13
    assert len(result.claims) == 13
    assert provider.extract_calls == 0
    assert provider.support_batch_calls == 1


def test_chinese_paragraph_citation_edges_do_not_cross_bind_or_inflate_claims():
    text = (
        "Apple 2024 财年总净销售额为 3910.35 亿美元。[1] "
        "Microsoft 2024 财年营收为 2451 亿美元。[2] "
        "Nvidia 2025 财年营收为 1305 亿美元。[3] "
        "一篇市场评论认为，Apple 服务业务的经常性收入使它在大型科技股中具有更强防御性。[4] "
        "另一位投资策略作者认为，Nvidia 的估值已经反映了很多 AI 增长预期，短期风险回报不再便宜。[5] "
        "另外，某公司 2026 年第一季度自由现金流为 42 亿美元。[6]\n\n"
        "来源指针\n"
        f'[1]: {URLS["1"]} "Apple FY2024 financial statements"\n'
        f'[2]: {URLS["2"]} "Microsoft 2024 Annual Report"\n'
        f'[3]: {URLS["3"]} "NVIDIA 2025 Annual Report"\n'
        f'[4]: {URLS["4"]} "Apple Services Business Commentary"\n'
        f'[5]: {URLS["5"]} "NVIDIA Valuation Opinion"\n'
        f'[6]: {URLS["6"]} "Missing quarterly report"'
    )
    search_provider = FailingSearchProvider()
    analyzer = SourceGroundingAnalyzer(
        enable_url_fetch=False,
        search_provider=search_provider,
        llm_provider=MockLLMProvider(
            extraction_outputs=_claim_for_statement,
            support_outputs=_support_for_claim,
        ),
    )

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=text,
            enable_url_fetch=False,
            enable_web_search=True,
            provided_sources=[
                {
                    "url": URLS["1"],
                    "title": "Apple FY2024 financial statements",
                    "source_type": "primary_fact_source",
                    "extracted_text": "Apple 2024 财年总净销售额为 3910.35 亿美元。",
                    "access_status": "accessible",
                },
                {
                    "url": URLS["2"],
                    "title": "Microsoft 2024 Annual Report",
                    "source_type": "primary_fact_source",
                    "extracted_text": "Microsoft 2024 财年营收为 2451 亿美元。",
                    "access_status": "accessible",
                },
                {
                    "url": URLS["3"],
                    "title": "NVIDIA 2025 Annual Report",
                    "source_type": "primary_fact_source",
                    "extracted_text": "Nvidia 2025 财年营收为 1305 亿美元。",
                    "access_status": "accessible",
                },
                {
                    "url": URLS["4"],
                    "title": "Apple Services Business Commentary",
                    "source_type": "opinion_analysis",
                    "extracted_text": "一篇市场评论认为，Apple 服务业务的经常性收入使它在大型科技股中具有更强防御性。",
                    "access_status": "accessible",
                },
                {
                    "url": URLS["5"],
                    "title": "NVIDIA Valuation Opinion",
                    "source_type": "opinion_analysis",
                    "extracted_text": "另一位投资策略作者认为，Nvidia 的估值已经反映了很多 AI 增长预期，短期风险回报不再便宜。",
                    "access_status": "accessible",
                },
            ],
        )
    )

    assert result.metadata["cited_statement_count"] == 6
    assert result.metadata["citation_unit_count"] == 6
    assert len(result.claims) == 6
    assert result.metadata["search_query_count"] == 0
    assert search_provider.calls == 0

    labels = [claim.citation_label for claim in result.claims]
    assert labels == ["1", "2", "3", "4", "5", "6"]
    for claim in result.claims:
        assert claim.citation_label is not None
        assert claim.citation_source_url == URLS[claim.citation_label]
        assert len(claim.citation_edges) == 1
        assert claim.citation_edges[0].source_url == URLS[claim.citation_label]


def test_botz_golden_unresolved_reasons_and_opinion_with_fact_premise():
    text = (
        "BOTZ 的费用率是 0.68%，0.68% 对长期持有有影响。[2] "
        "BOTZ 前十大持仓接近六成，买太多会让你以为自己分散了。[3] "
        "根据 Global X 的资料，美国、日本、中国、瑞士都是重要权重来源。[5] "
        "按我查到的 BOTZ 当前价格约 40.56 美元。[7]\n\n"
        "来源指针\n"
        "[2] BOTZ 的费用率来自 Global X 官方页面。([Global X ETFs][1])\n"
        "[3] BOTZ 的前十大持仓和权重来自 Global X 官方页面。([Global X ETFs][1])\n"
        "[5] BOTZ 的国家分布来自页面截图，暂无可公开 URL。\n"
        "[7] BOTZ 当前价格来自实时行情工具，暂无可公开 URL。\n\n"
        '[1]: https://www.globalxetfs.com/funds/botz/ "BOTZ"\n'
        '[2]: https://www.globalxetfs.com/funds/aiq/ "AIQ"'
    )
    provider = CountingProvider(support_outputs=_direct_fact_support)
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=text,
            enable_url_fetch=False,
            enable_web_search=False,
            provided_sources=[
                {
                    "url": "https://www.globalxetfs.com/funds/botz/",
                    "title": "BOTZ",
                    "source_type": "primary_fact_source",
                    "extracted_text": (
                        "Global X Robotics & Artificial Intelligence ETF BOTZ. "
                        "The Fund's total expense ratio is 0.68%. "
                        "Top Holdings: NVIDIA 12.3%; Keyence 9.8%; ABB 7.0%; Fanuc 6.2%; "
                        "Intuitive Surgical 5.8%; Yaskawa 5.2%; SMC 4.8%; Daifuku 4.1%; "
                        "Inovance 3.7%; Zebra Technologies 3.3%. "
                        "Top ten holdings account for 61.2% of net assets."
                    ),
                    "access_status": "accessible",
                }
            ],
        )
    )

    by_label = {claim.citation_label: claim for claim in result.claims}
    assert by_label["2"].citation_source_url == "https://www.globalxetfs.com/funds/botz/"
    assert by_label["2"].citation_source_url != "https://www.globalxetfs.com/funds/aiq/"
    assert by_label["3"].citation_source_url == "https://www.globalxetfs.com/funds/botz/"

    terminals = {result.claims[index].citation_label: terminal for index, terminal in enumerate(result.citation_terminal_results)}
    assert terminals["2"].terminal_class == TerminalClass.OPINION
    assert terminals["2"].terminal_reason == "opinion_with_fact_premise"
    assert terminals["3"].terminal_class == TerminalClass.OPINION
    assert terminals["3"].terminal_reason == "opinion_with_fact_premise"
    assert terminals["5"].terminal_class == TerminalClass.UNRESOLVED
    assert terminals["5"].unresolved_reason == UnresolvedReason.NO_SOURCE_URL
    assert terminals["7"].terminal_class == TerminalClass.UNRESOLVED
    assert terminals["7"].unresolved_reason == UnresolvedReason.NO_SOURCE_URL
    assert TerminalClass.MISMATCH not in {item.terminal_class for item in result.citation_terminal_results}
    assert all(item.unresolved_reason != UnresolvedReason.TERMINAL_MAPPING_MISSING for item in result.citation_terminal_results)


def test_cuhk_analysis_from_official_premise_is_opinion_not_mismatch():
    text = (
        "港中深整体硬实力更强，这个判断主要来自 CUHK 和港中深共享的学术治理结构。[5]\n\n"
        "来源指针\n"
        '[5]: https://www.cuhk.edu.cn/en/article/4974 "CUHK Shenzhen academic governance"'
    )
    provider = CountingProvider(support_outputs=_direct_fact_support)
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text=text,
            enable_url_fetch=False,
            enable_web_search=False,
            provided_sources=[
                {
                    "url": "https://www.cuhk.edu.cn/en/article/4974",
                    "title": "CUHK Shenzhen academic governance",
                    "source_type": "primary_fact_source",
                    "extracted_text": (
                        "CUHK Shenzhen, The Chinese University of Hong Kong, Shenzhen is established under "
                        "the oversight of The Chinese University of Hong Kong Senate and shares "
                        "academic structures and curriculum governance."
                    ),
                    "access_status": "accessible",
                }
            ],
        )
    )

    assert result.problematic_citations == []
    assert result.citation_terminal_results[0].terminal_class == TerminalClass.OPINION
    assert result.citation_terminal_results[0].terminal_reason == "opinion_with_fact_premise"
    assert result.citation_terminal_results[0].unresolved_reason is None
    assert all(item.terminal_reason != "terminal_mapping_missing" for item in result.citation_terminal_results)


def test_stock_price_formation_golden_retrieves_bilingual_finance_snippets():
    text = (
        "**最近一次成交价格不一定等于你下市价单时真正成交的价格。**[1] "
        "股票代表公司所有权的一部分，投资者买股票的原因包括资本增值、分红以及投票权。[2] "
        "股价可以理解为未来收益折现并补偿风险后的结果。[3] "
        "分红是公司利润的一部分，支付给股东。[4] "
        "普通股股东在破产清算时通常排在债权人和优先股股东之后。[5]\n\n"
        "来源指针\n"
        "[1] Investor.gov 对市价单和限价单的解释：market order 按可得价格执行，"
        "last-traded price 不一定是成交价格，limit order 指定价格或更好。([Investor.gov][1])\n"
        "[2] Investor.gov 对股票的解释：股票代表公司所有权的一部分；"
        "投资者买股票的原因包括资本增值、分红以及投票权。([Investor.gov][2])\n"
        "[3] Federal Reserve 对资产估值的解释：asset price equals expected discounted value "
        "of future payoffs, and investors require risk compensation or risk premium.([Federal Reserve][3])\n"
        "[4] Investor.gov Dividend 词条：dividend is a portion of company profit paid to shareholders.([Investor.gov][4])\n"
        "[5] Investor.gov Stocks FAQ：common stockholders are last in line; bondholders and "
        "preferred stockholders are paid before common stockholders in bankruptcy liquidation.([Investor.gov][2])\n\n"
        '[1]: https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-14 "Order Types"\n'
        '[2]: https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks "Stocks FAQ"\n'
        '[3]: https://www.federalreserve.gov/publications/may-2021-asset-valuations.htm "Asset Valuations"\n'
        '[4]: https://www.investor.gov/introduction-investing/investing-basics/glossary/dividend "Dividend"'
    )
    captured_bundles = {}

    def support_with_snippet_assertions(claim: Claim, source_bundle):
        captured_bundles[claim.citation_label] = source_bundle
        assert source_bundle["snippet_retrieval_status"] in {"lexical_match", "semantic_match", "semantic_rerank_needed"}
        snippet_text = " ".join(snippet["text"] for snippet in source_bundle["evidence_snippets"]).lower()
        if claim.citation_label == "1":
            assert "market order" in snippet_text
            assert "last-traded price" in snippet_text
            assert "limit order" in snippet_text
            relation = SupportRelation.PARTIALLY_SUPPORTS
            bucket = FinalGroundingBucket.WEAK_FACT_GROUNDING
        elif claim.citation_label == "2":
            assert "share of ownership" in snippet_text
            assert "capital appreciation" in snippet_text
            assert "dividend payments" in snippet_text
            relation = SupportRelation.DIRECTLY_SUPPORTS
            bucket = FinalGroundingBucket.HARD_FACT_GROUNDING
        elif claim.citation_label == "3":
            assert "expected discounted value" in snippet_text
            assert "future payoffs" in snippet_text
            assert "risk premium" in snippet_text
            relation = SupportRelation.PARTIALLY_SUPPORTS
            bucket = FinalGroundingBucket.WEAK_FACT_GROUNDING
        elif claim.citation_label == "4":
            assert "portion of a company's profit paid to shareholders" in snippet_text
            relation = SupportRelation.DIRECTLY_SUPPORTS
            bucket = FinalGroundingBucket.HARD_FACT_GROUNDING
        else:
            assert "common stockholders are the last in line" in snippet_text
            assert "bondholders will be paid first" in snippet_text
            relation = SupportRelation.DIRECTLY_SUPPORTS
            bucket = FinalGroundingBucket.HARD_FACT_GROUNDING
        return {
            "support_relation": relation.value,
            "final_bucket": bucket.value,
            "risk_flags": [],
            "reasoning_summary": "The retrieved evidence snippets support the cited statement or its factual premise.",
            "evidence_quote": source_bundle["evidence_snippets"][0]["text"][:300],
            "source_role": SourceRole.PRIMARY_FACT_SOURCE.value,
        }

    provider = CountingProvider(support_outputs=support_with_snippet_assertions)
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, llm_provider=provider)
    result = analyzer.analyze(
        AnalysisRequest(
            input_text=text,
            enable_url_fetch=False,
            enable_web_search=False,
            provided_sources=[
                {
                    "url": "https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-14",
                    "title": "Investor Bulletin: Understanding Order Types | Investor.gov",
                    "source_type": "primary_fact_source",
                    "extracted_text": (
                        "MARKET, LIMIT and STOP ORDERS Market Order A market order is an order to buy or sell "
                        "a stock at the best available price. It is important for investors to remember that the "
                        "last-traded price is not necessarily the price at which a market order will be executed. "
                        "Limit Order A limit order is an order to buy or sell a stock at a specific price or better."
                    ),
                    "access_status": "accessible",
                },
                {
                    "url": "https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks",
                    "title": "Stocks - FAQs | Investor.gov",
                    "source_type": "primary_fact_source",
                    "extracted_text": (
                        "Stocks are a type of security that gives stockholders a share of ownership in a company. "
                        "Capital appreciation occurs when a stock rises in price. Dividend payments come when the "
                        "company distributes some of its earnings to stockholders. Preferred stockholders usually "
                        "receive dividend payments before common stockholders do, and have priority over common "
                        "stockholders if the company goes bankrupt and its assets are liquidated. The company's "
                        "bondholders will be paid first, then holders of preferred stock. If a company goes bankrupt "
                        "and its assets are liquidated, common stockholders are the last in line to share in the proceeds. "
                        "If you are a common stockholder, you get whatever is left, which may be nothing."
                    ),
                    "access_status": "accessible",
                },
                {
                    "url": "https://www.federalreserve.gov/publications/may-2021-asset-valuations.htm",
                    "title": "The Fed - 1. Asset Valuations",
                    "source_type": "primary_fact_source",
                    "extracted_text": (
                        "According to a long-standing theory, an asset's price should equal the expected discounted "
                        "value today of future payoffs from holding assets. The difference in the expected returns "
                        "between risky assets and Treasury securities is the risk premium investors expect to receive "
                        "as compensation for the risk they take. An increase in asset prices might reflect higher "
                        "expected future payoffs or a decline in interest rates."
                    ),
                    "access_status": "accessible",
                },
                {
                    "url": "https://www.investor.gov/introduction-investing/investing-basics/glossary/dividend",
                    "title": "Dividend | Investor.gov",
                    "source_type": "primary_fact_source",
                    "extracted_text": (
                        "Role of the SEC How to Submit Comments to the SEC Researching the Federal Securities Laws "
                        "Through the SEC Website The Laws That Govern the Securities Industry Dividend A portion "
                        "of a company's profit paid to shareholders. Public companies that pay dividends usually "
                        "do so on a fixed schedule."
                    ),
                    "access_status": "accessible",
                },
            ],
        )
    )

    assert provider.extract_calls == 0
    assert result.metadata["cited_statement_count"] == 5
    assert len(result.claims) == 5
    assert set(captured_bundles) == {"1", "2", "3", "4", "5"}
    assert not result.claims[0].normalized_claim.startswith("**")
    assert all(
        claim.support_relation != SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET
        for claim in result.claims
    )
    assert all(
        terminal.unresolved_reason != UnresolvedReason.NO_RELEVANT_SNIPPET
        for terminal in result.citation_terminal_results
    )
