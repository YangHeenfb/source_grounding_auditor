from app.evidence_snippet_retriever import (
    EvidenceSnippet,
    build_retrieval_query,
    clean_markdown_boundaries,
    retrieve_evidence_snippets_with_reason,
)
from app.semantic_snippet_reranker import rerank_candidate_snippets


ORDER_TYPES_TEXT = """
Investor Bulletin: Understanding Order Types.
MARKET, LIMIT and STOP ORDERS Market Order A market order is an order to buy or sell a stock at the best available price.
It is important for investors to remember that the last-traded price is not necessarily the price at which a market order will be executed.
Limit Order A limit order is an order to buy or sell a stock at a specific price or better.
"""


STOCKS_FAQ_TEXT = """
Stocks are a type of security that gives stockholders a share of ownership in a company.
Here are some reasons investors buy stocks: Capital appreciation, which occurs when a stock rises in price; Dividend payments, which come when the company distributes some of its earnings to stockholders; Ability to vote shares and influence the company.
Preferred stockholders usually do not have voting rights but they receive dividend payments before common stockholders do, and have priority over common stockholders if the company goes bankrupt and its assets are liquidated.
The company's bondholders will be paid first, then holders of preferred stock.
If a company goes bankrupt and its assets are liquidated, common stockholders are the last in line to share in the proceeds.
If you are a common stockholder, you get whatever is left, which may be nothing.
"""


FED_ASSET_VALUATIONS_TEXT = """
According to a long-standing theory, an asset's price should equal the expected discounted value today of future payoffs from holding assets.
Investors also want to be compensated for the relative risk of their investments.
The difference in the expected returns between risky assets and Treasury securities is the risk premium investors expect to receive as compensation for the risk they take.
An increase in asset prices might reflect higher expected future payoffs or a decline in interest rates, which raises the current value of those future payoffs.
"""


DIVIDEND_TEXT_WITH_NAV = """
Role of the SEC How to Submit Comments to the SEC Researching the Federal Securities Laws Through the SEC Website The Laws That Govern the Securities Industry Dividend A portion of a company's profit paid to shareholders.
Public companies that pay dividends usually do so on a fixed schedule although they can issue them at any time.
Unscheduled dividend payments are known as special dividends or extra dividends.
"""


def test_markdown_cleanup_and_order_type_retrieval_uses_source_pointer_context():
    cited_text = "**最近一次成交价格不一定等于你下市价单时真正成交的价格。**"
    result = retrieve_evidence_snippets_with_reason(
        cited_text,
        ORDER_TYPES_TEXT,
        source_pointer_description=(
            "Investor.gov 对市价单和限价单的解释：market order 按可得价格执行，"
            "last-traded price 不一定是成交价格，limit order 指定价格或更好。"
        ),
        source_title="Investor Bulletin: Understanding Order Types | Investor.gov",
        source_url="https://www.investor.gov/example/order-types",
    )

    assert clean_markdown_boundaries(cited_text).startswith("最近一次成交价格")
    assert not build_retrieval_query(cited_text).startswith("**")
    assert result.status == "lexical_match"
    snippet_text = " ".join(snippet.text for snippet in result.snippets).lower()
    assert "market order" in snippet_text
    assert "last-traded price" in snippet_text
    assert "limit order" in snippet_text


def test_stock_faq_bilingual_finance_hints_recall_ownership_appreciation_and_dividends():
    result = retrieve_evidence_snippets_with_reason(
        "股票代表公司所有权的一部分，投资者买股票的原因包括资本增值、分红以及投票权。",
        STOCKS_FAQ_TEXT,
        source_pointer_description=(
            "Investor.gov 对股票的解释：股票代表公司所有权的一部分；"
            "投资者买股票的原因包括资本增值、分红以及投票权。"
        ),
        source_title="Stocks - FAQs | Investor.gov",
        source_url="https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks",
    )

    assert result.status == "lexical_match"
    snippet_text = " ".join(snippet.text for snippet in result.snippets).lower()
    assert "share of ownership" in snippet_text
    assert "capital appreciation" in snippet_text
    assert "dividend payments" in snippet_text


def test_fed_asset_valuation_retrieval_recalls_discounted_future_payoffs_and_risk():
    result = retrieve_evidence_snippets_with_reason(
        "股价可以理解为未来收益折现并补偿风险后的结果。",
        FED_ASSET_VALUATIONS_TEXT,
        source_pointer_description=(
            "Federal Reserve 对资产估值的解释：asset price equals expected discounted value "
            "of future payoffs, and investors require risk compensation or risk premium."
        ),
        source_title="The Fed - 1. Asset Valuations",
        source_url="https://www.federalreserve.gov/publications/may-2021-asset-valuations.htm",
    )

    assert result.status == "lexical_match"
    snippet_text = " ".join(snippet.text for snippet in result.snippets).lower()
    assert "expected discounted value" in snippet_text
    assert "future payoffs" in snippet_text
    assert "risk premium" in snippet_text


def test_dividend_definition_is_not_filtered_out_by_navigation_noise():
    result = retrieve_evidence_snippets_with_reason(
        "分红是公司利润的一部分，支付给股东。",
        DIVIDEND_TEXT_WITH_NAV,
        source_pointer_description="Investor.gov Dividend 词条：dividend is a portion of company profit paid to shareholders.",
        source_title="Dividend | Investor.gov",
        source_url="https://www.investor.gov/introduction-investing/investing-basics/glossary/dividend",
    )

    assert result.status == "lexical_match"
    assert "portion of a company's profit paid to shareholders" in result.snippets[0].text.lower()


def test_common_stock_liquidation_context_ranks_specific_snippets():
    result = retrieve_evidence_snippets_with_reason(
        "普通股股东在破产清算时通常排在债权人和优先股股东之后。",
        STOCKS_FAQ_TEXT,
        source_pointer_description=(
            "Investor.gov Stocks FAQ：common stockholders are last in line; bondholders and "
            "preferred stockholders are paid before common stockholders in bankruptcy liquidation."
        ),
        source_title="Stocks - FAQs | Investor.gov",
        source_url="https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks",
    )

    assert result.status == "lexical_match"
    snippet_text = " ".join(snippet.text for snippet in result.snippets[:5]).lower()
    assert "common stockholders are the last in line" in snippet_text
    assert "bondholders will be paid first" in snippet_text


def test_lexical_miss_with_source_body_returns_semantic_rerank_candidates():
    result = retrieve_evidence_snippets_with_reason(
        "这是一个没有英文关键词的中文说明。",
        "First content paragraph with useful source context. Second content paragraph with additional details.",
        source_title="",
        source_url="",
    )

    assert result.status == "semantic_rerank_needed"
    assert result.snippets


def test_semantic_reranker_uses_bilingual_finance_bridge_without_inventing_text():
    response = rerank_candidate_snippets(
        cited_text="股票代表公司所有权的一部分，也可能带来资本增值和分红。",
        source_title="Stocks - FAQs | Investor.gov",
        source_pointer_description="Investor.gov 对股票的解释。",
        candidate_snippets=[
            EvidenceSnippet("Navigation and unrelated glossary text.", 0, []),
            EvidenceSnippet("Stocks give stockholders a share of ownership in a company.", 0, []),
            EvidenceSnippet("Capital appreciation and dividend payments are reasons investors buy stocks.", 0, []),
        ],
    )

    assert response.selected_snippet_indexes
    assert set(response.selected_snippet_indexes).issubset({1, 2})


def test_chinese_cloud_doc_sentence_with_vpn_title_is_not_dropped_as_title_like():
    source_text = (
        "VPN 连接 场景类_腾讯云 本页目录：通过 SSL VPN 是否可以访问 Internet？"
        " 腾讯云 VPN 可以作为代理吗？"
        " 腾讯云 VPN 连接在国家相关政策法规下提供服务，不提供访问 Internet 功能，"
        "禁止通过技术方式绕过网络审查访问境外网络，同时不提供代理功能。"
    )
    result = retrieve_evidence_snippets_with_reason(
        "腾讯云官方 VPN 文档明确说，其 VPN 连接不提供访问 Internet 功能，也不提供代理功能，并禁止通过技术方式绕过网络审查访问境外网络。",
        source_text,
        source_pointer_description=(
            "腾讯云 VPN 场景类文档明确说，不支持通过腾讯云 VPN 访问境外 Google，"
            "不提供访问 Internet 功能，禁止通过技术方式绕过网络审查访问境外网络，同时不提供代理功能。"
        ),
        source_title="VPN 连接 场景类_腾讯云",
        source_url="https://cloud.tencent.com/document/product/554/79786",
    )

    assert result.status == "lexical_match"
    assert "不提供访问 Internet 功能" in result.snippets[0].text
    assert "不提供代理功能" in result.snippets[0].text
