from pathlib import Path

from app.document_evidence_graph_builder import build_document_evidence_graph
from app.review_queue_builder import build_review_queue
from app.terminal_classifier import (
    build_document_evidence_summary,
    classify_claim_terminal,
)
from app.schemas import (
    AccessStatus,
    Claim,
    ClaimType,
    DisplayCitationResult,
    DisplayStatus,
    DiscourseRole,
    EdgeBasis,
    EdgeType,
    EvidenceEdge,
    EvidenceExcerpt,
    FinalGroundingBucket,
    ImportanceLabel,
    Source,
    SourceType,
    SupportRelation,
    SupportScope,
    TerminalClass,
    UnresolvedReason,
)


def make_claim(
    *,
    claim_id="c001",
    source_id="s001",
    relation=SupportRelation.DIRECTLY_SUPPORTS,
    bucket=FinalGroundingBucket.HARD_FACT_GROUNDING,
    scope=SupportScope.OWN_INSTITUTIONAL_FACT,
    claim_type=ClaimType.FACTUAL,
    text=None,
    evidence_quote="Relevant source excerpt.",
):
    claim_text = text or f"测试引用陈述 {claim_id}"
    edge = EvidenceEdge(
        claim_id=claim_id,
        source_id=source_id,
        edge_type=EdgeType.AUTHOR_CITED,
        basis=EdgeBasis.FOOTNOTE,
        support_relation=relation,
        final_bucket=bucket,
        support_scope=scope,
        evidence_quote=evidence_quote,
    )
    return Claim(
        claim_id=claim_id,
        original_text_span=f"{claim_text}。[1]",
        normalized_claim=claim_text,
        claim_type=claim_type,
        discourse_role=DiscourseRole.ASSERTED_CLAIM,
        importance_label=ImportanceLabel.SUPPORTING,
        final_bucket=bucket,
        support_relation=relation,
        support_scope=scope,
        evidence_chain=[edge],
        evidence_quote=edge.evidence_quote,
        citation_label="1",
        citation_source_id=source_id,
        citation_source_url="https://example.com/source",
    )


def make_display(claim, status):
    return DisplayCitationResult(
        claim_id=claim.claim_id,
        display_claim_text=claim.original_text_span,
        display_status=status,
        display_label=status.value,
        display_explanation="display explanation",
    )


def source(source_id="s001", *, source_type=SourceType.PRIMARY_FACT_SOURCE, upstream=None, text="source text"):
    return Source(
        source_id=source_id,
        url=f"https://example.com/{source_id}",
        title=f"Source {source_id}",
        source_type=source_type,
        access_status=AccessStatus.ACCESSIBLE,
        extracted_text=text,
        upstream_source_ids=upstream or [],
    )


def terminal(claim, display_status, sources):
    return classify_claim_terminal(
        claim,
        make_display(claim, display_status),
        sources,
        max_terminal_trace_depth=2,
    )


def test_official_page_directly_supports_school_fact_terminal_fact():
    claim = make_claim()
    result = terminal(claim, DisplayStatus.VERIFIED_FACT_SUPPORT, {"s001": source()})

    assert result.terminal_class == TerminalClass.FACT
    assert result.terminal_reason == "verified_fact_support"
    assert result.best_evidence_excerpt is not None
    assert result.claim_source_comparison.comparison_label == "supports"


def test_blog_opinion_without_citation_terminal_opinion():
    claim = make_claim(
        relation=SupportRelation.BACKGROUND_ONLY,
        bucket=FinalGroundingBucket.WEAK_FACT_GROUNDING,
        scope=SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS,
        claim_type=ClaimType.JUDGMENT,
    )
    result = terminal(
        claim,
        DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES,
        {"s001": source(source_type=SourceType.OPINION_ANALYSIS)},
    )

    assert result.terminal_class == TerminalClass.OPINION
    assert result.terminal_reason == "opinion_with_fact_premise"


def test_blog_opinion_with_official_upstream_terminal_fact():
    claim = make_claim(
        relation=SupportRelation.BACKGROUND_ONLY,
        bucket=FinalGroundingBucket.WEAK_FACT_GROUNDING,
        scope=SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS,
        claim_type=ClaimType.JUDGMENT,
    )
    sources = {
        "s001": source(source_type=SourceType.OPINION_ANALYSIS, upstream=["s002"]),
        "s002": source("s002", source_type=SourceType.PRIMARY_FACT_SOURCE),
    }
    result = terminal(claim, DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES, sources)

    assert result.terminal_class == TerminalClass.OPINION
    assert result.terminal_reason == "opinion_with_fact_premise"
    assert result.depth == 0


def test_source_body_missing_terminal_unresolved():
    claim = make_claim(
        relation=SupportRelation.INACCESSIBLE,
        bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
    )
    result = terminal(
        claim,
        DisplayStatus.AUDIT_LIMITED,
        {"s001": source(text="")},
    )

    assert result.terminal_class == TerminalClass.UNRESOLVED
    assert result.terminal_reason == "source_body_missing"
    assert result.unresolved_reason == UnresolvedReason.SOURCE_BODY_MISSING


def test_missing_source_url_terminal_unresolved_reason_no_source_url():
    claim = make_claim(
        relation=SupportRelation.INACCESSIBLE,
        bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
    )
    claim.citation_source_url = None

    result = terminal(claim, DisplayStatus.AUDIT_LIMITED, {})

    assert result.terminal_class == TerminalClass.UNRESOLVED
    assert result.terminal_reason == "no_source_url"
    assert result.unresolved_reason == UnresolvedReason.NO_SOURCE_URL


def test_no_relevant_snippet_terminal_unresolved_reason_no_relevant_snippet():
    claim = make_claim(
        relation=SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
        bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
        evidence_quote="",
    )

    result = terminal(claim, DisplayStatus.AUDIT_LIMITED, {"s001": source()})

    assert result.terminal_class == TerminalClass.UNRESOLVED
    assert result.terminal_reason == "no_relevant_snippet"
    assert result.unresolved_reason == UnresolvedReason.NO_RELEVANT_SNIPPET
    assert result.best_evidence_excerpt is None
    assert result.claim_source_comparison.comparison_label == "no_relevant_excerpt"


def test_accessible_source_no_support_terminal_mismatch():
    claim = make_claim(
        relation=SupportRelation.NO_SUPPORT,
        bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
    )
    result = terminal(claim, DisplayStatus.TRUE_CITATION_PROBLEM, {"s001": source()})

    assert result.terminal_class == TerminalClass.MISMATCH
    assert result.claim_source_comparison.comparison_label == "weaker_than_claim"
    assert "Relevant source excerpt" in result.claim_source_comparison.source_says


def test_opinion_with_fact_premise_has_source_excerpt_comparison():
    claim = make_claim(
        relation=SupportRelation.PARTIALLY_SUPPORTS,
        bucket=FinalGroundingBucket.WEAK_FACT_GROUNDING,
        scope=SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS,
        claim_type=ClaimType.JUDGMENT,
        text="0.68% 对长期持有有影响",
        evidence_quote="The expense ratio is 0.68%.",
    )
    result = terminal(claim, DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES, {"s001": source()})

    assert result.terminal_class == TerminalClass.OPINION
    assert result.best_evidence_excerpt is not None
    assert "expense ratio" in result.best_evidence_excerpt.text
    assert result.claim_source_comparison.comparison_label == "supports_fact_premise"
    assert "source 支持事实前提" in result.claim_source_comparison.gap


def test_mismatch_comparison_can_use_explicit_weaker_excerpt():
    claim = make_claim(
        relation=SupportRelation.NO_SUPPORT,
        bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
        text="研究证明 X 导致 Y",
        evidence_quote="The study found an association between X and Y.",
    )
    result = terminal(claim, DisplayStatus.TRUE_CITATION_PROBLEM, {"s001": source()})

    assert result.terminal_class == TerminalClass.MISMATCH
    assert result.best_evidence_excerpt.text == "The study found an association between X and Y."
    assert result.claim_source_comparison.comparison_label == "weaker_than_claim"


def test_no_source_url_comparison_is_source_unavailable_without_quote_block():
    claim = make_claim(
        relation=SupportRelation.INACCESSIBLE,
        bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
        evidence_quote="",
    )
    claim.citation_source_url = None
    claim.evidence_chain = []
    result = terminal(claim, DisplayStatus.AUDIT_LIMITED, {})

    assert result.terminal_class == TerminalClass.UNRESOLVED
    assert result.best_evidence_excerpt is None
    assert result.claim_source_comparison.comparison_label == "source_unavailable"
    assert result.claim_source_comparison.source_says == ""


def test_terminal_result_preserves_multiple_evidence_excerpts():
    claim = make_claim()
    claim.evidence_quote = ""
    claim.evidence_excerpts = [
        EvidenceExcerpt(excerpt_id="e1", source_id="s001", source_title="Source", text="First source excerpt.", excerpt_role="direct_support"),
        EvidenceExcerpt(excerpt_id="e2", source_id="s001", source_title="Source", text="Second source excerpt.", excerpt_role="direct_support"),
    ]
    result = terminal(claim, DisplayStatus.VERIFIED_FACT_SUPPORT, {"s001": source()})

    assert result.best_evidence_excerpt.text == "First source excerpt."
    assert [excerpt.text for excerpt in result.evidence_excerpts] == [
        "First source excerpt.",
        "Second source excerpt.",
    ]


def test_summary_pie_ratios_exclude_mismatch():
    results = [
        terminal(make_claim(claim_id="c001"), DisplayStatus.VERIFIED_FACT_SUPPORT, {"s001": source()}),
        terminal(
            make_claim(claim_id="c002", relation=SupportRelation.BACKGROUND_ONLY, scope=SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS, claim_type=ClaimType.JUDGMENT),
            DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES,
            {"s001": source(source_type=SourceType.OPINION_ANALYSIS)},
        ),
        terminal(
            make_claim(claim_id="c003", relation=SupportRelation.INACCESSIBLE, bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH),
            DisplayStatus.AUDIT_LIMITED,
            {"s001": source(text="")},
        ),
        terminal(
            make_claim(claim_id="c004", relation=SupportRelation.NO_SUPPORT, bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH),
            DisplayStatus.TRUE_CITATION_PROBLEM,
            {"s001": source()},
        ),
    ]
    summary = build_document_evidence_summary(results)

    assert summary.total_cited_statements == 4
    assert summary.mismatch_count == 1
    assert summary.fact_terminal_ratio == 0.3333
    assert summary.opinion_terminal_ratio == 0.3333
    assert summary.unresolved_ratio == 0.3333


def test_review_queue_defaults_to_mismatch_and_opinion_only():
    results = [
        terminal(make_claim(claim_id="c001"), DisplayStatus.VERIFIED_FACT_SUPPORT, {"s001": source()}),
        terminal(
            make_claim(claim_id="c002", relation=SupportRelation.BACKGROUND_ONLY, scope=SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS, claim_type=ClaimType.JUDGMENT),
            DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES,
            {"s001": source(source_type=SourceType.OPINION_ANALYSIS)},
        ),
        terminal(
            make_claim(claim_id="c003", relation=SupportRelation.INACCESSIBLE, bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH),
            DisplayStatus.AUDIT_LIMITED,
            {"s001": source(text="")},
        ),
        terminal(
            make_claim(claim_id="c004", relation=SupportRelation.NO_SUPPORT, bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH),
            DisplayStatus.TRUE_CITATION_PROBLEM,
            {"s001": source()},
        ),
    ]

    queue = build_review_queue(results)

    assert [item.terminal_class for item in queue.needs_review] == [
        TerminalClass.MISMATCH,
        TerminalClass.OPINION,
    ]
    assert all("测试引用陈述" in item.cited_text for item in queue.needs_review)
    assert queue.needs_review[0].best_evidence_excerpt is not None
    assert queue.needs_review[0].claim_source_comparison is not None
    assert [item.terminal_class for item in queue.verified_fact.items] == [TerminalClass.FACT]
    assert queue.verified_fact.default_collapsed is True
    assert queue.unresolved[0].unresolved_reason == UnresolvedReason.SOURCE_BODY_MISSING


def test_document_graph_is_claim_terminal_tree_with_cited_statement_nodes():
    c1 = make_claim(claim_id="c001")
    c2 = make_claim(claim_id="c002")
    results = [
        terminal(c1, DisplayStatus.VERIFIED_FACT_SUPPORT, {"s001": source()}),
        terminal(
            c2,
            DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES,
            {"s001": source(source_type=SourceType.OPINION_ANALYSIS)},
        ),
    ]
    graph = build_document_evidence_graph(results)

    statement_nodes = [node for node in graph.nodes if node.type == "cited_statement"]
    assert len(statement_nodes) == 2
    assert any("测试引用陈述 c001" in node.label for node in statement_nodes)
    assert all(node.label != "2 条引用" for node in statement_nodes)
    assert any(node.id == "terminal:fact" and node.metadata["default_expanded"] is False for node in graph.nodes)
    assert any(node.id == "terminal:opinion" and node.metadata["default_expanded"] is True for node in graph.nodes)
    assert any(edge.type == "terminal_contains_statement" for edge in graph.edges)
    assert any(node.type == "reason" for node in graph.nodes)


def test_default_ui_uses_terminal_pie_and_warning_badge_without_debug_tags():
    html = Path("backend/app/static/index.html").read_text()

    assert "引用最终落点" in html
    assert "复核建议" in html
    assert "需要人工复核" in html
    assert "引用来源片段" in html
    assert "原文 claim" in html
    assert "引用来源" in html
    assert "source 片段 / 结论" in html
    assert "对比结论" in html
    assert "未找到可展示的 source 片段" in html
    assert "证据终点图" in html
    assert "terminalPie" in html
    assert "mismatchBadge" in html
    assert "legendRow('mismatch'" not in html
    assert "查看具体引用" not in html
    assert "聚合证据树" not in html
    assert "共分析" not in html
    assert "条最终落到事实来源" not in html
    assert "debug_tags" not in html
    assert "problematicCitations" not in html
