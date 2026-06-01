from pathlib import Path

from app.document_evidence_graph_builder import build_document_evidence_graph
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
):
    edge = EvidenceEdge(
        claim_id=claim_id,
        source_id=source_id,
        edge_type=EdgeType.AUTHOR_CITED,
        basis=EdgeBasis.FOOTNOTE,
        support_relation=relation,
        final_bucket=bucket,
        support_scope=scope,
        evidence_quote="Relevant source excerpt.",
    )
    return Claim(
        claim_id=claim_id,
        original_text_span=f"测试引用陈述 {claim_id}。[1]",
        normalized_claim=f"测试引用陈述 {claim_id}",
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
    )

    result = terminal(claim, DisplayStatus.AUDIT_LIMITED, {"s001": source()})

    assert result.terminal_class == TerminalClass.UNRESOLVED
    assert result.terminal_reason == "no_relevant_snippet"
    assert result.unresolved_reason == UnresolvedReason.NO_RELEVANT_SNIPPET


def test_accessible_source_no_support_terminal_mismatch():
    claim = make_claim(
        relation=SupportRelation.NO_SUPPORT,
        bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
    )
    result = terminal(claim, DisplayStatus.TRUE_CITATION_PROBLEM, {"s001": source()})

    assert result.terminal_class == TerminalClass.MISMATCH


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


def test_document_graph_merges_same_source_and_counts_edges():
    c1 = make_claim(claim_id="c001")
    c2 = make_claim(claim_id="c002")
    results = [
        terminal(c1, DisplayStatus.VERIFIED_FACT_SUPPORT, {"s001": source()}),
        terminal(c2, DisplayStatus.VERIFIED_FACT_SUPPORT, {"s001": source()}),
    ]
    graph = build_document_evidence_graph(results)

    source_nodes = [node for node in graph.nodes if node.type == "source" and node.label == "Source s001"]
    assert len(source_nodes) == 1
    assert source_nodes[0].count == 2
    assert len(source_nodes[0].metadata["cited_texts"]) == 2
    assert any(edge.type == "points_to_source" and edge.count == 2 for edge in graph.edges)


def test_default_ui_uses_terminal_pie_and_warning_badge_without_debug_tags():
    html = Path("backend/app/static/index.html").read_text()

    assert "引用最终落点" in html
    assert "terminalPie" in html
    assert "mismatchBadge" in html
    assert "legendRow('mismatch'" not in html
    assert "点击来源或终点节点查看具体引用" in html
    assert "debug_tags" not in html
    assert "problematicCitations" not in html
