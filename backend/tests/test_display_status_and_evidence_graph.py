from pathlib import Path

from app.display_status_mapper import map_claim_to_display_result
from app.evidence_graph_builder import build_evidence_graph
from app.schemas import (
    AccessStatus,
    Claim,
    ClaimReviewCategory,
    ClaimType,
    DisplayStatus,
    DiscourseRole,
    EdgeBasis,
    EdgeType,
    EvidenceEdge,
    FinalGroundingBucket,
    ImportanceLabel,
    RiskFlag,
    Source,
    SupportRelation,
    SupportScope,
)


def claim(
    *,
    relation=SupportRelation.DIRECTLY_SUPPORTS,
    bucket=FinalGroundingBucket.HARD_FACT_GROUNDING,
    claim_type=ClaimType.FACTUAL,
    role=DiscourseRole.ASSERTED_CLAIM,
    category=ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK,
    flags=None,
    scope=SupportScope.UNKNOWN,
    evidence_quote="evidence",
):
    edge = EvidenceEdge(
        claim_id="c001",
        source_id="s001",
        edge_type=EdgeType.AUTHOR_CITED,
        basis=EdgeBasis.FOOTNOTE,
        support_relation=relation,
        final_bucket=bucket,
        support_scope=scope,
        evidence_quote=evidence_quote,
        reasoning_summary="Internal reasoning summary.",
    )
    return Claim(
        claim_id="c001",
        original_text_span="Claim text [1].",
        normalized_claim="Claim text",
        claim_type=claim_type,
        discourse_role=role,
        importance_label=ImportanceLabel.SUPPORTING,
        final_bucket=bucket,
        support_relation=relation,
        review_category=category,
        risk_flags=flags or [],
        support_scope=scope,
        evidence_chain=[edge],
        evidence_quote=evidence_quote,
        citation_label="1",
        citation_source_url="https://example.com/source",
    )


def source(access_status=AccessStatus.ACCESSIBLE, text="evidence text") -> Source:
    return Source(
        source_id="s001",
        url="https://example.com/source",
        title="Example source",
        access_status=access_status,
        extracted_text=text,
    )


def test_inaccessible_and_no_relevant_snippet_display_as_audit_limited():
    item = map_claim_to_display_result(
        claim(
            relation=SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
            bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
            category=ClaimReviewCategory.AUDIT_LIMITED,
            flags=[RiskFlag.INACCESSIBLE_SOURCE, RiskFlag.AUDIT_LIMITED_NO_RELEVANT_SNIPPET],
            evidence_quote="",
        )
    )

    assert item.display_status == DisplayStatus.AUDIT_LIMITED
    assert item.display_label == "审计受限"
    assert not item.should_count_as_true_mismatch
    assert "asserted_claim" in item.debug_tags


def test_no_support_with_usable_source_body_display_as_true_problem():
    item = map_claim_to_display_result(
        claim(
            relation=SupportRelation.NO_SUPPORT,
            bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
            category=ClaimReviewCategory.HIGH_RISK,
        )
    )

    assert item.display_status == DisplayStatus.TRUE_CITATION_PROBLEM
    assert item.should_show_in_problematic
    assert item.should_count_as_true_mismatch


def test_premise_support_for_analysis_display_is_not_mismatch():
    item = map_claim_to_display_result(
        claim(
            relation=SupportRelation.PARTIALLY_SUPPORTS,
            bucket=FinalGroundingBucket.WEAK_FACT_GROUNDING,
            claim_type=ClaimType.JUDGMENT,
            role=DiscourseRole.JUDGMENT_OR_ANALYSIS,
            scope=SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS,
        )
    )

    assert item.display_status == DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES
    assert not item.should_count_as_true_mismatch


def test_attribution_only_display_is_attribution_support():
    item = map_claim_to_display_result(
        claim(
            relation=SupportRelation.ATTRIBUTION_ONLY,
            bucket=FinalGroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING,
            claim_type=ClaimType.ATTRIBUTION,
            role=DiscourseRole.ATTRIBUTION_REPORT,
            category=ClaimReviewCategory.ATTRIBUTION_SUPPORTED,
        )
    )

    assert item.display_status == DisplayStatus.ATTRIBUTION_SUPPORT


def test_direct_hard_grounding_display_is_verified_fact_support():
    item = map_claim_to_display_result(claim())

    assert item.display_status == DisplayStatus.VERIFIED_FACT_SUPPORT
    assert item.display_label == "事实支撑成立"


def test_display_claim_text_preserves_original_chinese_when_normalized_claim_is_english():
    c = claim()
    c.original_span = "DKU 本科完成后会拿到 DKU 的中国学位和 Duke University 的学位，并成为两校校友。[3]"
    c.original_text_span = c.original_span
    c.normalized_claim = "DKU students receive both a DKU Chinese degree and a Duke University degree."

    item = map_claim_to_display_result(c)
    graph = build_evidence_graph(c, {"s001": source()})

    assert item.display_claim_text == c.original_span
    claim_node = next(node for node in graph.nodes if node.type == "claim")
    assert "DKU 本科完成后" in claim_node.label
    assert "students receive" not in claim_node.label


def test_each_claim_can_generate_evidence_graph():
    graph = build_evidence_graph(claim(), {"s001": source()})

    assert graph.claim_id == "c001"
    assert {node.type for node in graph.nodes} >= {"claim", "citation", "source", "evidence"}
    subtitles = {node.type: node.subtitle for node in graph.nodes}
    assert subtitles["claim"] == "待审计主张"
    assert subtitles["citation"] == "引用标记"
    assert subtitles["evidence"] == "证据片段"
    assert {edge.label for edge in graph.edges} >= {"引用", "指向来源", "支撑"}


def test_audit_limited_graph_contains_missing_evidence_node():
    c = claim(
        relation=SupportRelation.INACCESSIBLE,
        bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
        category=ClaimReviewCategory.AUDIT_LIMITED,
        flags=[RiskFlag.INACCESSIBLE_SOURCE],
        evidence_quote="",
    )
    graph = build_evidence_graph(c, {"s001": source(access_status=AccessStatus.FAILED, text="")})

    assert "missing_evidence" in {node.type for node in graph.nodes}


def test_true_problem_graph_edge_marks_no_support():
    graph = build_evidence_graph(
        claim(
            relation=SupportRelation.NO_SUPPORT,
            bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
            category=ClaimReviewCategory.HIGH_RISK,
        ),
        {"s001": source()},
    )

    assert any(edge.relation == "no_support" for edge in graph.edges)


def test_ui_hides_risk_flags_behind_technical_details():
    html = Path("backend/app/static/index.html").read_text()

    assert "查看技术细节" in html
    assert html.index("查看技术细节") < html.index("risk_flags")
    assert "风险标签" not in html
    assert "不可验证或错配" not in html
    assert "Claim 明细" not in html
    assert "Ratios basis" not in html
