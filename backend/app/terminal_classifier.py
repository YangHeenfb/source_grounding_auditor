from __future__ import annotations

from collections import Counter
import re

from .display_status_mapper import display_claim_text_for_claim
from .schemas import (
    AccessStatus,
    CitationTerminalResult,
    Claim,
    ClaimSourceComparison,
    ClaimType,
    DisplayCitationResult,
    DisplayStatus,
    DocumentEvidenceSummary,
    EvidenceExcerpt,
    RiskFlag,
    Source,
    SourceRole,
    SourceRoleForClaim,
    SourceType,
    SupportRelation,
    SupportScope,
    TerminalClass,
    UnresolvedReason,
)


FACT_SOURCE_TYPES = {
    SourceType.PRIMARY_FACT_SOURCE,
    SourceType.EVIDENCE_SYNTHESIS,
}
FACT_SOURCE_ROLES = {
    SourceRole.PRIMARY_FACT_SOURCE,
    SourceRole.OFFICIAL_ANNOUNCEMENT,
}
FACT_SOURCE_ROLES_FOR_CLAIM = {
    SourceRoleForClaim.OFFICIAL_INSTITUTION_SOURCE,
    SourceRoleForClaim.OFFICIAL_COMPANY_SOURCE,
    SourceRoleForClaim.REGULATORY_OR_FILING_SOURCE,
    SourceRoleForClaim.SCHOLARLY_PRIMARY_SOURCE,
    SourceRoleForClaim.EVIDENCE_SYNTHESIS_SOURCE,
}
FACT_SUPPORT_SCOPES = {
    SupportScope.OWN_INSTITUTIONAL_FACT,
    SupportScope.OWN_PRODUCT_OR_PROGRAM_FACT,
    SupportScope.OWN_REPORTED_DATA,
    SupportScope.OFFICIAL_ANNOUNCEMENT,
    SupportScope.ATTRIBUTION_ONLY,
}
OPINION_SOURCE_TYPES = {
    SourceType.OPINION_ANALYSIS,
    SourceType.SECONDARY_REPORTING,
    SourceType.ANONYMOUS_OR_OPAQUE,
}


def classify_citation_terminals(
    claims: list[Claim],
    display_citations: list[DisplayCitationResult],
    sources: dict[str, Source] | list[Source],
    *,
    max_terminal_trace_depth: int = 2,
) -> list[CitationTerminalResult]:
    source_lookup = sources if isinstance(sources, dict) else {source.source_id: source for source in sources}
    display_by_claim_id = {item.claim_id: item for item in display_citations}
    return [
        classify_claim_terminal(
            claim,
            display_by_claim_id.get(claim.claim_id),
            source_lookup,
            max_terminal_trace_depth=max_terminal_trace_depth,
        )
        for claim in claims
    ]


def classify_claim_terminal(
    claim: Claim,
    display: DisplayCitationResult | None,
    sources_by_id: dict[str, Source],
    *,
    max_terminal_trace_depth: int = 2,
) -> CitationTerminalResult:
    display_status = display.display_status if display else DisplayStatus.AUDIT_LIMITED
    primary_source = _primary_source_for_claim(claim, sources_by_id)
    source_title = primary_source.title if primary_source else claim.citation_source_title
    source_url = primary_source.url if primary_source else claim.citation_source_url
    path_nodes = [_citation_path_node(claim), _source_path_node(primary_source, claim)]
    path_edges = [
        {
            "source": f"citation:{claim.claim_id}",
            "target": _source_node_id(primary_source, claim),
            "type": "points_to_source",
            "label": "指向来源",
        }
    ]

    terminal_class, reason, depth, unresolved_reason = _terminal_class_for_claim(
        claim,
        display_status,
        primary_source,
        sources_by_id,
        path_nodes,
        path_edges,
        max_terminal_trace_depth=max_terminal_trace_depth,
    )
    excerpts = _evidence_excerpts_for_claim(claim, primary_source)
    best_excerpt = excerpts[0] if excerpts else None
    if best_excerpt:
        evidence_node_id = f"evidence:{claim.claim_id}:1"
        path_nodes.append(
            {
                "id": evidence_node_id,
                "type": "evidence",
                "label": best_excerpt.text,
                "source_id": best_excerpt.source_id,
                "excerpt_role": best_excerpt.excerpt_role,
            }
        )
        path_edges.append(
            {
                "source": _last_source_node_id(path_nodes),
                "target": evidence_node_id,
                "type": "source_has_evidence",
                "label": "来源片段",
            }
        )
    path_nodes.append(
        {
            "id": f"terminal:{terminal_class.value}",
            "type": _terminal_node_type(terminal_class),
            "label": _terminal_label(terminal_class),
            "terminal_class": terminal_class.value,
        }
    )
    path_edges.append(
        {
            "source": _last_source_node_id(path_nodes),
            "target": f"terminal:{terminal_class.value}",
            "type": _terminal_edge_type(terminal_class),
            "label": _terminal_label(terminal_class),
        }
    )
    return CitationTerminalResult(
        citation_id=claim.claim_id,
        cited_text=display.display_claim_text if display and display.display_claim_text else display_claim_text_for_claim(claim),
        citation_label=claim.citation_label,
        source_title=source_title or "",
        source_url=source_url,
        terminal_class=terminal_class,
        terminal_reason=reason,
        unresolved_reason=unresolved_reason,
        path_nodes=path_nodes,
        path_edges=path_edges,
        depth=depth,
        short_explanation=_short_explanation(terminal_class, reason),
        debug_claim_ids=[claim.claim_id],
        debug_tags=_dedupe((display.debug_tags if display else []) + _terminal_debug_tags(claim)),
        best_evidence_excerpt=best_excerpt,
        evidence_excerpts=excerpts,
        claim_source_comparison=_claim_source_comparison(
            claim=claim,
            terminal_class=terminal_class,
            terminal_reason=reason,
            unresolved_reason=unresolved_reason,
            source=primary_source,
            best_excerpt=best_excerpt,
        ),
    )


def build_document_evidence_summary(results: list[CitationTerminalResult]) -> DocumentEvidenceSummary:
    total = len(results)
    counts = Counter(result.terminal_class for result in results)
    pie_denom = (
        counts[TerminalClass.FACT]
        + counts[TerminalClass.OPINION]
        + counts[TerminalClass.UNRESOLVED]
    )
    return DocumentEvidenceSummary(
        total_cited_statements=total,
        fact_terminal_count=counts[TerminalClass.FACT],
        opinion_terminal_count=counts[TerminalClass.OPINION],
        unresolved_terminal_count=counts[TerminalClass.UNRESOLVED],
        mismatch_count=counts[TerminalClass.MISMATCH],
        fact_terminal_ratio=_ratio(counts[TerminalClass.FACT], pie_denom),
        opinion_terminal_ratio=_ratio(counts[TerminalClass.OPINION], pie_denom),
        unresolved_ratio=_ratio(counts[TerminalClass.UNRESOLVED], pie_denom),
    )


def _terminal_class_for_claim(
    claim: Claim,
    display_status: DisplayStatus,
    source: Source | None,
    sources_by_id: dict[str, Source],
    path_nodes: list[dict],
    path_edges: list[dict],
    *,
    max_terminal_trace_depth: int,
) -> tuple[TerminalClass, str, int, UnresolvedReason | None]:
    if _is_cited_span_parse_error(display_claim_text_for_claim(claim)):
        return (
            TerminalClass.UNRESOLVED,
            UnresolvedReason.CITED_SPAN_PARSE_ERROR.value,
            0,
            UnresolvedReason.CITED_SPAN_PARSE_ERROR,
        )
    if display_status == DisplayStatus.TRUE_CITATION_PROBLEM:
        return TerminalClass.MISMATCH, "source_citation_mismatch", 0, None
    if display_status == DisplayStatus.AUDIT_LIMITED:
        reason = _unresolved_reason(claim, source)
        return TerminalClass.UNRESOLVED, reason.value, 0, reason
    if display_status == DisplayStatus.VERIFIED_FACT_SUPPORT:
        return TerminalClass.FACT, "verified_fact_support", 0, None
    if display_status == DisplayStatus.PARTIAL_OR_WEAK_SUPPORT:
        if _is_opinion_or_analysis_claim(claim):
            return TerminalClass.OPINION, "opinion_with_fact_premise", 0, None
        if _claim_has_fact_source_support(claim, source):
            return TerminalClass.FACT, "fact_source_partial_support", 0, None
        terminal, reason, depth = _trace_opinion_or_return_opinion(
            source,
            sources_by_id,
            path_nodes,
            path_edges,
            reason_without_upstream="weak_or_partial_opinion_support",
            max_terminal_trace_depth=max_terminal_trace_depth,
        )
        return terminal, reason, depth, None
    if display_status == DisplayStatus.ATTRIBUTION_SUPPORT:
        if _attribution_lands_on_opinion(claim, source):
            terminal, reason, depth = _trace_opinion_or_return_opinion(
                source,
                sources_by_id,
                path_nodes,
                path_edges,
                reason_without_upstream="opinion_attribution",
                max_terminal_trace_depth=max_terminal_trace_depth,
            )
            return terminal, reason, depth, None
        return TerminalClass.FACT, "fact_about_attribution", 0, None
    if display_status == DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES:
        return TerminalClass.OPINION, "opinion_with_fact_premise", 0, None
    if claim.support_relation == SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET:
        return (
            TerminalClass.UNRESOLVED,
            UnresolvedReason.NO_RELEVANT_SNIPPET.value,
            0,
            UnresolvedReason.NO_RELEVANT_SNIPPET,
        )
    if claim.support_relation == SupportRelation.INACCESSIBLE:
        reason = _unresolved_reason(claim, source)
        return TerminalClass.UNRESOLVED, reason.value, 0, reason
    if claim.support_relation in {
        SupportRelation.PARTIALLY_SUPPORTS,
        SupportRelation.SUPPORTS_WEAKER_CLAIM,
        SupportRelation.BACKGROUND_ONLY,
        SupportRelation.OPINION_ONLY,
    }:
        return TerminalClass.OPINION, "opinion_with_fact_premise", 0, None
    if claim.support_relation == SupportRelation.DIRECTLY_SUPPORTS:
        if _is_opinion_or_analysis_claim(claim):
            return TerminalClass.OPINION, "opinion_with_fact_premise", 0, None
        if _claim_has_fact_source_support(claim, source):
            return TerminalClass.FACT, "verified_fact_support", 0, None
    return (
        TerminalClass.UNRESOLVED,
        UnresolvedReason.TERMINAL_MAPPING_MISSING.value,
        0,
        UnresolvedReason.TERMINAL_MAPPING_MISSING,
    )


def _trace_opinion_or_return_opinion(
    source: Source | None,
    sources_by_id: dict[str, Source],
    path_nodes: list[dict],
    path_edges: list[dict],
    *,
    reason_without_upstream: str,
    max_terminal_trace_depth: int,
) -> tuple[TerminalClass, str, int]:
    if not source:
        return TerminalClass.OPINION, reason_without_upstream, 0
    terminal, depth = _trace_upstream_for_fact_source(
        source,
        sources_by_id,
        path_nodes,
        path_edges,
        max_depth=max_terminal_trace_depth,
    )
    if terminal == TerminalClass.FACT:
        return TerminalClass.FACT, "opinion_source_grounded_in_fact_upstream", depth
    return TerminalClass.OPINION, reason_without_upstream, depth


def _trace_upstream_for_fact_source(
    source: Source,
    sources_by_id: dict[str, Source],
    path_nodes: list[dict],
    path_edges: list[dict],
    *,
    max_depth: int,
    depth: int = 0,
    seen: set[str] | None = None,
) -> tuple[TerminalClass, int]:
    seen = seen or set()
    if depth >= max_depth or source.source_id in seen:
        return TerminalClass.OPINION, depth
    seen.add(source.source_id)
    best_depth = depth
    for upstream_id in source.upstream_source_ids:
        upstream = sources_by_id.get(upstream_id)
        if upstream is None:
            continue
        path_nodes.append(_source_path_node(upstream, None, upstream=True))
        path_edges.append(
            {
                "source": f"source:{source.source_id}",
                "target": f"source:{upstream.source_id}",
                "type": "source_cites_upstream",
                "label": "上游来源",
            }
        )
        if _source_is_fact_like(upstream):
            return TerminalClass.FACT, depth + 1
        terminal, child_depth = _trace_upstream_for_fact_source(
            upstream,
            sources_by_id,
            path_nodes,
            path_edges,
            max_depth=max_depth,
            depth=depth + 1,
            seen=seen,
        )
        best_depth = max(best_depth, child_depth)
        if terminal == TerminalClass.FACT:
            return TerminalClass.FACT, child_depth
    return TerminalClass.OPINION, best_depth


def _primary_source_for_claim(claim: Claim, sources_by_id: dict[str, Source]) -> Source | None:
    for edge in claim.evidence_chain:
        if edge.source_id and edge.source_id in sources_by_id:
            return sources_by_id[edge.source_id]
    if claim.citation_source_id and claim.citation_source_id in sources_by_id:
        return sources_by_id[claim.citation_source_id]
    for source_id in claim.linked_source_ids:
        if source_id in sources_by_id:
            return sources_by_id[source_id]
    return None


def _evidence_excerpts_for_claim(claim: Claim, source: Source | None) -> list[EvidenceExcerpt]:
    excerpts = list(claim.evidence_excerpts)
    for edge in claim.evidence_chain:
        excerpts.extend(edge.evidence_excerpts)
        if edge.best_evidence_excerpt:
            excerpts.append(edge.best_evidence_excerpt)
    if not excerpts and claim.evidence_quote:
        excerpts.append(
            EvidenceExcerpt(
                excerpt_id=f"{claim.claim_id}:quote:1",
                source_id=source.source_id if source else claim.citation_source_id,
                source_title=(source.title if source else claim.citation_source_title) or "",
                source_url=(source.url if source else claim.citation_source_url),
                text=claim.evidence_quote,
                excerpt_role=_excerpt_role_from_claim(claim),
                selection_method="llm_selected",
                confidence="medium",
                explanation="Evidence quote selected during source support checking.",
            )
        )
    deduped: list[EvidenceExcerpt] = []
    seen: set[str] = set()
    for excerpt in excerpts:
        key = " ".join((excerpt.text or "").split())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(excerpt)
        if len(deduped) >= 3:
            break
    return deduped


def _excerpt_role_from_claim(claim: Claim) -> str:
    if claim.support_relation == SupportRelation.DIRECTLY_SUPPORTS:
        return "direct_support"
    if claim.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS:
        return "fact_premise_support"
    if claim.support_relation in {SupportRelation.PARTIALLY_SUPPORTS, SupportRelation.SUPPORTS_WEAKER_CLAIM}:
        return "partial_or_nuanced_support"
    if claim.support_relation == SupportRelation.OPINION_ONLY:
        return "opinion_statement"
    if claim.support_relation == SupportRelation.CONTRADICTS:
        return "contradicting_excerpt"
    return "closest_available_context"


def _claim_source_comparison(
    *,
    claim: Claim,
    terminal_class: TerminalClass,
    terminal_reason: str,
    unresolved_reason: UnresolvedReason | None,
    source: Source | None,
    best_excerpt: EvidenceExcerpt | None,
) -> ClaimSourceComparison:
    claim_text = display_claim_text_for_claim(claim)
    source_text = best_excerpt.text if best_excerpt else ""
    if terminal_class == TerminalClass.FACT:
        return ClaimSourceComparison(
            claim_says=claim_text,
            source_says=source_text,
            comparison_label="supports",
            gap="source 直接或充分支持该 cited statement。",
        )
    if terminal_class == TerminalClass.OPINION:
        if best_excerpt and best_excerpt.excerpt_role in {"fact_premise_support", "direct_support", "partial_or_nuanced_support"}:
            return ClaimSourceComparison(
                claim_says=claim_text,
                source_says=source_text,
                comparison_label="supports_fact_premise",
                gap="source 支持事实前提，但 cited statement 包含作者的分析、判断或建议。",
            )
        return ClaimSourceComparison(
            claim_says=claim_text,
            source_says=source_text,
            comparison_label="opinion_only",
            gap="source 本身是观点或分析，不是事实性来源。",
        )
    if terminal_class == TerminalClass.MISMATCH:
        if claim.support_relation == SupportRelation.CONTRADICTS or (best_excerpt and best_excerpt.excerpt_role == "contradicting_excerpt"):
            return ClaimSourceComparison(
                claim_says=claim_text,
                source_says=source_text,
                comparison_label="contradicts",
                gap="source 片段与 cited statement 存在矛盾。",
            )
        if best_excerpt:
            return ClaimSourceComparison(
                claim_says=claim_text,
                source_says=source_text,
                comparison_label="weaker_than_claim",
                gap="source 片段没有支持 cited statement，或只支持较弱版本。",
            )
        return ClaimSourceComparison(
            claim_says=claim_text,
            source_says="",
            comparison_label="no_relevant_excerpt",
            gap="source 可访问，但没有找到能支撑该句的片段。",
        )
    if unresolved_reason == UnresolvedReason.NO_RELEVANT_SNIPPET:
        return ClaimSourceComparison(
            claim_says=claim_text,
            source_says="",
            comparison_label="no_relevant_excerpt",
            gap="source 可访问，但当前没有找到能对应该 cited statement 的片段。",
        )
    if source is None or unresolved_reason in {
        UnresolvedReason.NO_SOURCE_URL,
        UnresolvedReason.SOURCE_FETCH_FAILED,
        UnresolvedReason.SOURCE_BODY_MISSING,
    }:
        return ClaimSourceComparison(
            claim_says=claim_text,
            source_says="",
            comparison_label="source_unavailable",
            gap="source body 不可用，无法显示对应片段。",
        )
    return ClaimSourceComparison(
        claim_says=claim_text,
        source_says=source_text,
        comparison_label="source_unavailable" if not source_text else "no_relevant_excerpt",
        gap=f"无法生成 claim/source 对比。terminal reason: {terminal_reason}",
    )


def _claim_has_fact_source_support(claim: Claim, source: Source | None) -> bool:
    if claim.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS:
        return False
    if any(edge.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS for edge in claim.evidence_chain):
        return False
    return (
        _source_is_fact_like(source)
        or claim.source_role_for_claim in FACT_SOURCE_ROLES_FOR_CLAIM
        or claim.support_scope in FACT_SUPPORT_SCOPES
        or any(edge.support_scope in FACT_SUPPORT_SCOPES for edge in claim.evidence_chain)
    )


def _source_is_fact_like(source: Source | None) -> bool:
    if not source:
        return False
    return source.source_type in FACT_SOURCE_TYPES


def _attribution_lands_on_opinion(claim: Claim, source: Source | None) -> bool:
    if claim.claim_type == ClaimType.JUDGMENT:
        return True
    if source and source.source_type == SourceType.OPINION_ANALYSIS:
        return True
    if claim.source_role == SourceRole.OPINION_OR_ANALYSIS:
        return True
    return claim.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS


def _unresolved_reason(claim: Claim, source: Source | None) -> UnresolvedReason:
    if _is_cited_span_parse_error(display_claim_text_for_claim(claim)):
        return UnresolvedReason.CITED_SPAN_PARSE_ERROR
    if claim.support_relation == SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET:
        return UnresolvedReason.NO_RELEVANT_SNIPPET
    if source is None:
        if not claim.citation_source_url:
            return UnresolvedReason.NO_SOURCE_URL
        return UnresolvedReason.SOURCE_FETCH_FAILED
    if source.access_status != AccessStatus.ACCESSIBLE:
        return UnresolvedReason.SOURCE_FETCH_FAILED
    if not source.extracted_text:
        return UnresolvedReason.SOURCE_BODY_MISSING
    return UnresolvedReason.NO_RELEVANT_SNIPPET


def _is_cited_span_parse_error(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text or "")
    if not cleaned:
        return True
    return bool(re.fullmatch(r"[*_`~]+", cleaned))


def _is_opinion_or_analysis_claim(claim: Claim) -> bool:
    if claim.claim_type == ClaimType.JUDGMENT:
        return True
    if claim.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS:
        return True
    if any(edge.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS for edge in claim.evidence_chain):
        return True
    text = display_claim_text_for_claim(claim).lower()
    return any(
        term in text
        for term in [
            "更适合",
            "更强",
            "硬实力",
            "影响",
            "会让",
            "以为",
            "值得",
            "说明",
            "意味着",
            "建议",
            "风险",
            "长期持有",
            "opinion",
            "analysis",
            "suggest",
            "should",
            "risk",
            "impact",
        ]
    )


def _citation_path_node(claim: Claim) -> dict:
    return {
        "id": f"citation:{claim.claim_id}",
        "type": "citation_group",
        "label": f"[{claim.citation_label}]" if claim.citation_label else "引用",
        "cited_text": display_claim_text_for_claim(claim),
        "claim_id": claim.claim_id,
    }


def _source_path_node(source: Source | None, claim: Claim | None, *, upstream: bool = False) -> dict:
    if source:
        return {
            "id": f"source:{source.source_id}",
            "type": "source",
            "label": source.title or source.publisher_or_author or source.url or source.source_id,
            "source_id": source.source_id,
            "source_url": source.url,
            "source_type": source.source_type.value,
            "upstream": upstream,
        }
    return {
        "id": _source_node_id(None, claim),
        "type": "source",
        "label": (claim.citation_source_title or claim.citation_source_url or "未解析来源") if claim else "未解析来源",
        "source_url": claim.citation_source_url if claim else None,
        "source_type": SourceType.UNKNOWN.value,
        "upstream": upstream,
    }


def _source_node_id(source: Source | None, claim: Claim | None) -> str:
    if source:
        return f"source:{source.source_id}"
    if claim and claim.citation_source_url:
        return f"source:url:{claim.citation_source_url}"
    return f"source:missing:{claim.claim_id if claim else 'unknown'}"


def _last_source_node_id(path_nodes: list[dict]) -> str:
    for node in reversed(path_nodes):
        if node.get("type") == "source":
            return node["id"]
    return "document"


def _terminal_node_type(terminal_class: TerminalClass) -> str:
    return {
        TerminalClass.FACT: "terminal_fact",
        TerminalClass.OPINION: "terminal_opinion",
        TerminalClass.UNRESOLVED: "terminal_unresolved",
        TerminalClass.MISMATCH: "terminal_mismatch",
    }[terminal_class]


def _terminal_edge_type(terminal_class: TerminalClass) -> str:
    return {
        TerminalClass.FACT: "lands_on_fact",
        TerminalClass.OPINION: "lands_on_opinion",
        TerminalClass.UNRESOLVED: "unresolved",
        TerminalClass.MISMATCH: "mismatch",
    }[terminal_class]


def _terminal_label(terminal_class: TerminalClass) -> str:
    return {
        TerminalClass.FACT: "事实终点",
        TerminalClass.OPINION: "观点终点",
        TerminalClass.UNRESOLVED: "无法审计",
        TerminalClass.MISMATCH: "引用错配",
    }[terminal_class]


def _short_explanation(terminal_class: TerminalClass, reason: str) -> str:
    explanations = {
        TerminalClass.FACT: "该引用最终落到良好定义的事实来源。",
        TerminalClass.OPINION: "该引用最终停在观点、评论或分析来源。",
        TerminalClass.UNRESOLVED: "本轮没有足够来源正文或引用信息来判断最终落点。",
        TerminalClass.MISMATCH: "来源可审计，但没有支撑被引用陈述或与之矛盾。",
    }
    return f"{explanations[terminal_class]} ({reason})"


def _terminal_debug_tags(claim: Claim) -> list[str]:
    tags = [
        _value(claim.claim_type),
        _value(claim.support_relation),
        _value(claim.support_scope),
        _value(claim.source_role_for_claim),
    ]
    tags.extend(_value(flag) for flag in claim.risk_flags)
    return [tag for tag in dict.fromkeys(tags) if tag]


def _ratio(count: int, denom: int) -> float:
    return round(count / denom, 4) if denom else 0.0


def _dedupe(values: list[str]) -> list[str]:
    return [value for value in dict.fromkeys(values) if value]


def _value(item) -> str:
    if item is None:
        return ""
    return getattr(item, "value", str(item))
