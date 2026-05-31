from __future__ import annotations

from .display_status_mapper import display_claim_text_for_claim, map_claim_to_display_result
from .schemas import (
    AccessStatus,
    Claim,
    DisplayStatus,
    EdgeBasis,
    EvidenceGraph,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    RiskFlag,
    Source,
    SourceOpacity,
    SourceRoleForClaim,
    SupportRelation,
)


def build_evidence_graphs(claims: list[Claim], sources: list[Source] | dict[str, Source]) -> list[EvidenceGraph]:
    source_lookup = sources if isinstance(sources, dict) else {source.source_id: source for source in sources}
    return [build_evidence_graph(claim, source_lookup) for claim in claims]


def build_evidence_graph(claim: Claim, sources_by_id: dict[str, Source]) -> EvidenceGraph:
    display = map_claim_to_display_result(claim)
    nodes: dict[str, EvidenceGraphNode] = {}
    edges: list[EvidenceGraphEdge] = []

    def add_node(node: EvidenceGraphNode) -> str:
        nodes.setdefault(node.id, node)
        return node.id

    def add_edge(edge: EvidenceGraphEdge) -> None:
        if edge.id not in {existing.id for existing in edges}:
            edges.append(edge)

    graph_id = f"g-{claim.claim_id}"
    claim_node = add_node(
        EvidenceGraphNode(
            id=f"{claim.claim_id}:claim",
            type="claim",
            label=_shorten(display_claim_text_for_claim(claim), 120),
            subtitle="待审计主张",
            status=display.display_status.value,
            claim_id=claim.claim_id,
            metadata={"display_label": display.display_label},
        )
    )
    citation_label = _citation_label(claim)
    citation_node = add_node(
        EvidenceGraphNode(
            id=f"{claim.claim_id}:citation",
            type="citation",
            label=citation_label,
            subtitle="引用标记",
            status=display.display_status.value,
            claim_id=claim.claim_id,
            metadata={
                "citation_label": claim.citation_label,
                "source_url": claim.citation_source_url,
                "source_registry_entry": claim.source_registry_entry,
            },
        )
    )
    add_edge(
        EvidenceGraphEdge(
            id=f"{claim.claim_id}:claim-to-citation",
            source=claim_node,
            target=citation_node,
            label="引用",
            relation="author_cited",
            status=display.display_status.value,
            basis="explicit_citation" if claim.citation_label or claim.citation_source_url else "",
        )
    )

    if claim.evidence_chain:
        for index, evidence_edge in enumerate(claim.evidence_chain, start=1):
            source_node = _add_source_node(claim, sources_by_id, evidence_edge.source_id, index, display.display_status, add_node)
            add_edge(
                EvidenceGraphEdge(
                    id=f"{claim.claim_id}:citation-to-source:{index}",
                    source=citation_node,
                    target=source_node,
                    label="指向来源",
                    relation=_value(evidence_edge.edge_type),
                    status=display.display_status.value,
                    basis=_value(evidence_edge.basis),
                    metadata={"source_id": evidence_edge.source_id},
                )
            )
            _add_evidence_or_missing_node(
                claim,
                source_node,
                evidence_edge,
                sources_by_id.get(evidence_edge.source_id or ""),
                index,
                display.display_status,
                add_node,
                add_edge,
            )
            _add_upstream_nodes(
                claim,
                source_node,
                evidence_edge.upstream_source_ids,
                sources_by_id,
                index,
                add_node,
                add_edge,
            )
            _add_opaque_node_if_needed(claim, source_node, index, add_node, add_edge)
    else:
        source_node = _add_source_node(claim, sources_by_id, claim.citation_source_id, 1, display.display_status, add_node)
        add_edge(
            EvidenceGraphEdge(
                id=f"{claim.claim_id}:citation-to-source:1",
                source=citation_node,
                target=source_node,
                label="指向来源",
                relation="author_cited",
                status=display.display_status.value,
                basis=_value(EdgeBasis.EXPLICIT_LINK if claim.citation_source_url else EdgeBasis.NONE),
            )
        )
        _add_missing_node(claim, source_node, 1, display.display_status, "没有可用证据片段", add_node, add_edge)
        _add_opaque_node_if_needed(claim, source_node, 1, add_node, add_edge)

    return EvidenceGraph(graph_id=graph_id, claim_id=claim.claim_id, nodes=list(nodes.values()), edges=edges)


def _add_source_node(
    claim: Claim,
    sources_by_id: dict[str, Source],
    source_id: str | None,
    index: int,
    status: DisplayStatus,
    add_node,
) -> str:
    source = sources_by_id.get(source_id or "")
    node_id = f"{claim.claim_id}:source:{source_id or index}"
    if source:
        label = source.title or source.publisher_or_author or source.url or source.source_id
        subtitle = source.url or source.publisher_or_author or "来源"
        metadata = {
            "source_id": source.source_id,
            "url": source.url,
            "access_status": _value(source.access_status),
            "source_type": _value(source.source_type),
            "publisher_or_author": source.publisher_or_author,
        }
    else:
        label = claim.citation_source_title or claim.citation_source_url or "未解析来源"
        subtitle = claim.citation_source_url or "来源"
        metadata = {
            "source_id": source_id,
            "url": claim.citation_source_url,
            "source_registry_entry": claim.source_registry_entry,
        }
    return add_node(
        EvidenceGraphNode(
            id=node_id,
            type="source",
            label=_shorten(label, 90),
            subtitle=_shorten(subtitle, 120),
            status=status.value,
            source_id=source.source_id if source else source_id,
            claim_id=claim.claim_id,
            metadata=metadata,
        )
    )


def _add_evidence_or_missing_node(
    claim: Claim,
    source_node: str,
    evidence_edge,
    source: Source | None,
    index: int,
    status: DisplayStatus,
    add_node,
    add_edge,
) -> None:
    snippet = evidence_edge.evidence_quote or evidence_edge.evidence_span
    if snippet:
        evidence_node = add_node(
            EvidenceGraphNode(
                id=f"{claim.claim_id}:evidence:{index}",
                type="evidence",
                label=_shorten(snippet, 180),
                subtitle="证据片段",
                status=status.value,
                source_id=evidence_edge.source_id,
                claim_id=claim.claim_id,
                metadata={"reasoning_summary": evidence_edge.reasoning_summary},
            )
        )
        add_edge(
            EvidenceGraphEdge(
                id=f"{claim.claim_id}:source-to-evidence:{index}",
                source=source_node,
                target=evidence_node,
                label="支撑" if evidence_edge.support_relation == SupportRelation.DIRECTLY_SUPPORTS else "已检查",
                relation=_value(evidence_edge.support_relation),
                status=status.value,
                basis=_value(evidence_edge.basis),
                metadata={"final_bucket": _value(evidence_edge.final_bucket)},
            )
        )
        return

    if _missing_evidence_required(evidence_edge, source):
        reason = "来源正文不可用" if not source or not source.extracted_text else "没有召回到相关证据片段"
        _add_missing_node(claim, source_node, index, status, reason, add_node, add_edge)


def _add_missing_node(claim: Claim, source_node: str, index: int, status: DisplayStatus, reason: str, add_node, add_edge) -> None:
    missing_node = add_node(
        EvidenceGraphNode(
            id=f"{claim.claim_id}:missing-evidence:{index}",
            type="missing_evidence",
            label="缺少可审计证据",
            subtitle=reason,
            status=DisplayStatus.AUDIT_LIMITED.value if status != DisplayStatus.TRUE_CITATION_PROBLEM else status.value,
            claim_id=claim.claim_id,
            metadata={"reason": reason},
        )
    )
    add_edge(
        EvidenceGraphEdge(
            id=f"{claim.claim_id}:source-to-missing:{index}",
            source=source_node,
            target=missing_node,
            label="缺少证据",
            relation="missing_evidence",
            status=DisplayStatus.AUDIT_LIMITED.value,
            basis=reason,
        )
    )


def _add_upstream_nodes(
    claim: Claim,
    source_node: str,
    upstream_ids: list[str],
    sources_by_id: dict[str, Source],
    index: int,
    add_node,
    add_edge,
) -> None:
    for upstream_index, upstream_id in enumerate(upstream_ids, start=1):
        upstream = sources_by_id.get(upstream_id)
        upstream_node = add_node(
            EvidenceGraphNode(
                id=f"{claim.claim_id}:upstream:{index}:{upstream_id}",
                type="upstream_source",
                label=_shorten((upstream.title if upstream else upstream_id) or upstream_id, 90),
                subtitle=_shorten((upstream.url if upstream else "") or "上游来源", 120),
                status="info",
                source_id=upstream_id,
                claim_id=claim.claim_id,
                metadata={"url": upstream.url if upstream else None},
            )
        )
        add_edge(
            EvidenceGraphEdge(
                id=f"{claim.claim_id}:source-to-upstream:{index}:{upstream_index}",
                source=source_node,
                target=upstream_node,
                label="上游来源",
                relation="upstream_source",
                status="info",
                basis="explicit_upstream_reference",
            )
        )


def _add_opaque_node_if_needed(claim: Claim, source_node: str, index: int, add_node, add_edge) -> None:
    if not (
        claim.source_opacity
        in {
            SourceOpacity.ANONYMOUS_SOURCE,
            SourceOpacity.VAGUE_SOURCE_MENTION,
            SourceOpacity.NAMED_SECONDARY_WITH_OPAQUE_UNDERLYING,
        }
        or claim.source_role_for_claim == SourceRoleForClaim.ANONYMOUS_OR_OPAQUE_SOURCE
        or RiskFlag.ANONYMOUS_SOURCE in claim.risk_flags
        or RiskFlag.VAGUE_SOURCE in claim.risk_flags
    ):
        return
    opaque_node = add_node(
        EvidenceGraphNode(
            id=f"{claim.claim_id}:opaque-source:{index}",
            type="opaque_source",
            label="不透明上游",
            subtitle="匿名、模糊或无法公开审计的来源链",
            status="warning",
            claim_id=claim.claim_id,
            metadata={"source_opacity": _value(claim.source_opacity)},
        )
    )
    add_edge(
        EvidenceGraphEdge(
            id=f"{claim.claim_id}:source-to-opaque:{index}",
            source=source_node,
            target=opaque_node,
            label="不透明来源",
            relation="opaque_source",
            status="warning",
            basis=_value(claim.source_opacity),
        )
    )


def _missing_evidence_required(evidence_edge, source: Source | None) -> bool:
    if source is None:
        return True
    if source.access_status != AccessStatus.ACCESSIBLE:
        return True
    if not source.extracted_text:
        return True
    return evidence_edge.support_relation in {
        SupportRelation.INACCESSIBLE,
        SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
        SupportRelation.NOT_CHECKED,
        SupportRelation.NO_SUPPORT,
        SupportRelation.CONTRADICTS,
    }


def _citation_label(claim: Claim) -> str:
    if claim.citation_label:
        return f"[{claim.citation_label}]"
    if claim.citation_source_url:
        return claim.citation_source_url
    if claim.source_registry_entry:
        return _shorten(claim.source_registry_entry, 90)
    return "引用"


def _shorten(text: str | None, limit: int) -> str:
    value = " ".join((text or "").split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}…"


def _value(item) -> str:
    if item is None:
        return ""
    return getattr(item, "value", str(item))
