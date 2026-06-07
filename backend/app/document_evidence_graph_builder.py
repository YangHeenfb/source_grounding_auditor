from __future__ import annotations

from .schemas import (
    CitationTerminalResult,
    DocumentEvidenceGraph,
    DocumentEvidenceGraphEdge,
    DocumentEvidenceGraphNode,
    TerminalClass,
    UnresolvedReason,
)


TERMINAL_LABELS = {
    TerminalClass.FACT: "事实终点",
    TerminalClass.OPINION: "观点终点",
    TerminalClass.UNRESOLVED: "无法审计",
    TerminalClass.MISMATCH: "引用不对应",
}

TERMINAL_NODE_TYPES = {
    TerminalClass.FACT: "terminal_fact",
    TerminalClass.OPINION: "terminal_opinion",
    TerminalClass.UNRESOLVED: "terminal_unresolved",
    TerminalClass.MISMATCH: "terminal_mismatch",
}

TERMINAL_EDGE_TYPES = {
    TerminalClass.FACT: "lands_on_fact",
    TerminalClass.OPINION: "lands_on_opinion",
    TerminalClass.UNRESOLVED: "unresolved",
    TerminalClass.MISMATCH: "mismatch",
}

UNRESOLVED_REASON_LABELS = {
    UnresolvedReason.NO_SOURCE_URL: "缺少 URL",
    UnresolvedReason.SOURCE_FETCH_FAILED: "抓取失败",
    UnresolvedReason.SOURCE_BODY_MISSING: "来源正文为空",
    UnresolvedReason.NO_RELEVANT_SNIPPET: "没有证据片段",
    UnresolvedReason.CITED_SPAN_PARSE_ERROR: "citation span 解析失败",
    UnresolvedReason.TERMINAL_MAPPING_MISSING: "terminal mapping 缺口",
}

DEFAULT_EXPANDED = {
    TerminalClass.FACT: False,
    TerminalClass.OPINION: True,
    TerminalClass.UNRESOLVED: False,
    TerminalClass.MISMATCH: True,
}


def build_document_evidence_graph(results: list[CitationTerminalResult]) -> DocumentEvidenceGraph:
    nodes: list[DocumentEvidenceGraphNode] = []
    edges: list[DocumentEvidenceGraphEdge] = []
    grouped = {terminal_class: [] for terminal_class in TerminalClass}
    for result in results:
        grouped[result.terminal_class].append(result)

    for terminal_class in [
        TerminalClass.MISMATCH,
        TerminalClass.OPINION,
        TerminalClass.FACT,
        TerminalClass.UNRESOLVED,
    ]:
        terminal_results = grouped[terminal_class]
        if not terminal_results:
            continue
        group_id = f"terminal:{terminal_class.value}"
        nodes.append(
            DocumentEvidenceGraphNode(
                id=group_id,
                type=TERMINAL_NODE_TYPES[terminal_class],
                label=TERMINAL_LABELS[terminal_class],
                count=len(terminal_results),
                terminal_class=terminal_class,
                metadata={
                    "default_expanded": DEFAULT_EXPANDED[terminal_class],
                    "item_count": len(terminal_results),
                },
            )
        )
        for result in terminal_results:
            statement_id = f"statement:{result.citation_id}"
            source_id = f"source:{result.citation_id}"
            reason_id = f"reason:{result.citation_id}"
            nodes.extend(
                [
                    DocumentEvidenceGraphNode(
                        id=statement_id,
                        type="cited_statement",
                        label=_short_text(result.cited_text),
                        count=1,
                        terminal_class=result.terminal_class,
                        metadata={
                            "cited_text": result.cited_text,
                            "citation_label": result.citation_label,
                            "terminal_reason": result.terminal_reason,
                        },
                    ),
                    DocumentEvidenceGraphNode(
                        id=source_id,
                        type="source",
                        label=_source_label(result),
                        count=1,
                        terminal_class=result.terminal_class,
                        metadata={
                            "source_url": result.source_url,
                            "source_title": result.source_title,
                            "citation_label": result.citation_label,
                        },
                    ),
                    DocumentEvidenceGraphNode(
                        id=reason_id,
                        type="reason",
                        label=_reason_label(result),
                        count=1,
                        terminal_class=result.terminal_class,
                        metadata={
                            "short_explanation": result.short_explanation,
                            "terminal_reason": result.terminal_reason,
                            "unresolved_reason": result.unresolved_reason.value if result.unresolved_reason else None,
                        },
                    ),
                ]
            )
            edges.extend(
                [
                    DocumentEvidenceGraphEdge(
                        id=f"{group_id}->{statement_id}",
                        source=group_id,
                        target=statement_id,
                        type="terminal_contains_statement",
                        label="包含陈述",
                        count=1,
                    ),
                    DocumentEvidenceGraphEdge(
                        id=f"{statement_id}->{source_id}",
                        source=statement_id,
                        target=source_id,
                        type="points_to_source",
                        label="指向来源",
                        count=1,
                    ),
                    DocumentEvidenceGraphEdge(
                        id=f"{source_id}->{reason_id}",
                        source=source_id,
                        target=reason_id,
                        type=TERMINAL_EDGE_TYPES[result.terminal_class],
                        label=TERMINAL_LABELS[result.terminal_class],
                        count=1,
                    ),
                ]
            )
    return DocumentEvidenceGraph(nodes=nodes, edges=edges)


def _source_label(result: CitationTerminalResult) -> str:
    return result.source_title or _domain(result.source_url or "") or "未解析来源"


def _reason_label(result: CitationTerminalResult) -> str:
    if result.terminal_class == TerminalClass.FACT:
        return _evidence_label(result) or "官方事实来源支持"
    if result.terminal_class == TerminalClass.OPINION:
        return result.claim_source_comparison.gap if result.claim_source_comparison else "最终停在分析或判断来源"
    if result.terminal_class == TerminalClass.UNRESOLVED:
        if result.claim_source_comparison and result.claim_source_comparison.gap:
            return result.claim_source_comparison.gap
        if result.unresolved_reason:
            return UNRESOLVED_REASON_LABELS.get(result.unresolved_reason, result.unresolved_reason.value)
        return "无法完成本轮审计"
    return result.claim_source_comparison.gap if result.claim_source_comparison else "source 不支持该陈述"


def _evidence_label(result: CitationTerminalResult) -> str:
    if result.best_evidence_excerpt and result.best_evidence_excerpt.text:
        return _short_text(result.best_evidence_excerpt.text, limit=90)
    for node in reversed(result.path_nodes):
        if node.get("type") == "evidence":
            label = str(node.get("label") or "")
            if label:
                return _short_text(label, limit=90)
    return ""


def _short_text(value: str, *, limit: int = 90) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _domain(url: str) -> str:
    if "://" not in url:
        return url
    return url.split("://", 1)[1].split("/", 1)[0]
