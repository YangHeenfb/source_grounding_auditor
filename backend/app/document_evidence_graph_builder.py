from __future__ import annotations

from .schemas import (
    CitationTerminalResult,
    DocumentEvidenceGraph,
    DocumentEvidenceGraphEdge,
    DocumentEvidenceGraphNode,
    TerminalClass,
)


TERMINAL_LABELS = {
    TerminalClass.FACT: "事实终点",
    TerminalClass.OPINION: "观点终点",
    TerminalClass.UNRESOLVED: "无法审计",
    TerminalClass.MISMATCH: "引用错配",
}

GROUP_LABELS = {
    TerminalClass.FACT: "落到事实的引用",
    TerminalClass.OPINION: "停在观点的引用",
    TerminalClass.UNRESOLVED: "无法审计的引用",
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


def build_document_evidence_graph(results: list[CitationTerminalResult]) -> DocumentEvidenceGraph:
    node_state: dict[str, dict] = {}
    edge_state: dict[str, dict] = {}

    def touch_node(
        node_id: str,
        *,
        node_type: str,
        label: str,
        terminal_class: TerminalClass | None = None,
        source_id: str | None = None,
        result: CitationTerminalResult | None = None,
        metadata: dict | None = None,
        increment: bool = True,
    ) -> None:
        if node_id not in node_state:
            node_state[node_id] = {
                "id": node_id,
                "type": node_type,
                "label": label,
                "count": 0,
                "terminal_class": terminal_class,
                "source_id": source_id,
                "metadata": metadata or {},
                "cited_texts": [],
                "citation_ids": [],
            }
        if increment:
            node_state[node_id]["count"] += 1
        if result is not None:
            node_state[node_id]["cited_texts"].append(result.cited_text)
            node_state[node_id]["citation_ids"].append(result.citation_id)

    def touch_edge(
        edge_id: str,
        *,
        source: str,
        target: str,
        edge_type: str,
        label: str,
        result: CitationTerminalResult | None = None,
    ) -> None:
        if edge_id not in edge_state:
            edge_state[edge_id] = {
                "id": edge_id,
                "source": source,
                "target": target,
                "type": edge_type,
                "label": label,
                "count": 0,
                "cited_texts": [],
                "citation_ids": [],
            }
        edge_state[edge_id]["count"] += 1
        if result is not None:
            edge_state[edge_id]["cited_texts"].append(result.cited_text)
            edge_state[edge_id]["citation_ids"].append(result.citation_id)

    touch_node(
        "document",
        node_type="document",
        label="文档",
        metadata={"description": "全部带引用内容"},
        increment=False,
    )
    for result in results:
        group_id = f"group:{result.terminal_class.value}"
        terminal_id = f"terminal:{result.terminal_class.value}"
        source_id = _source_node_id(result)
        source_label = result.source_title or result.source_url or "未解析来源"

        touch_node("document", node_type="document", label="文档", result=result)
        touch_node(
            group_id,
            node_type="citation_group",
            label=GROUP_LABELS[result.terminal_class],
            terminal_class=result.terminal_class,
            result=result,
        )
        touch_edge(
            f"document->{group_id}",
            source="document",
            target=group_id,
            edge_type="cites",
            label="引用",
            result=result,
        )
        touch_node(
            source_id,
            node_type="source",
            label=source_label,
            source_id=_path_source_id(result),
            result=result,
            metadata={"source_url": result.source_url, "source_title": result.source_title},
        )
        touch_edge(
            f"{group_id}->{source_id}",
            source=group_id,
            target=source_id,
            edge_type="points_to_source",
            label="指向来源",
            result=result,
        )
        upstream_source_id = _first_upstream_source_node_id(result)
        if upstream_source_id:
            upstream_label = _node_label(result, upstream_source_id)
            touch_node(
                upstream_source_id,
                node_type="source",
                label=upstream_label,
                source_id=upstream_source_id.removeprefix("source:"),
                result=result,
            )
            touch_edge(
                f"{source_id}->{upstream_source_id}",
                source=source_id,
                target=upstream_source_id,
                edge_type="source_cites_upstream",
                label="上游来源",
                result=result,
            )
            terminal_source_id = upstream_source_id
        else:
            terminal_source_id = source_id

        touch_node(
            terminal_id,
            node_type=TERMINAL_NODE_TYPES[result.terminal_class],
            label=TERMINAL_LABELS[result.terminal_class],
            terminal_class=result.terminal_class,
            result=result,
        )
        edge_type = TERMINAL_EDGE_TYPES[result.terminal_class]
        touch_edge(
            f"{terminal_source_id}->{terminal_id}",
            source=terminal_source_id,
            target=terminal_id,
            edge_type=edge_type,
            label=TERMINAL_LABELS[result.terminal_class],
            result=result,
        )

    nodes = [
        DocumentEvidenceGraphNode(
            id=node["id"],
            type=node["type"],
            label=node["label"],
            count=node["count"],
            terminal_class=node["terminal_class"],
            source_id=node["source_id"],
            metadata={
                **node["metadata"],
                "cited_texts": _dedupe(node["cited_texts"]),
                "citation_ids": _dedupe(node["citation_ids"]),
            },
        )
        for node in node_state.values()
    ]
    edges = [
        DocumentEvidenceGraphEdge(
            id=edge["id"],
            source=edge["source"],
            target=edge["target"],
            type=edge["type"],
            label=edge["label"],
            count=edge["count"],
            metadata={
                "cited_texts": _dedupe(edge["cited_texts"]),
                "citation_ids": _dedupe(edge["citation_ids"]),
            },
        )
        for edge in edge_state.values()
    ]
    return DocumentEvidenceGraph(nodes=nodes, edges=edges)


def _source_node_id(result: CitationTerminalResult) -> str:
    if result.source_url:
        return f"source:url:{result.source_url}"
    if result.source_title:
        return f"source:title:{_slug(result.source_title)}"
    return f"source:missing:{result.citation_id}"


def _path_source_id(result: CitationTerminalResult) -> str | None:
    for node in result.path_nodes:
        if node.get("type") == "source" and node.get("source_id"):
            return str(node["source_id"])
    return None


def _first_upstream_source_node_id(result: CitationTerminalResult) -> str | None:
    for node in result.path_nodes:
        if node.get("type") == "source" and node.get("upstream"):
            return str(node["id"])
    return None


def _node_label(result: CitationTerminalResult, node_id: str) -> str:
    for node in result.path_nodes:
        if node.get("id") == node_id:
            return str(node.get("label") or node_id)
    return node_id


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")[:80] or "source"


def _dedupe(values: list[str]) -> list[str]:
    return [value for value in dict.fromkeys(values) if value]
