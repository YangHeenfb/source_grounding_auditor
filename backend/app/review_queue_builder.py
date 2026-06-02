from __future__ import annotations

from collections import defaultdict

from .schemas import (
    CitationTerminalResult,
    ReviewQueue,
    ReviewQueueGroup,
    ReviewQueueItem,
    TerminalClass,
    UnresolvedReason,
)


UNRESOLVED_REASON_LABELS = {
    UnresolvedReason.NO_SOURCE_URL: "缺少 URL",
    UnresolvedReason.SOURCE_FETCH_FAILED: "抓取失败",
    UnresolvedReason.SOURCE_BODY_MISSING: "来源正文为空",
    UnresolvedReason.NO_RELEVANT_SNIPPET: "没有证据片段",
    UnresolvedReason.CITED_SPAN_PARSE_ERROR: "citation span 解析失败",
    UnresolvedReason.TERMINAL_MAPPING_MISSING: "terminal mapping 缺口",
}


def build_review_queue(results: list[CitationTerminalResult]) -> ReviewQueue:
    needs_review = [
        _item(result)
        for result in sorted(results, key=_needs_review_sort_key)
        if result.terminal_class in {TerminalClass.MISMATCH, TerminalClass.OPINION}
    ]
    verified_fact = ReviewQueueGroup(
        group_id="verified_fact",
        label="事实来源",
        terminal_class=TerminalClass.FACT,
        default_collapsed=True,
        items=[_item(result) for result in results if result.terminal_class == TerminalClass.FACT],
    )

    unresolved_by_reason: dict[UnresolvedReason, list[ReviewQueueItem]] = defaultdict(list)
    for result in results:
        if result.terminal_class != TerminalClass.UNRESOLVED:
            continue
        reason = result.unresolved_reason or UnresolvedReason.TERMINAL_MAPPING_MISSING
        unresolved_by_reason[reason].append(_item(result))

    unresolved_groups = [
        ReviewQueueGroup(
            group_id=f"unresolved:{reason.value}",
            label=UNRESOLVED_REASON_LABELS.get(reason, reason.value),
            terminal_class=TerminalClass.UNRESOLVED,
            unresolved_reason=reason,
            default_collapsed=True,
            items=items,
        )
        for reason, items in sorted(unresolved_by_reason.items(), key=lambda pair: pair[0].value)
    ]
    return ReviewQueue(
        needs_review=needs_review,
        verified_fact=verified_fact,
        unresolved=unresolved_groups,
    )


def _item(result: CitationTerminalResult) -> ReviewQueueItem:
    return ReviewQueueItem(
        citation_id=result.citation_id,
        cited_text=result.cited_text,
        citation_label=result.citation_label,
        source_title=result.source_title,
        source_url=result.source_url,
        terminal_class=result.terminal_class,
        short_explanation=result.short_explanation,
        terminal_reason=result.terminal_reason,
        unresolved_reason=result.unresolved_reason,
    )


def _needs_review_sort_key(result: CitationTerminalResult) -> tuple[int, str]:
    if result.terminal_class == TerminalClass.MISMATCH:
        return (0, result.citation_id)
    if result.terminal_class == TerminalClass.OPINION:
        return (1, result.citation_id)
    return (2, result.citation_id)
