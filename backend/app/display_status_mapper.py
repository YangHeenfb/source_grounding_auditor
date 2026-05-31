from __future__ import annotations

from .schemas import (
    Claim,
    ClaimReviewCategory,
    ClaimType,
    DisplayCitationResult,
    DisplayStatus,
    DiscourseRole,
    FinalGroundingBucket,
    RiskFlag,
    SupportRelation,
    SupportScope,
)


AUDIT_LIMITED_FLAGS = {
    RiskFlag.INACCESSIBLE_SOURCE,
    RiskFlag.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
    RiskFlag.SOURCE_FETCH_FAILED,
    RiskFlag.SOURCE_BODY_MISSING,
}

EXCLUDED_ROLES = {
    DiscourseRole.CAVEAT_OR_LIMITATION,
    DiscourseRole.UNSUPPORTED_EXAMPLE,
    DiscourseRole.SOURCE_POINTER,
    DiscourseRole.USER_QUESTION,
    DiscourseRole.SECTION_HEADING,
    DiscourseRole.CONTEXT_OR_TRANSITION,
}

DISPLAY_LABELS = {
    DisplayStatus.VERIFIED_FACT_SUPPORT: "事实支撑成立",
    DisplayStatus.PARTIAL_OR_WEAK_SUPPORT: "部分或弱支撑",
    DisplayStatus.ATTRIBUTION_SUPPORT: "归属支撑",
    DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES: "分析判断",
    DisplayStatus.AUDIT_LIMITED: "审计受限",
    DisplayStatus.TRUE_CITATION_PROBLEM: "引用问题",
    DisplayStatus.EXCLUDED_OR_CONTEXT: "已排除",
}

DISPLAY_EXPLANATIONS = {
    DisplayStatus.VERIFIED_FACT_SUPPORT: "引用来源中找到了与该 claim 对应的证据，事实支撑较直接。",
    DisplayStatus.PARTIAL_OR_WEAK_SUPPORT: "引用来源提供了部分支撑，或只支持一个更弱的说法。",
    DisplayStatus.ATTRIBUTION_SUPPORT: "引用来源支持“某来源说过这件事”，但这不等于被转述内容本身已被一手证明。",
    DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES: "引用来源支撑了分析判断所依赖的事实前提，但没有直接证明该判断本身。",
    DisplayStatus.AUDIT_LIMITED: "本轮没有取得可审计的来源正文或相关证据片段，因此不能完成支撑关系判断。",
    DisplayStatus.TRUE_CITATION_PROBLEM: "可用证据没有支撑该 claim，或与 claim 表述存在明确冲突。",
    DisplayStatus.EXCLUDED_OR_CONTEXT: "这部分不是作者实际主张，或只是限制说明、来源指针、问题、标题、过渡或反例。",
}

SEVERITY_BY_STATUS = {
    DisplayStatus.VERIFIED_FACT_SUPPORT: "ok",
    DisplayStatus.PARTIAL_OR_WEAK_SUPPORT: "low",
    DisplayStatus.ATTRIBUTION_SUPPORT: "info",
    DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES: "info",
    DisplayStatus.AUDIT_LIMITED: "warning",
    DisplayStatus.TRUE_CITATION_PROBLEM: "problem",
    DisplayStatus.EXCLUDED_OR_CONTEXT: "muted",
}


def map_claim_to_display_result(claim: Claim) -> DisplayCitationResult:
    status, reason = _display_status_for_claim(claim)
    return DisplayCitationResult(
        claim_id=claim.claim_id,
        display_claim_text=display_claim_text_for_claim(claim),
        display_status=status,
        display_label=DISPLAY_LABELS[status],
        display_explanation=DISPLAY_EXPLANATIONS[status],
        severity=SEVERITY_BY_STATUS[status],
        primary_reason=reason,
        debug_tags=_debug_tags_for_claim(claim),
        should_show_in_problematic=status == DisplayStatus.TRUE_CITATION_PROBLEM,
        should_count_as_true_mismatch=status == DisplayStatus.TRUE_CITATION_PROBLEM,
    )


def map_claims_to_display_results(claims: list[Claim]) -> list[DisplayCitationResult]:
    return [map_claim_to_display_result(claim) for claim in claims]


def display_claim_text_for_claim(claim: Claim) -> str:
    """Use the user's original wording for display; keep normalized_claim for debug."""
    return (
        (claim.original_span or "").strip()
        or (claim.original_text_span or "").strip()
        or (claim.normalized_claim or "").strip()
        or claim.claim_id
    )


def _display_status_for_claim(claim: Claim) -> tuple[DisplayStatus, str]:
    if (
        claim.review_category == ClaimReviewCategory.EXCLUDED_OR_CONTEXT
        or claim.claim_type == ClaimType.NON_CLAIM
        or claim.not_asserted_by_author
        or claim.discourse_role in EXCLUDED_ROLES
        or claim.final_bucket == FinalGroundingBucket.EXCLUDED_OR_CONTEXT
    ):
        return DisplayStatus.EXCLUDED_OR_CONTEXT, "excluded_or_context"

    if _is_true_citation_problem(claim):
        return DisplayStatus.TRUE_CITATION_PROBLEM, "source_does_not_support_claim"

    if _is_audit_limited(claim):
        return DisplayStatus.AUDIT_LIMITED, "source_access_or_snippet_limited"

    if _is_analysis_from_sourced_premises(claim):
        return DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES, "premise_support_for_analysis"

    if claim.support_relation == SupportRelation.ATTRIBUTION_ONLY:
        return DisplayStatus.ATTRIBUTION_SUPPORT, "attribution_preserved"

    if (
        claim.support_relation == SupportRelation.DIRECTLY_SUPPORTS
        and claim.final_bucket == FinalGroundingBucket.HARD_FACT_GROUNDING
    ):
        return DisplayStatus.VERIFIED_FACT_SUPPORT, "direct_fact_support"

    if claim.support_relation in {
        SupportRelation.PARTIALLY_SUPPORTS,
        SupportRelation.SUPPORTS_WEAKER_CLAIM,
        SupportRelation.OPINION_ONLY,
        SupportRelation.BACKGROUND_ONLY,
    }:
        return DisplayStatus.PARTIAL_OR_WEAK_SUPPORT, "partial_or_weaker_support"

    if claim.review_category == ClaimReviewCategory.ATTRIBUTION_SUPPORTED:
        return DisplayStatus.ATTRIBUTION_SUPPORT, "attribution_supported_by_review"

    if claim.review_category == ClaimReviewCategory.AUDIT_LIMITED:
        return DisplayStatus.AUDIT_LIMITED, "audit_limited_by_review"

    if claim.final_bucket == FinalGroundingBucket.HARD_FACT_GROUNDING:
        return DisplayStatus.VERIFIED_FACT_SUPPORT, "hard_fact_grounding"

    if claim.final_bucket in {
        FinalGroundingBucket.WEAK_FACT_GROUNDING,
        FinalGroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING,
    }:
        return DisplayStatus.PARTIAL_OR_WEAK_SUPPORT, "non_problematic_grounding"

    return DisplayStatus.AUDIT_LIMITED, "not_enough_user_facing_evidence"


def _is_true_citation_problem(claim: Claim) -> bool:
    has_problem_relation = claim.support_relation in {
        SupportRelation.NO_SUPPORT,
        SupportRelation.CONTRADICTS,
    }
    has_mismatch_flag = RiskFlag.SOURCE_CLAIM_MISMATCH in claim.risk_flags
    if not (has_problem_relation or has_mismatch_flag):
        return False
    return _has_usable_evidence_snippet(claim)


def _is_audit_limited(claim: Claim) -> bool:
    if claim.support_relation in {
        SupportRelation.INACCESSIBLE,
        SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
        SupportRelation.NOT_CHECKED,
    }:
        return True
    flags = set(claim.risk_flags)
    if flags and flags.issubset(AUDIT_LIMITED_FLAGS):
        return True
    if (
        claim.support_relation in {SupportRelation.NO_SUPPORT, SupportRelation.CONTRADICTS}
        and not _has_usable_evidence_snippet(claim)
    ):
        return True
    return any(
        edge.support_relation
        in {
            SupportRelation.INACCESSIBLE,
            SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
        }
        for edge in claim.evidence_chain
    ) and not _has_usable_evidence_snippet(claim)


def _is_analysis_from_sourced_premises(claim: Claim) -> bool:
    if claim.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS:
        return True
    if any(edge.support_scope == SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS for edge in claim.evidence_chain):
        return True
    return claim.claim_type == ClaimType.JUDGMENT and claim.support_relation in {
        SupportRelation.PARTIALLY_SUPPORTS,
        SupportRelation.BACKGROUND_ONLY,
        SupportRelation.SUPPORTS_WEAKER_CLAIM,
    }


def _has_usable_evidence_snippet(claim: Claim) -> bool:
    if claim.evidence_quote:
        return True
    return any(
        edge.source_id
        and bool(edge.evidence_quote or edge.evidence_span)
        and edge.support_relation
        not in {
            SupportRelation.INACCESSIBLE,
            SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
        }
        for edge in claim.evidence_chain
    )


def _debug_tags_for_claim(claim: Claim) -> list[str]:
    tags = [
        _value(claim.claim_type),
        _value(claim.discourse_role),
        _value(claim.support_relation),
        _value(claim.final_bucket),
        _value(claim.review_category),
        _value(claim.source_opacity),
        _value(claim.source_role_for_claim),
        _value(claim.support_scope),
    ]
    tags.extend(_value(flag) for flag in claim.risk_flags)
    return [tag for tag in dict.fromkeys(tags) if tag]


def _value(item) -> str:
    if item is None:
        return ""
    return getattr(item, "value", str(item))
